import pickle
from collections import defaultdict
from jutils import mesh_utils, geom_utils
from pytorch3d.structures import Meshes
import os
import os.path as osp
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
import yaml
from ema_pytorch import EMA
from jutils import model_utils, plot_utils
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from tqdm import tqdm

from ..manip.model import build_model, CondGaussianDiffusion
from ..manip.data import build_dataloader
from ..visualization.rerun_visualizer import RerunVisualizer
from ..visualization.pt3d_visualizer import Pt3dVisualizer
from ..utils.motion_repr import HandWrapper
from ..manip.model.guidance_optimizer_jax import (
    do_guidance_optimization,
    se3_to_wxyz_xyz,
    wxyz_xyz_to_se3,
)

from move_utils.slurm_utils import slurm_engine

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model: CondGaussianDiffusion,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        vis_every=40000,
        save_and_sample_every=40000,
        results_folder="./results",
        use_wandb=True,
        load_data=True,
    ):
        super().__init__()
        self.opt = opt
        self.use_wandb = use_wandb

        self.hand_wrapper = HandWrapper(opt.paths.mano_dir)
        if self.use_wandb:
            save_dir = results_folder
            expname = opt.expname
            # Loggers
            os.makedirs(save_dir + "/log/wandb", exist_ok=True)

            runid = None
            if os.path.exists(f"{save_dir}/runid.txt"):
                runid = open(f"{save_dir}/runid.txt").read().strip()

            log = wandb.init(
                project="er_" + osp.dirname(expname),
                name=osp.basename(expname),
                dir=osp.join(save_dir, "log"),
                id=runid,
                save_code=True,
                settings=wandb.Settings(start_method="fork"),
                # config=opt,
                # project=opt.wandb_pj_name,
                # name=opt.exp_name,
                # dir=opt.save_dir,
            )
            runid = log.id

            def save_runid():
                with open(f"{save_dir}/runid.txt", "w") as fp:
                    fp.write(runid)

            save_runid()

        self.batch_size = train_batch_size
        self.model = diffusion_model

        if load_data:
            self.prep_dataloader(window_size=opt.model.window)
        
        diffusion_model.to(device)
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.vis_every = vis_every

        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp  # BUG: amp on gives nan loss??
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt

        self.data_root_folder = getattr(self.opt, "data_root_folder", "data")

        self.window = opt.model.window

        self.use_object_split = getattr(self.opt, "use_object_split", False)

        # self.bm_dict = self.ds.bm_dict

        self.test_on_train = getattr(self.opt, "test_sample_res_on_train", False)

        self.add_hand_processing = getattr(self.opt, "add_hand_processing", False)

        self.for_quant_eval = getattr(self.opt, "for_quant_eval", False)

        self.visualizer = RerunVisualizer(
            exp_name=opt.expname,
            save_dir=opt.exp_dir,
            enable_visualization=True,
            mano_models_dir=opt.paths.mano_dir,
            object_mesh_dir=opt.paths.object_mesh_dir,
        )
        self.viz_off = Pt3dVisualizer(
            exp_name=opt.expname,
            save_dir=opt.exp_dir,
            mano_models_dir=opt.paths.mano_dir,
            object_mesh_dir=opt.paths.object_mesh_dir,
        )

    def prep_dataloader(self, window_size):
        opt = self.opt
        if not opt.test:
            self.prep_train_dataset(opt)
            self.model.set_metadata(self.ds.stats)
            self.prep_val_dataset(opt)
        else:
            self.prep_train_dataset(opt)
            self.model.set_metadata(self.ds.stats)
            self.prep_val_dataset(opt)
            # self.model.set_metadata(self.val_ds.stats)

    def prep_train_dataset(self, opt):
        dl, ds = build_dataloader(
            opt.traindata,
            opt,
            is_train=True,
            shuffle=not opt.test,
            batch_size=self.batch_size,
            num_workers=10,
        )
        self.dl = cycle(dl)
        self.ds = ds

    def prep_val_dataset(self, opt):
        dl, ds = build_dataloader(
            opt.testdata,
            opt,
            is_train=False,
            shuffle=False,
            batch_size=1,
            num_workers=1,
        )
        self.val_dl = dl
        self.val_ds = ds

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(
            data, os.path.join(self.results_folder, "model-" + str(milestone) + ".pt")
        )

    def load(self, milestone, pretrained_path=None):
        if pretrained_path is None:
            data = torch.load(
                os.path.join(self.results_folder, "model-" + str(milestone) + ".pt")
            )
        else:
            data = torch.load(pretrained_path)

        self.step = data["step"]
        self.model.load_state_dict(data["model"], strict=False)
        self.ema.load_state_dict(data["ema"], strict=False)
        self.scaler.load_state_dict(data["scaler"])

    def prep_temporal_condition_mask(self, data, t_idx=0):
        # Missing regions are ones, the condition regions are zeros.
        mask = torch.ones_like(data).to(data.device)  # BS X T X D
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(
            data.device
        )  # BS X D

        return mask

    def prep_joint_condition_mask(self, data, joint_idx, pos_only):
        # data: BS X T X D
        # head_idx = 15
        # hand_idx = 20, 21
        # Condition part is zeros, while missing part is ones.
        mask = torch.ones_like(data).to(data.device)

        cond_pos_dim_idx = joint_idx * 3
        cond_rot_dim_idx = 24 * 3 + joint_idx * 6

        mask[:, :, cond_pos_dim_idx : cond_pos_dim_idx + 3] = torch.zeros(
            data.shape[0], data.shape[1], 3
        ).to(data.device)

        if not pos_only:
            mask[:, :, cond_rot_dim_idx : cond_rot_dim_idx + 6] = torch.zeros(
                data.shape[0], data.shape[1], 6
            ).to(data.device)

        return mask

    def train(self):
        init_step = self.step
        for idx in tqdm(range(init_step, self.train_num_steps)):
            self.optimizer.zero_grad()

            nan_exists = (
                False  # If met nan in loss or gradient, need to skip to next data.
            )
            for i in range(self.gradient_accumulate_every):
                sample = next(self.dl)
                sample = model_utils.to_cuda(sample)

                object_motion = sample["target"].cuda()
                cond = sample["condition"].cuda()
                seq_len = torch.tensor([cond.shape[1]]).to(
                    device
                )  # [1] - sequence length

                ######### add occlusion mask for traj repr, with some schedules
                # mask_prob = 0.5
                # max_infill_ratio = 0.1
                # prob = random.uniform(0, 1)
                # batch_size, clip_len, _ = cond.shape
                # if prob > 1 - mask_prob and 'traj_noisy_raw' in sample:
                #     traj_feat_dim = sample["traj_noisy_raw"].shape[-1]
                #     start = (
                #         torch.FloatTensor(batch_size).uniform_(0, clip_len - 1).long()
                #     )
                #     mask_len = (
                #         clip_len
                #         * torch.FloatTensor(batch_size).uniform_(0, 1)
                #         * max_infill_ratio
                #     ).long()
                #     end = start + mask_len
                #     end[end > clip_len] = clip_len
                #     mask_traj = torch.ones(batch_size, clip_len).to(device)  # [bs, t]
                #     for bs in range(batch_size):
                #         mask_traj[bs, start[bs] : end[bs]] = 0
                #     mask_traj_exp = mask_traj.unsqueeze(-1).repeat(
                #         1, 1, traj_feat_dim
                #     )  # [bs, t, 4]
                #     cond[:, :, -traj_feat_dim:] = (
                #         cond[:, :, -traj_feat_dim:] * mask_traj_exp
                #     )
                # else:
                # mask_traj = torch.ones(batch_size, clip_len).to(device)

                # Extract data from sample and move to device
                # Generate padding mask
                actual_seq_len = seq_len + 2
                tmp_mask = torch.arange(self.window + 2, device=device).expand(
                    1, self.window + 2
                ) < actual_seq_len[:, None].repeat(1, self.window + 2)
                padding_mask = tmp_mask[:, None, :]

                # with autocast(device_type='cuda', enabled=self.amp):
                with autocast(enabled=self.amp):
                    loss_diffusion = self.model(
                        object_motion,
                        cond,
                        padding_mask,
                        newPoints=sample["newPoints"],
                        hand_raw=sample["hand_raw"],
                    )

                    loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print("WARNING: NaN loss. Skipping to next data...")
                        nan_exists = True
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    # check gradients
                    parameters = [
                        p for p in self.model.parameters() if p.grad is not None
                    ]
                    total_norm = torch.norm(
                        torch.stack(
                            [torch.norm(p.grad.detach(), 2.0) for p in parameters]
                        ),
                        2.0,
                    )
                    if torch.isnan(total_norm):
                        print("WARNING: NaN gradients. Skipping to next data...")
                        nan_exists = True
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        log_dict = {
                            "Train/Loss/Total Loss": loss.item(),
                            "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                        }
                        wandb.log(log_dict, step=self.step)

                    if idx % 50 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            if self.step % self.opt.general.eval_every == 0:
                # evaluation step
                self.ema.ema_model.eval()
                all_metrics = defaultdict(list)

                metrics_train = self.validation_step(sample, pref=f"train/")
                self.accumulate_metrics(
                    metrics_train, all_metrics, pref="train/"
                )

                # Collect all metrics for comprehensive logging
                with torch.no_grad():
                    for b, val_data_dict in enumerate(self.val_dl):
                        if b >= 2:
                            break
                        val_data_dict = model_utils.to_cuda(val_data_dict)

                        metrics_val = self.validation_step(val_data_dict, pref=f"val/{b:03d}")
                        self.accumulate_metrics(metrics_val, all_metrics, pref="val/")

                self.log_metrics_to_wandb(
                    all_metrics,
                    self.step,
                )

            if self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

            self.step += 1

        print("training complete")

        if self.use_wandb:
            wandb.run.finish()

    def test(self):
        self.ema.ema_model.eval()
        with torch.no_grad():
            for b, val_data_dict in enumerate(self.val_dl):
                val_data_dict = model_utils.to_cuda(val_data_dict)

                # baseline
                bs_for_vis = 1

                # if 'traj_noisy_raw' in val_data_dict:
                #     x = val_data_dict["traj_noisy_raw"]
                # else:
                #     x = val_data_dict["traj_raw"]
                # obs = self.model.get_obs(guide=True, batch=val_data_dict, shape=x.shape)

                # # x_opt = self.model.guide_jax(x, model_kwargs={'obs': obs})
                # x_opt, _ = do_guidance_optimization(
                #     traj=se3_to_wxyz_xyz(x),
                #     obs=obs,
                #     guidance_mode=self.opt.guide.hint,
                #     phase="fp",
                #     verbose=True,
                #     # verbose=False,
                # )

                # points = val_data_dict["newPoints"][:bs_for_vis]
                # points = plot_utils.pc_to_cubic_meshes(points[:, :vis_P])

                # hand_poses_raw = val_data_dict["hand_raw"][
                #     :bs_for_vis
                # ]  # [1, T, 2*D] - left + right hand
                # object_motion_raw = val_data_dict["target_raw"][:bs_for_vis]
                # left_hand_raw, right_hand_raw = torch.split(
                #     hand_poses_raw, 21 * 3, dim=-1
                # )
                # print(
                #     "x_opt",
                #     x_opt.shape,
                #     x.shape,
                #     object_motion_raw.shape,
                #     left_hand_raw.shape,
                #     right_hand_raw.shape,
                # )
                # fname = self.gen_vis_res(
                #     self.step,
                #     left_hand_raw,
                #     right_hand_raw,
                #     object_motion_raw,
                #     object_noisy=x[:bs_for_vis],
                #     object_pred=x_opt[:bs_for_vis],
                #     object_id=points,  # val_data_dict["object_id"][bs_for_vis-1],
                #     seq_len=x_opt.shape[1],
                #     pref=f"{self.opt.test_folder}/{b}_sample_fp_",
                # )

                for s in range(3):
                    _, (pred, _) = self.validation_step(
                        val_data_dict,
                        pref=f"{self.opt.test_folder}/{b}_sample_{s}_",
                        just_vis=True,
                        sample_guide=True,
                        rtn_sample=True,
                    )

                    # batch = val_data_dict
                    # fname = "outputs/tmp.pkl"
                    # if not osp.exists(fname):
                    #     print("save tmp")
                    #     x = batch["target"]
                    #     wTo_pred_se3 = self.model.denormalize_data(pred)  # (B, T, 3+6)
                    #     wTo_gt = batch["target_raw"]
                    #     with open(fname, "wb") as f:
                    #         pickle.dump(
                    #             {
                    #                 "x": wTo_pred_se3,
                    #                 "gt": wTo_gt,
                    #                 "batch": batch,
                    #             },
                    #             f,
                    #         )

            for b, val_data_dict in enumerate(self.dl):
                if b >= 10:
                    break
                val_data_dict = model_utils.to_cuda(val_data_dict)
                for s in range(2):
                    self.validation_step(
                        val_data_dict,
                        pref=f"{self.opt.test_folder}/train_{b}_sample_{s}_",
                        just_vis=True,
                        sample_guide=True,
                    )

    def accumulate_metrics(self, metrics, all_metrics, pref):
        for metric_name, metric_dict in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = defaultdict(list)
            for uid, uid_values in metric_dict.items():
                all_metrics[metric_name][uid].extend(uid_values)

    def log_metrics_to_wandb(self, metrics, step, pref=""):
        wandb_log = {}
        for metric_name, metric_dict in metrics.items():
            total_values = []
            for uid, uid_values in metric_dict.items():
                wandb_log[f"{pref}_{metric_name}/uid_{uid}"] = np.mean(uid_values)
                total_values.extend(uid_values)
            wandb_log[f"{pref}{metric_name}/avg"] = np.mean(total_values)
            wandb_log[f"{pref}{metric_name}/distribution"] = wandb.Histogram(
                total_values
            )
        if self.use_wandb:
            wandb.log(wandb_log, step=self.step)

    def validation_step(
        self,
        val_data_dict,
        pref="val/",
        just_vis=False,
        sample_guide=False,
        rtn_sample=False,
    ):
        cond = val_data_dict["condition"]
        device = cond.device
        seq_len = torch.tensor([cond.shape[1]]).to(device)

        actual_seq_len = seq_len + 1
        tmp_mask = torch.arange(self.window + 2, device=device).expand(
            1, self.window + 2
        ) < actual_seq_len[:, None].repeat(1, self.window + 2)
        padding_mask = tmp_mask[:, None, :]
        object_motion = val_data_dict["target"]
        cond = val_data_dict["condition"]

        # with autocast(device_type='cuda', enabled=self.amp):
        with autocast(enabled=self.amp):
            val_loss_diffusion = self.model(
                object_motion,
                cond,
                padding_mask=padding_mask,
                newPoints=val_data_dict["newPoints"],
                hand_raw=val_data_dict["hand_raw"],
            )

        val_loss = val_loss_diffusion
        if self.use_wandb:
            val_log_dict = {
                "Validation/Loss/Total Loss": val_loss.item(),
                "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
            }
            wandb.log(val_log_dict, step=self.step)

        bs_for_vis = len(val_data_dict["object_id"])

        sample = val_data_dict
        diffusion_model = self.ema.ema_model

        hand_poses_raw = sample["hand_raw"][
            :bs_for_vis
        ]  # [1, T, 2*D] - left + right hand
        object_motion_raw = sample["target_raw"][:bs_for_vis]
        left_hand_raw, right_hand_raw = torch.split(hand_poses_raw, 21 * 3, dim=-1)
        cond = sample["condition"][:bs_for_vis]
        seq_len = torch.tensor([cond.shape[1]])
        object_motion = sample["target"][:bs_for_vis]

        object_pred_raw, _ = diffusion_model.sample_raw(
            torch.randn_like(object_motion),
            cond,
            padding_mask=padding_mask,
            newPoints=sample["newPoints"][:bs_for_vis],
            hand_raw=sample["hand_raw"][:bs_for_vis],
        )
        metrics = self.eval_step(
            object_pred_raw, object_motion_raw, val_data_dict["object_id"]
        )

        # add guide
        guided_object_pred_raw = None
        if sample_guide:
            # directly guide object_pred_raw
            obs = diffusion_model.get_obs(
                guide=True, batch=sample, shape=object_pred_raw.shape
            )
            post_object_pred_raw, _ = do_guidance_optimization(
                traj=se3_to_wxyz_xyz(object_pred_raw),
                obs=obs,
                guidance_mode=self.opt.guide.hint,
                phase="post",
                verbose=True,
            )
            post_object_pred_raw = wxyz_xyz_to_se3(post_object_pred_raw)
            metrics_post = self.eval_step(
                post_object_pred_raw, object_motion_raw, val_data_dict["object_id"]
            )
            metrics.update({f"{k}_post": v for k, v in metrics_post.items()})

            guided_object_pred_raw, _ = diffusion_model.sample_raw(
                torch.randn_like(object_motion),
                cond,
                padding_mask=padding_mask,
                guide=True,
                obs=sample,
                newPoints=sample["newPoints"][:bs_for_vis],
                hand_raw=sample["hand_raw"][:bs_for_vis],
            )
            metrics_guided = self.eval_step(
                guided_object_pred_raw, object_motion_raw, val_data_dict["object_id"]
            )
            metrics.update({f"{k}_guided": v for k, v in metrics_guided.items()})

        if self.step % self.vis_every == 0 or just_vis:
            bs_for_vis = 1
            # if 'newPoints' in sample:
            points = val_data_dict["newPoints"][:bs_for_vis]
            points = plot_utils.pc_to_cubic_meshes(points[:, :vis_P])

            fname = self.gen_vis_res(
                self.step,
                left_hand_raw[:bs_for_vis],
                right_hand_raw[:bs_for_vis],
                object_motion_raw[:bs_for_vis],
                object_noisy=sample["traj_noisy_raw"][:bs_for_vis],
                object_pred=object_pred_raw[:bs_for_vis],
                object_id=points,  # val_data_dict["object_id"][bs_for_vis-1],
                seq_len=seq_len,
                pref=f"{pref}uid_{val_data_dict['object_id'][bs_for_vis - 1]}",
            )
            if self.use_wandb:
                object_id = val_data_dict["object_id"][bs_for_vis - 1]
                wandb.log(
                    {f"{pref}vis": wandb.Video(fname)}, step=self.step
                )

            if sample_guide:
                _ = self.gen_vis_res(
                    self.step,
                    left_hand_raw[:bs_for_vis],
                    right_hand_raw[:bs_for_vis],
                    object_motion_raw[:bs_for_vis],
                    object_pred=post_object_pred_raw[:bs_for_vis],
                    object_id=points,
                    seq_len=seq_len,
                    pref=f"{pref}uid_{val_data_dict['object_id'][bs_for_vis - 1]}_post",
                )
                fname = self.gen_vis_res(
                    self.step,
                    left_hand_raw[:bs_for_vis],
                    right_hand_raw[:bs_for_vis],
                    object_motion_raw[:bs_for_vis],
                    object_pred=guided_object_pred_raw[:bs_for_vis],
                    object_id=points,
                    seq_len=seq_len,
                    pref=f"{pref}uid_{val_data_dict['object_id'][bs_for_vis - 1]}_guided",
                )

            if self.use_wandb:
                wandb.log(
                    {f"{pref}vis_guided": wandb.Video(fname)},
                    step=self.step,
                )
        if rtn_sample:
            return metrics, (object_pred_raw, guided_object_pred_raw)
        return metrics

    def eval_step(self, pred_moton, gt_motion, object_id_list, scale=0.05):
        """_summary_

        :param pred_moton: (B, T, 3+6)
        :param gt_motion: (B, T, 3+6)
        :return: {'tsl_error': (B, ), 'rot_error': (B, ), 'total_error': (B, )}
        rot error calculation: transform xyz-axis by motion, then calculate the mean error

        """

        # pred_moton, gt_motion: (B, T, 3+6)
        # We'll compare the translation and the orientation (rot6d) by transforming canonical axes
        # Unpack translation and rotation
        pred_tsl = pred_moton[..., :3]  # (B, T, 3)
        pred_rot6d = pred_moton[..., 3:9]  # (B, T, 6)
        gt_tsl = gt_motion[..., :3]
        gt_rot6d = gt_motion[..., 3:9]

        # Compute translation error (L2 norm per frame, then mean over time)
        tsl_error = torch.norm(pred_tsl - gt_tsl, dim=-1).mean(dim=-1)  # (B,)

        # Prepare canonical axes: (4, 3): origin and 3 axes
        axes = (
            torch.eye(3, device=pred_moton.device) * scale
        )  # (3, 3)  this is like cube size 5cm
        origin = torch.zeros(1, 3, device=pred_moton.device)
        cano_points = torch.cat([origin, axes], dim=0)  # (4, 3)

        # Expand to batch and time
        B, T = pred_rot6d.shape[:2]
        cano_points = cano_points[None, None, ...].expand(B, T, 4, 3)  # (B, T, 4, 3)

        # Get rotation matrices
        pred_rotmat = geom_utils.rotation_6d_to_matrix(
            pred_rot6d.reshape(-1, 6)
        ).reshape(B, T, 3, 3)
        gt_rotmat = geom_utils.rotation_6d_to_matrix(gt_rot6d.reshape(-1, 6)).reshape(
            B, T, 3, 3
        )

        # Transform canonical points (only rotate, no translation)
        pred_points = torch.matmul(
            cano_points, pred_rotmat.transpose(-2, -1)
        )  # (B, T, 4, 3)
        gt_points = torch.matmul(
            cano_points, gt_rotmat.transpose(-2, -1)
        )  # (B, T, 4, 3)

        # Compute mean L2 error over the 4 points, per frame, then mean over time
        rot_error = (
            torch.norm(pred_points - gt_points, dim=-1).mean(dim=-1).mean(dim=-1)
        )  # (B,)

        # Total error: sum or mean of both
        # Compute total_error as the mean error of the cano_points after applying both rotation and translation,
        # using homogeneous coordinates for clarity.

        # Prepare canonical points in homogeneous coordinates: (4, 4)
        cano_points_homo = torch.cat(
            [cano_points, torch.ones_like(cano_points[..., :1])], dim=-1
        )  # (B, T, 4, 4)

        # Build homogeneous transformation matrices for pred and gt: (B, T, 4, 4)
        pred_rotmat_homo = (
            torch.eye(4, device=pred_moton.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(B, T, 1, 1)
        )
        pred_rotmat_homo[..., :3, :3] = pred_rotmat
        pred_rotmat_homo[..., :3, 3] = pred_tsl

        gt_rotmat_homo = (
            torch.eye(4, device=gt_motion.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(B, T, 1, 1)
        )
        gt_rotmat_homo[..., :3, :3] = gt_rotmat
        gt_rotmat_homo[..., :3, 3] = gt_tsl

        # Transform canonical points
        pred_points_full = (cano_points_homo @ pred_rotmat_homo.transpose(-2, -1))[
            ..., :3
        ]  # (B, T, 4, 3)
        gt_points_full = (cano_points_homo @ gt_rotmat_homo.transpose(-2, -1))[
            ..., :3
        ]  # (B, T, 4, 3)

        # Compute mean L2 error over the 4 points, per frame, then mean over time
        total_error = (
            torch.norm(pred_points_full - gt_points_full, dim=-1)
            .mean(dim=-1)
            .mean(dim=-1)
        )  # (B,)

        metrics = {
            "tsl_error": defaultdict(list),
            "rot_error": defaultdict(list),
            "total_error": defaultdict(list),
        }
        for b, object_id in enumerate(object_id_list):
            metrics["tsl_error"][object_id].append(tsl_error[b].item())
            metrics["rot_error"][object_id].append(rot_error[b].item())
            metrics["total_error"][object_id].append(total_error[b].item())

        return metrics

    @torch.no_grad()
    def gen_vis_res(
        self,
        step,
        left_hand,
        right_hand,
        object_gt,
        object_pred=None,
        object_noisy=None,
        seq_len=None,
        pref="training/",
        object_id=None,
    ):
        kwargs = {
            "object_pred": object_pred,
            "seq_len": seq_len,
            "pref": pref,
        }
        if self.opt.condition.noisy_obj:
            kwargs["object_noisy"] = object_noisy
        # self.visualizer.log_training_step(
        #     step,
        #     left_hand,
        #     right_hand,
        #     object_gt,
        #     **kwargs
        # )
        left_hand_verts, left_hand_faces = self.hand_wrapper.joint2verts_faces(
            left_hand[0]
        )
        right_hand_verts, right_hand_faces = self.hand_wrapper.joint2verts_faces(
            right_hand[0]
        )

        left_hand_meshes = Meshes(verts=left_hand_verts, faces=left_hand_faces).to(
            device
        )
        left_hand_meshes.textures = mesh_utils.pad_texture(left_hand_meshes, "blue")
        right_hand_meshes = Meshes(verts=right_hand_verts, faces=right_hand_faces).to(
            device
        )
        right_hand_meshes.textures = mesh_utils.pad_texture(right_hand_meshes, "blue")

        wTo_list = [object_gt[0], object_pred[0]]
        color_list = ["red", "yellow"]
        if self.opt.condition.noisy_obj:
            wTo_list.append(object_noisy[0])
            color_list.append("purple")

        return self.viz_off.log_training_step(
            left_hand_meshes,
            right_hand_meshes,
            wTo_list,
            color_list,
            object_id,
            step=step,
            pref=pref,
        )


def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.exp_dir)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    OmegaConf.save(config=opt, f=os.path.join(save_dir / "opt.yaml"))

    diffusion_model = build_model(opt)
    if opt.ckpt:
        ckpt = torch.load(opt.ckpt)
        model_utils.load_my_state_dict(diffusion_model, ckpt["model"])

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.train.batch_size,  # 32
        train_lr=opt.train.lr,  # 1e-4
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=opt.train.ema_decay,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=opt.general.wandb,
        save_and_sample_every=opt.general.save_and_sample_every,
        vis_every=opt.general.vis_every,
        train_num_steps=opt.general.train_num_steps,
    )

    if opt.test:
        trainer.test()
    else:
        trainer.train()

    torch.cuda.empty_cache()


@hydra.main(config_path="../../config", config_name="train", version_base=None)
@slurm_engine()
def main(opt):
    # if opt.test:
    #     run_sample(opt, device)
    # else:
    run_train(opt, device)
    return


vis_P = 500
device = torch.device(f"cuda:0")
if __name__ == "__main__":
    main()
