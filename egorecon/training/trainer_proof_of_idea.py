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
from jutils import model_utils
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm

from ..manip.data.hand_to_object_dataset import HandToObjectDataset
from ..manip.model.transformer_hand_to_object_diffusion_model import \
    CondGaussianDiffusion
from ..visualization.rerun_visualizer import RerunVisualizer


def run_smplx_model(
    root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=False
):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3
    # betas: BS X 16
    # gender: BS
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(
            aa_rot_rep.device
        )  # BS X T X 30 X 3
        aa_rot_rep = torch.cat(
            (aa_rot_rep, padding_zeros_hand), dim=2
        )  # BS X T X 52 X 3

    aa_rot_rep = aa_rot_rep.reshape(bs * num_steps, -1, 3)  # (BS*T) X n_joints X 3
    betas = (
        betas[:, None, :].repeat(1, num_steps, 1).reshape(bs * num_steps, -1)
    )  # (BS*T) X 16
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist()  # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3)  # (BS*T) X 3
    smpl_betas = betas  # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :]  # (BS*T) X 3
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63)  # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90)  # (BS*T) X 90

    B = smpl_trans.shape[0]  # (BS*T)

    smpl_vals = [
        smpl_trans,
        smpl_root_orient,
        smpl_betas,
        smpl_pose_body,
        smpl_pose_hand,
    ]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ["male", "female"]
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=int) * -1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=int)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue

        # reconstruct SMPL
        (
            cur_pred_trans,
            cur_pred_orient,
            cur_betas,
            cur_pred_pose,
            cur_pred_pose_hand,
        ) = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(
            pose_body=cur_pred_pose,
            pose_hand=cur_pred_pose_hand,
            betas=cur_betas,
            root_orient=cur_pred_orient,
            trans=cur_pred_trans,
        )

        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    if return_joints24:
        x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0)  # () X 52 X 3
        lmiddle_index = 28
        rmiddle_index = 43
        x_pred_smpl_joints = torch.cat(
            (
                x_pred_smpl_joints_all[:, :22, :],
                x_pred_smpl_joints_all[:, lmiddle_index : lmiddle_index + 1, :],
                x_pred_smpl_joints_all[:, rmiddle_index : rmiddle_index + 1, :],
            ),
            dim=1,
        )
    else:
        x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]

    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map]  # (BS*T) X 22 X 3

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map]  # (BS*T) X 6890 X 3

    x_pred_smpl_joints = x_pred_smpl_joints.reshape(
        bs, num_steps, -1, 3
    )  # BS X T X 22 X 3/BS X T X 24 X 3
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(
        bs, num_steps, -1, 3
    )  # BS X T X 6890 X 3

    mesh_faces = pred_body.f

    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=40000,
        results_folder="./results",
        use_wandb=True,
    ):
        super().__init__()
        self.opt = opt
        self.use_wandb = use_wandb
        if self.use_wandb:
            save_dir = results_folder
            expname = opt.expname
            # Loggers
            os.makedirs(save_dir + '/log/wandb', exist_ok=True)

            runid = None
            if os.path.exists(f"{save_dir}/runid.txt"):
                runid = open(f"{save_dir}/runid.txt").read().strip()

            log = wandb.init(
                project='er_' + osp.dirname(expname),
                name=osp.basename(expname),
                dir=osp.join(save_dir, 'log'),
                id=runid,
                save_code=True,
                settings=wandb.Settings(start_method='fork'),
                # config=opt,
                # project=opt.wandb_pj_name,
                # name=opt.exp_name,
                # dir=opt.save_dir,
            )
            runid = log.id
            def save_runid():
                with  open(f"{save_dir}/runid.txt", 'w') as fp:
                    fp.write(runid)

            save_runid()

        self.batch_size = train_batch_size
        self.prep_dataloader(window_size=opt.model.window)

        self.model = diffusion_model
        diffusion_model.set_metadata(self.ds.stats)
        diffusion_model.to(device)
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt

        self.data_root_folder = getattr(self.opt, 'data_root_folder', 'data')

        self.window = opt.model.window

        self.use_object_split = getattr(self.opt, 'use_object_split', False)

        # self.bm_dict = self.ds.bm_dict

        self.test_on_train = getattr(self.opt, 'test_sample_res_on_train', False)

        self.add_hand_processing = getattr(self.opt, 'add_hand_processing', False)

        self.for_quant_eval = getattr(self.opt, 'for_quant_eval', False)

        self.visualizer = RerunVisualizer(
            exp_name=opt.expname,
            save_dir=opt.exp_dir,
            enable_visualization=True,
            mano_models_dir=opt.paths.mano_dir,
            
        )

    def prep_dataloader(self, window_size):
        opt = self.opt
        train_dataset = HandToObjectDataset(
            is_train=True,
            data_path=opt.data.data_path,
            window_size=opt.model.window,
            single_demo=opt.data.demo_id,
            single_object=opt.data.target_object_id,
            sampling_strategy="random",
            split=opt.datasets.split,
            split_seed=42,  # Ensure reproducible splits
            noise_scheme="syn",
            **opt.datasets.augument,
            opt=opt,
        )

        val_dataset = HandToObjectDataset(
            is_train=False,
            data_path=opt.data.data_path,
            window_size=opt.model.window,
            single_demo=opt.data.demo_id,
            single_object=opt.data.target_object_id,
            sampling_strategy="random",
            split="mini",
            split_seed=42,  # Use same seed for consistent splits
            noise_scheme="real",
            opt=opt,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
            )
        )
        # self.val_dl = cycle(
        #     data.DataLoader(
        #         self.val_ds,
        #         batch_size=1,
        #         shuffle=False,
        #         pin_memory=True,
        #         num_workers=4,
        #     )
        # )
        self.val_dl = data.DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

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
                mask_prob = 0.5
                max_infill_ratio = 0.1
                prob = random.uniform(0, 1)
                batch_size, clip_len, _ = cond.shape
                if prob > 1 - mask_prob:
                    traj_feat_dim = sample["traj_noisy_raw"].shape[-1]
                    start = (
                        torch.FloatTensor(batch_size).uniform_(0, clip_len - 1).long()
                    )
                    mask_len = (
                        clip_len
                        * torch.FloatTensor(batch_size).uniform_(0, 1)
                        * max_infill_ratio
                    ).long()
                    end = start + mask_len
                    end[end > clip_len] = clip_len
                    mask_traj = torch.ones(batch_size, clip_len).to(device)  # [bs, t]
                    for bs in range(batch_size):
                        mask_traj[bs, start[bs] : end[bs]] = 0
                    mask_traj_exp = mask_traj.unsqueeze(-1).repeat(
                        1, 1, traj_feat_dim
                    )  # [bs, t, 4]
                    cond[:, :, -traj_feat_dim:] = (
                        cond[:, :, -traj_feat_dim:] * mask_traj_exp
                    )
                else:
                    mask_traj = torch.ones(batch_size, clip_len).to(device)

                # Extract data from sample and move to device
                # Generate padding mask
                actual_seq_len = seq_len + 1
                tmp_mask = torch.arange(self.window + 1, device=device).expand(
                    1, self.window + 1
                ) < actual_seq_len[:, None].repeat(1, self.window + 1)
                padding_mask = tmp_mask[:, None, :]

                with autocast(device_type='cuda', enabled=self.amp):

                    loss_diffusion = self.model(object_motion, cond, padding_mask)

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
                        wandb.log(log_dict)

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

                with torch.no_grad():
                    for b, val_data_dict in enumerate(self.val_dl):
                        print(b, val_data_dict["object_id"])
                        val_data_dict = model_utils.to_cuda(val_data_dict)

                        self.validation_step(sample, pref=f"{b}/train/")
                        self.validation_step(val_data_dict, pref=f"{b}/val/")
            self.step += 1

        print("training complete")

        if self.use_wandb:
            wandb.run.finish()

    def validation_step(self, val_data_dict, pref="val/"):
        cond = val_data_dict["condition"]
        device = cond.device
        seq_len = torch.tensor([cond.shape[1]]).to(device)

        actual_seq_len = seq_len + 1
        tmp_mask = torch.arange(self.window + 1, device=device).expand(
            1, self.window + 1
        ) < actual_seq_len[:, None].repeat(1, self.window + 1)
        padding_mask = tmp_mask[:, None, :]
        object_motion = val_data_dict["target"]
        cond = val_data_dict["condition"]

        with autocast(device_type='cuda', enabled=self.amp):
            val_loss_diffusion = self.model(
                object_motion, cond, padding_mask=padding_mask
            )

        val_loss = val_loss_diffusion
        if self.use_wandb:
            val_log_dict = {
                "Validation/Loss/Total Loss": val_loss.item(),
                "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
            }
            wandb.log(val_log_dict)

        milestone = self.step // self.opt.general.save_and_sample_every
        bs_for_vis = 1

        if self.step % self.opt.general.save_and_sample_every == 0:
            self.save(milestone)
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
                torch.randn_like(object_motion), cond, padding_mask=padding_mask
            )
            self.gen_vis_res(
                self.step,
                left_hand_raw,
                right_hand_raw,
                object_motion_raw,
                object_noisy=sample["traj_noisy_raw"],
                object_pred=object_pred_raw,
                seq_len=seq_len,
                pref=pref,
            )

    def gen_vis_res(self, *args, **kwargs):
        self.visualizer.log_training_step(*args, **kwargs)


def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.exp_dir)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    OmegaConf.save(config=opt, f=os.path.join(save_dir / "opt.yaml"))

    # Define model
    repr_dim = 24 * 3 + 22 * 6

    repr_dim = 9  # Output dimension (3D translation + 6D rotation)
    cond_dim = 2 * 21 * 3 + 9  # Input dimension (2 hands × pose_dim each)

    diffusion_model = CondGaussianDiffusion(
        opt,
        d_feats=repr_dim,
        out_dim=repr_dim,
        condition_dim=cond_dim,
        max_timesteps=opt.model.window + 1,
        timesteps=1000,
        loss_type="l1",
        **opt.model,
    )
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.train.batch_size,  # 32
        train_lr=opt.train.lr,  # 1e-4
        gradient_accumulate_every=opt.train.ema_update_every,  # gradient accumulation steps
        ema_decay=opt.train.ema_decay,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=opt.general.wandb,
        save_and_sample_every=opt.general.vis_every,
        train_num_steps=opt.general.train_num_steps,
    )

    trainer.train()

    torch.cuda.empty_cache()



@hydra.main(config_path="../../config", config_name="train", version_base=None)
def main(opt):
    if opt.test:
        run_sample(opt, device)
    else:
        run_train(opt, device)
    return 


if __name__ == "__main__":

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    main()