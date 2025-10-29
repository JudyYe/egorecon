import os
from pathlib import Path
import imageio
import cv2
from tqdm import tqdm
import hydra
import torch
import wandb
from jutils import model_utils
from move_utils.slurm_utils import slurm_engine
from omegaconf import OmegaConf
from pytorch3d.structures import Meshes
from torch.cuda.amp import autocast
from ..manip.model import build_model
from ..manip.model.guidance_optimizer_jax import (
    do_guidance_optimization,
    se3_to_wxyz_xyz,
    wxyz_xyz_to_se3,
)
from ..manip.model.transformer_hand_to_object_diffusion_model import (
    CondGaussianDiffusion,
)
from ..visualization.pt3d_visualizer import Pt3dVisualizer
from .trainer_proof_of_idea import Trainer as BaseTrainer
import os.path as osp


def vis_gen_process(x_list, model, viz_off, opt, step, batch, pref):
    video_list = []
    for i, (x, t) in enumerate(tqdm(x_list)):
        x_raw = model.denormalize_data(x)
        pred_dict = model.decode_dict(x_raw)
        f_pref = f"{pref}_process_{i:04d}"
        pred_dict["newMesh"] = batch["newMesh"]
        log_dict = gen_one(
            model, viz_off, opt, step, pred_dict, name=f_pref
            # model, viz_off, opt, step, batch, {"pred_raw": x_raw}, pref=f_pref
        )
        print(log_dict[f"{f_pref}"]._path, t)
        video_list.append((log_dict[f"{f_pref}"], t))

    print(video_list)
    # read and concat these videos
    frames = []
    for video, i in video_list:
        # wandb.Video()._path
        video = video._path
        save_dir = osp.dirname(video)
        image_list = imageio.mimread(video)
        # print text
        for frame in image_list:
            cv2.putText(
                frame,
                f"Frame {i}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        frames.extend(image_list)

    save_dir = osp.join(save_dir, f"{pref}_process.mp4")
    imageio.mimwrite(save_dir, frames, fps=30)


def gen_one(model, viz_off, opt, step, output, name):
    """
    Generate one visualization result for hand-object interaction.

    Args:
        model: The diffusion model used for decoding
        viz_off: Pt3dVisualizer instance for visualization
        opt: Configuration object containing hand settings
        step: Current step number
        output: 
            "newMesh": The object mesh
            "left_hand_params": The left hand parameters
            "right_hand_params": The right hand parameters
            "contact": The contact information
            "wTo": The object transformation
        name: Name of the visualization result
    """
    oObj = output["newMesh"]  # a Mesh??

    oObj = Meshes(
        verts=[oObj.verts_list()[0]],
        faces=[oObj.faces_list()[0]],
    ).to(device)

    wTo = output["wTo"][0]
    hand_meshes = model.decode_hand_mesh(
        output["left_hand_params"][0],
        output["right_hand_params"][0],
        hand_rep="theta",
    )
    fname = viz_off.log_hoi_step(
        *hand_meshes,
        wTo,
        oObj,
        pref=name + f"_{step:07d}",
        contact=output["contact"][0],
    )
    log_dict = {name: wandb.Video(fname)}

    return log_dict


def gen_vis_res(model, viz_off, opt, step, batch, output, pref):
    """Generate visualization results for hand-object interaction.
    
    Args:
        model: The diffusion model used for decoding
        viz_off: Pt3dVisualizer instance for visualization
        opt: Configuration object containing hand settings
        step: Current step number
        batch: Input batch data
        output: 
        pref: Prefix for visualization files
    """
    assert False, "This function is deprecated"
    oObj = batch["newMesh"]  # a Mesh??

    oObj = Meshes(
        verts=[batch["newMesh"].verts_list()[0]],
        faces=[batch["newMesh"].faces_list()[0]],
    ).to(device)

    # motion_raw = batch['motion_raw']    # well object raw
    # gt_
    wTo_gt = batch["target_raw"][0]
    hand_meshes = model.decode_hand_mesh(
        batch["left_hand_params"][0],
        batch["right_hand_params"][0],
        hand_rep="theta",
    )
    fname = viz_off.log_hoi_step(
        *hand_meshes,
        wTo_gt,
        oObj,
        pref=pref + f"_gt_{step:07d}",
        contact=batch["contact"][0],
    )
    log_dict = {f"{pref}vis_gt": wandb.Video(fname)}

    if output is not None:
        pred_raw = output["pred_raw"]
        pred_dict = model.decode_dict(pred_raw)
        hand_io = opt.get("hand", "cond")
        if hand_io == "out":
            hand_pred_meshes = model.decode_hand_mesh(
                pred_dict["left_hand_params"][0], pred_dict["right_hand_params"][0]
            )
        elif hand_io == "cond":
            hand_pred_meshes = hand_meshes
        wTo_pred = pred_dict["wTo"][0]
        fname_pred = viz_off.log_hoi_step(
            *hand_pred_meshes,
            wTo_pred,
            oObj,
            pref=pref + f"_pred_{step:07d}",
            contact=pred_dict["contact"][0],
        )
        log_dict[f"{pref}vis_pred"] = wandb.Video(fname_pred)

    return log_dict


class Trainer(BaseTrainer):
    def __init__(self, opt, diffusion_model, *args, **kwargs):
        super().__init__(opt, diffusion_model, *args, **kwargs)

    def validation_step(
        self,
        val_data_dict,
        pref="val/",
        just_vis=False,
        sample_guide=False,
        rtn_sample=False,
    ):
        val_data_dict = model_utils.to_cuda(val_data_dict)
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
            val_loss, losses = self.model(
                object_motion,
                cond,
                padding_mask=padding_mask,
                newPoints=val_data_dict["newPoints"],
                hand_raw=val_data_dict["hand_raw"],
                gt_contact=val_data_dict["contact"],
                training_info=self.training_info(),
            )

        if self.use_wandb:
            # val_log_dict = {
            #     "Validation/Loss/Total Loss": val_loss.item(),
            #     "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
            # }
            val_log_dict = {f"Validation/Loss/{k}": v.item() for k, v in losses.items()}
            wandb.log(val_log_dict, step=self.step)

        bs_for_vis = len(val_data_dict["object_id"])

        sample = val_data_dict
        diffusion_model = self.ema.ema_model

        object_motion_raw = sample["target_raw"][:bs_for_vis]
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

            val_data_dict_for_vis = {
                k: v[:bs_for_vis] for k, v in val_data_dict.items()
            }
            log_dict = gen_vis_hoi_res(
                self.model,
                self.viz_off,
                self.opt,
                self.step,
                val_data_dict_for_vis,
                {"pred_raw": object_pred_raw[:bs_for_vis]},
                pref=f"{pref}",
            )
            if self.use_wandb:
                wandb.log(log_dict, step=self.step)

            if sample_guide:
                _ = gen_vis_hoi_res(
                    self.model,
                    self.viz_off,
                    self.opt,
                    self.step,
                    val_data_dict_for_vis,
                    {"pred_raw": post_object_pred_raw[:bs_for_vis]},
                    pref=f"{pref}_post",
                )
                log_dict = gen_vis_hoi_res(
                    self.model,
                    self.viz_off,
                    self.opt,
                    self.step,
                    val_data_dict_for_vis,
                    {"pred_raw": guided_object_pred_raw[:bs_for_vis]},
                    pref=f"{pref}_guided",
                )

            if self.use_wandb:
                wandb.log(log_dict, step=self.step)
        if rtn_sample:
            return metrics, (object_pred_raw, guided_object_pred_raw)
        return metrics

    def test(self):
        for batch in self.val_dl:
            batch = model_utils.to_cuda(batch)
            gen_vis_hoi_res(
                self.model, self.viz_off, self.opt, self.step, batch, None, pref="test"
            )
        return

def gen_vis_hoi_res(model: CondGaussianDiffusion, viz_off: Pt3dVisualizer, opt: OmegaConf, step: int, batch: dict, output: dict, pref: str):
    all_log_dict = {}
    if opt.hand == "cond":
        # batch_dict = {
        #     "newMesh": batch["newMesh"],
        #     "left_hand_params": batch["left_hand_params"],
        #     "right_hand_params": batch["right_hand_params"],
        #     "contact": batch["contact"],
        #     "wTo": batch["target_raw"],
        # }
        target_raw = model.denormalize_data(batch["target"])
        batch_dict = model.decode_dict(target_raw)
        batch_dict["newMesh"] = batch["newMesh"]
        batch_dict["contact"] = batch["contact"]
        batch_dict["left_hand_params"] = batch["left_hand_params"]
        batch_dict["right_hand_params"] = batch["right_hand_params"]

        log_dict = gen_one(model, viz_off, opt, step, batch_dict, pref +'_gt')
        all_log_dict.update(log_dict)

        if output is not None:
            pred_dict = batch_dict
            pred_dict["wTo"] = model.decode_dict(output["pred_raw"])["wTo"]
            log_dict = gen_one(model, viz_off, opt, step, pred_dict, pref +'_pred')
            all_log_dict.update(log_dict)

    elif opt.hand == "cond_out":
        print("target raw shape:", batch["target_raw"].shape)
        target_raw = model.denormalize_data(batch["target"])
        batch_dict = model.decode_dict(target_raw)
        batch_dict["newMesh"] = batch["newMesh"]
        batch_dict["contact"] = batch["contact"]
        log_dict = gen_one(model, viz_off, opt, step, batch_dict, pref +'_gt')
        all_log_dict.update(log_dict)

        if output is not None:
            pred_dict = model.decode_dict(output["pred_raw"])
            pred_dict["newMesh"] = batch["newMesh"]
            log_dict = gen_one(model, viz_off, opt, step, pred_dict, pref +'_pred')
            all_log_dict.update(log_dict)

    return all_log_dict

@hydra.main(config_path="../../config", config_name="train", version_base=None)
@slurm_engine()
def main(opt):
    # if opt.test:
    #     run_sample(opt, device)
    # else:
    run_train(opt, device)
    return


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


device = torch.device("cuda:0")
if __name__ == "__main__":
    main()
