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

from ..manip.model.transformer_hand_to_object_diffusion_model import (
    CondGaussianDiffusion,
)
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
from .trainer_proof_of_idea import Trainer as BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, opt, diffusion_model):
        super().__init__(opt, diffusion_model)


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

            val_data_dict_for_vis = {k: v[:bs_for_vis] for k, v in val_data_dict.items()}
            fname = self.gen_vis_res(
                self.step,
                val_data_dict_for_vis,
                {'pred_raw': object_pred_raw[:bs_for_vis]},
                seq_len=seq_len,
                pref=f"{pref}",
            )
            if self.use_wandb:
                object_id = val_data_dict["object_id"][bs_for_vis - 1]
                wandb.log(
                    {f"{pref}vis": wandb.Video(fname)}, step=self.step,
                )

            if sample_guide:
                _ = self.gen_vis_res(
                    self.step,
                    val_data_dict_for_vis,
                    {'pred_raw': post_object_pred_raw[:bs_for_vis]},
                    seq_len=seq_len,
                    pref=f"{pref}uid_{val_data_dict['object_id'][bs_for_vis - 1]}_post",
                )
                fname = self.gen_vis_res(
                    self.step,
                    val_data_dict_for_vis,
                    {'pred_raw': guided_object_pred_raw[:bs_for_vis]},
                    seq_len=seq_len,
                    pref=f"{pref}_guided",
                )

            if self.use_wandb:
                wandb.log(
                    {f"{pref}vis_guided": wandb.Video(fname)},
                    step=self.step,
                )
        if rtn_sample:
            return metrics, (object_pred_raw, guided_object_pred_raw)
        return metrics

    def gen_vis_res(self, step, batch, output, seq_len, pref):
        oObj = batch['newMesh'] # a Mesh?? 

        # motion_raw = batch['motion_raw']    # well object raw
        # gt_
        wTo_gt = batch['target_raw']
        hand_meshes = self.decode_mesh(batch['left_hand_params'], batch['right_hand_params'])
        self.viz_off.log_hoi_step(hand_meshes, wTo_gt, oObj, pref=pref + f'_gt_{step:07d}', contact=batch['contact'])

        pred_raw = output['pred_raw']
        pred_dict = self.decode_dict(pred_raw)
        hand_pred_meshes = self.decode_mesh(pred_dict['left_hand'], pred_dict['right_hand'])
        wTo_pred = pred_dict['wTo']
        self.viz_off.log_hoi_step(hand_pred_meshes, wTo_pred, oObj, pref=pref + f'_pred_{step:07d}', contact=pred_dict['contact'])


    def decode_dict(self, target_raw):
        rtn = {
            'wTo': , 
            'left': ,
            'right': ,
            'contact': ,
        }
        return 

    def test(self):
        return 

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
