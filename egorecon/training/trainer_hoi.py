import os
import os.path as osp
import random
from pathlib import Path

import hydra
import torch
import wandb
from jutils import model_utils, plot_utils
from move_utils.slurm_utils import slurm_engine
from omegaconf import OmegaConf
from pytorch3d.structures import Meshes
from torch.cuda.amp import autocast

from ..manip.data import build_dataloader
from ..manip.model import build_model
from ..manip.model.guidance_optimizer_jax import (do_guidance_optimization,
                                                  se3_to_wxyz_xyz,
                                                  wxyz_xyz_to_se3)
from .trainer_proof_of_idea import Trainer as BaseTrainer


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

    def gen_vis_res(self, step, batch, output, pref):
        oObj = batch['newMesh'] # a Mesh?? 

        oObj = Meshes(verts=[batch['newMesh'].verts_list()[0]], faces=[batch['newMesh'].faces_list()[0]]).to(device)

        # motion_raw = batch['motion_raw']    # well object raw
        # gt_
        wTo_gt = batch['target_raw'][0]
        hand_meshes = self.decode_hand_mesh(batch['left_hand_params'][0], batch['right_hand_params'][0])
        self.viz_off.log_hoi_step(*hand_meshes, wTo_gt, oObj, pref=pref + f'_gt_{step:07d}', contact=batch['contact'][0])

        if output is not None:
            pred_raw = output['pred_raw']
            pred_dict = self.decode_dict(pred_raw)
            hand_pred_meshes = self.decode_hand_mesh(pred_dict['left_hand_params'][0], pred_dict['right_hand_params'][0])
            wTo_pred = pred_dict['wTo'][0]
            self.viz_off.log_hoi_step(*hand_pred_meshes, wTo_pred, oObj, pref=pref + f'_pred_{step:07d}', contact=pred_dict['contact'][0])

    def decode_dict(self, target_raw):
        """
        Decode target_raw back into its components.
        
        Args:
            target_raw: [B, T, D] - denormalized target containing:
                - object trajectory (9D: 3D translation + 6D rotation)
                - hand representation (if hand == "out")
                - contact information (if output.contact == True)
        
        Returns:
            dict: Dictionary containing decoded components
        """
        B, T, D = target_raw.shape
        
        # Start with object trajectory (always first 9D)
        obj_dim = 9
        wTo = target_raw[..., :obj_dim]  # [B, T, 9]
        
        # Track current position in the concatenated tensor
        current_pos = obj_dim
        
        # Extract hand representation if hand == "out"
        left_hand_params = None
        right_hand_params = None
        if self.opt.hand == "out":
            if self.opt.hand_rep == "joints":
                # Hand joints: 21 joints * 3D * 2 hands = 126D
                hand_dim = 21 * 3 * 2
                hand_rep = target_raw[..., current_pos:current_pos + hand_dim]  # [B, T, 126]
                left_hand, right_hand = torch.split(hand_rep, 21 * 3, dim=-1)  # [B, T, 63] each
                
                # Convert joints to hand parameters (this would need the inverse of joint2verts_faces_joints)
                # For now, we'll need to implement this conversion or use a different approach
                left_hand_params = left_hand
                right_hand_params = right_hand
                
            elif self.opt.hand_rep == "theta":
                # Hand theta: (3+3+15+10) * 2 = 62D
                hand_dim = (3 + 3 + 15 + 10) * 2
                hand_rep = target_raw[..., current_pos:current_pos + hand_dim]  # [B, T, 62]
                left_hand_params, right_hand_params = torch.split(hand_rep, hand_dim // 2, dim=-1)  # [B, T, 31] each
            
            current_pos += hand_dim
        
        # Extract contact information if present
        contact = None
        if self.opt.output.contact:
            contact_dim = 2  # left and right hand contact
            contact = target_raw[..., current_pos:current_pos + contact_dim]  # [B, T, 2]
            current_pos += contact_dim
        
        rtn = {
            'wTo': wTo,
            'left_hand_params': left_hand_params,
            'right_hand_params': right_hand_params,
            'contact': contact,
        }
        return rtn

    def decode_hand_mesh(self, left_hand, right_hand):
        device = left_hand.device if left_hand is not None else right_hand.device
        
        hand_rep = self.opt.hand_rep
        if left_hand is not None:
            if hand_rep == "joints":
                verts, faces = plot_utils.pc_to_cubic_meshes(left_hand)
            elif hand_rep == "theta":
                verts, faces, joints = self.model.hand_wrapper.hand_para2verts_faces_joints(left_hand, side='left')
            print(verts.shape, faces.shape, left_hand.shape)
            left_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
        else:
            left_hand_meshes = None

        if right_hand is not None:
            if hand_rep == "joints":
                verts, faces = plot_utils.pc_to_cubic_meshes(right_hand)
            elif hand_rep == "theta":
                verts, faces, joints = self.model.hand_wrapper.hand_para2verts_faces_joints(right_hand, side='right')
            right_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
        else:
            right_hand_meshes = None

        return (left_hand_meshes, right_hand_meshes)

    def test(self):
        for batch in self.val_dl:
            batch = model_utils.to_cuda(batch)
            self.gen_vis_res(self.step, batch, None, pref="test/")
        return 

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

device = torch.device(f"cuda:0")
if __name__ == "__main__":
    main()
