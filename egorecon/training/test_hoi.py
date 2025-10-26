import pickle
import os
import os.path as osp

from tqdm import tqdm
import hydra
import torch
from jutils import model_utils
from move_utils.slurm_utils import slurm_engine
from omegaconf import OmegaConf
from ..manip.model import build_model
from ..manip.model.guidance_optimizer_jax import (
    do_guidance_optimization,
    se3_to_wxyz_xyz,
    wxyz_xyz_to_se3,
)
from .trainer_hoi import gen_vis_res, vis_gen_process
from ..manip.model.transformer_hand_to_object_diffusion_model import (
    CondGaussianDiffusion,
)
from ..manip.data import build_dataloader
from ..utils.evaluation_metrics import compute_wTo_error
from ..visualization.pt3d_visualizer import Pt3dVisualizer

device = torch.device("cuda:0")


def build_model_from_ckpt(opt):
    cfg_file = os.path.join(opt.exp_dir, "opt.yaml")
    with open(cfg_file, "r") as f:
        cfg = OmegaConf.load(f)
    diffusion_model = build_model(cfg)

    ckpt = torch.load(opt.ckpt)
    model_utils.load_my_state_dict(diffusion_model, ckpt["model"])
    diffusion_model.to(device)
    return diffusion_model


def test_guided_generation(diffusion_model: CondGaussianDiffusion, dl, opt):
    model_cfg = diffusion_model.opt
    viz_off = Pt3dVisualizer(
        exp_name=opt.expname,
        save_dir=osp.join(model_cfg.exp_dir, opt.test_folder),
        mano_models_dir=opt.paths.mano_dir,
        object_mesh_dir=opt.paths.object_mesh_dir,
    )

    # add guidance
    for b, batch in enumerate(tqdm(dl)):
        if opt.test_num > 0 and b >= opt.test_num:
            break
        b = batch["ind"][0]
        sample = batch = model_utils.to_cuda(batch)

        seq_len = torch.tensor([sample["condition"].shape[1]]).to(device)
        actual_seq_len = seq_len + 2
        tmp_mask = torch.arange(model_cfg.model.window + 2, device=device).expand(
            1, model_cfg.model.window + 2
        ) < actual_seq_len[:, None].repeat(1, model_cfg.model.window + 2)
        padding_mask = tmp_mask[:, None, :]

        metrics = {}
        guided_object_pred_raw, info = diffusion_model.sample_raw(
            torch.randn_like(sample["target"]),
            sample["condition"],
            padding_mask=padding_mask,
            guide=opt.inner_guidance,
            obs=sample,
            newPoints=sample["newPoints"],
            hand_raw=sample["hand_raw"],
            rtn_x_list=True,
        )

        tmp_file = "outputs/tmp.pkl"

        with open(tmp_file, "wb") as f:
            pickle.dump(
                {
                    "sample": sample,
                    "pred_raw": guided_object_pred_raw,
                },
                f,
            )
        assert False

        save_file = osp.join(
            model_cfg.exp_dir, opt.test_folder, f"test_guided_{b:04d}.pkl"
        )
        # with open(save_file, "wb") as f:
        #     pickle.dump(
        #         {
        #             "x_list": info["x_list"],
        #             "wJoints": sample["hand_raw"],
        #             "newPoints": sample["newPoints"],
        #             "wTo": guided_object_pred_raw,
        #             "batch": sample,
        #         },
        #         f,
        #     )
        # print(f"Saved to {save_file}")

        # # vis x_list
        # vis_gen_process(
        #     info["x_list"],
        #     diffusion_model,
        #     viz_off,
        #     model_cfg,
        #     step=0,
        #     batch=sample,
        #     pref=f"test_guided_{b:04d}_xt",
        # )

        # vis_gen_process(
        #     info["x_0_packed_pred"],
        #     diffusion_model,
        #     viz_off,
        #     model_cfg,
        #     step=0,
        #     batch=sample,
        #     pref=f"test_guided_{b:04d}_x0",
        # )

        # print(guided_object_pred_raw.shape)
        metrics_guided = compute_wTo_error(
            guided_object_pred_raw, sample["target_raw"], sample["object_id"]
        )
        metrics.update({f"{k}_guided": v for k, v in metrics_guided.items()})

        gen_vis_res(
            diffusion_model,
            viz_off,
            model_cfg,
            step=0,
            batch=sample,
            output={"pred_raw": guided_object_pred_raw},
            pref=f"test_guided_{b:04d}",
        )

        if opt.post_guidance:
            # directly guide object_pred_raw
            obs = diffusion_model.get_obs(
                guide=True, batch=sample, shape=guided_object_pred_raw.shape
            )
            post_object_pred_raw, _ = do_guidance_optimization(
                traj=se3_to_wxyz_xyz(guided_object_pred_raw),
                obs=obs,
                guidance_mode=opt.guide.hint,
                phase="post",
                verbose=True,
            )
            post_object_pred_raw = wxyz_xyz_to_se3(post_object_pred_raw)
            metrics_post = compute_wTo_error(
                post_object_pred_raw, sample["target_raw"], sample["object_id"]
            )
            metrics.update({f"{k}_post": v for k, v in metrics_post.items()})

            gen_vis_res(
                diffusion_model,
                viz_off,
                model_cfg,
                step=0,
                batch=sample,
                output={"pred_raw": post_object_pred_raw},
                pref=f"test_post_{b:04d}",
            )

        # Generate visualization for the guided prediction


def set_test_cfg(opt, model_cfg):
    model_cfg.guide.hint = opt.guide.hint
    model_cfg.ddim = opt.ddim
    model_cfg.post_guidance = opt.post_guidance


@hydra.main(config_path="../../config", config_name="test", version_base=None)
@slurm_engine()
def main(opt):
    diffusion_model = build_model_from_ckpt(opt)
    set_test_cfg(opt, diffusion_model.opt)

    torch.manual_seed(123)

    dl, ds = build_dataloader(
        opt.testdata,
        diffusion_model.opt,
        is_train=False,
        shuffle=True,
        batch_size=1,
        num_workers=1,
    )

    test_guided_generation(diffusion_model, dl, opt)

    return



if __name__ == "__main__":
    main()
