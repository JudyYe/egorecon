import numpy as np
from glob import glob
import pickle
import os
import os.path as osp
from .trainer_hoi import gen_one
from tqdm import tqdm
import hydra
import torch
from jutils import model_utils
from move_utils.slurm_utils import slurm_engine
from omegaconf import OmegaConf
from ..manip.model import build_model
from ..manip.model.guidance_optimizer_hoi_jax import (
    do_guidance_optimization,
)
# from ..manip.model.guidance_optimizer_jax import (
#     do_guidance_optimization,
#     se3_to_wxyz_xyz,
#     wxyz_xyz_to_se3,
# )
from .trainer_hoi import gen_vis_res, vis_gen_process, gen_vis_hoi_res
from ..manip.model.transformer_hand_to_object_diffusion_model import (
    CondGaussianDiffusion,
)
from ..manip.data import build_dataloader
from ..utils.evaluation_metrics import compute_wTo_error
from ..visualization.pt3d_visualizer import Pt3dVisualizer
from ..manip.model import fncmano_jax
from pathlib import Path
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



@torch.no_grad()
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
        # if b.item() != 494:
        #     print(b)
        #     continue
        index = f"{batch['demo_id'][0]}_{batch['object_id'][0]}"

        sample = batch = model_utils.to_cuda(batch)

        seq_len = torch.tensor([sample["condition"].shape[1]]).to(device)
        if model_cfg.get('legacy_mask', True):
            actual_seq_len = seq_len + 2
            tmp_mask = torch.arange(model_cfg.model.window + 2, device=device).expand(
                1, model_cfg.model.window + 2
            ) < actual_seq_len[:, None].repeat(1, model_cfg.model.window + 2)
            padding_mask = tmp_mask[:, None, :]
        else:
            padding_mask = None
        padding_mask = None            

        for i in range(1):
            guided_object_pred_raw, info = diffusion_model.sample_raw(
                torch.randn_like(sample["target"]),
                sample["condition"],
                padding_mask=padding_mask,
                guide=False,
                obs=sample,
                newPoints=sample["newPoints"],
                hand_raw=sample["hand_raw"],
                rtn_x_list=True,
            )

            pred_dict = diffusion_model.decode_dict(guided_object_pred_raw)
            pred_dict["newMesh"] = sample["newMesh"]
            if opt.vis_x:
                gen_one(
                    diffusion_model,
                    viz_off,
                    model_cfg,
                    step=0,
                    output=pred_dict,
                    name=f"test_sample_{i}_{b:04d}",
                )        
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
        save_prediction(diffusion_model.decode_dict(guided_object_pred_raw), index, osp.join(model_cfg.exp_dir, opt.test_folder, "guided", f"{index}.pkl"))

        def get_info_str(info):

            B, T = sample["condition"].shape[:2]
            info_per_time = [[] for _ in range(T)]
            for t in range(T):
                for b in range(B):
                    info_str = f"t={t:04d} \n"
                    for k, v in info.items(): 
                        info_str += f"  {k}={v[b, t]:.4f} \n"
                    info_per_time[t].append(info_str)

            return info_per_time

        if opt.vis_x:
            info_per_time = get_info_str(info['info_inner'][-1])
            gen_vis_hoi_res(
                diffusion_model,
                viz_off,
                model_cfg,
                step=0,
                batch=sample,
                output={"pred_raw": guided_object_pred_raw},
                pref=f"test_guided_{b:04d}",
                debug_info=info_per_time,
            )

        # save for debug
        # tmp_file = "outputs/tmp.pkl"
        # with open(tmp_file, "wb") as f:
        #     pickle.dump(
        #         {
        #             "sample": sample,
        #             "pred_raw": guided_object_pred_raw,
        #         },
        #         f,
        #     )
        # # assert False

        if opt.vis_x0_list:
            vis_gen_process(
                info["x_0_packed_pred"],
                diffusion_model,
                viz_off,
                model_cfg,
                step=0,
                batch=sample,
                pref=f"test_guided_{b:04d}_x0",
            )

        left_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="left")
        right_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="right")


        if opt.post_guidance:
            # directly guide object_pred_raw
            obs = diffusion_model.get_obs(
                guide=True, batch=sample, shape=guided_object_pred_raw.shape
            )

            post_pred_dict, debug_info = do_guidance_optimization(
                pred_dict=diffusion_model.decode_dict(guided_object_pred_raw),
                obs=obs,
                left_mano_model=left_mano_model,
                right_mano_model=right_mano_model,
                guidance_mode=opt.guide.hint,
                phase="post",
                verbose=True,
            )
            info_per_time = get_info_str(debug_info)
            post_object_pred_raw = diffusion_model.encode_dict_to_params(post_pred_dict)
            pred_dict = diffusion_model.decode_dict(post_object_pred_raw)
            if opt.vis_x:
                pred_dict["newMesh"] = sample["newMesh"]
                gen_one(
                    diffusion_model,
                    viz_off,
                    model_cfg,
                    step=0,
                    # batch=sample,
                    output=pred_dict,
                    name=f"test_post_{b:04d}",
                    debug_info=info_per_time,
                )
            
            save_prediction(post_pred_dict, index, osp.join(model_cfg.exp_dir, opt.test_folder, "post", f"{index}.pkl"))


def save_prediction(pred_dict, index, fname):
    os.makedirs(osp.dirname(fname), exist_ok=True)
    to_save = {}
    for key in ["wTo", "left_hand_params", "right_hand_params", "contact", "left_joints", "right_joints"]:
        value = pred_dict.get(key, None)
        if value is not None:
            to_save[key] = value.detach().cpu().numpy()
    to_save["index"] = index

    with open(fname, "wb") as f:
        pickle.dump(to_save, f)
    print(f"Saved to {fname}")


def aggregate_prediction(opt):
    if opt.dir is None:
        save_dir = osp.join(opt.exp_dir, opt.test_folder)
    else:
        save_dir = opt.dir
    pred_list = glob(osp.join(save_dir,  "*.pkl"))
    # for hand, ignore object id
    seq_dict = {}
    for pred_file in pred_list:
        with open(pred_file, "rb") as f:
            pred_dict = pickle.load(f)
        index = pred_dict["index"]
        seq = index.split("_")[0]

        for k, v in pred_dict.items():
            if isinstance(v, np.ndarray) and v.shape[0] == 1:
                pred_dict[k] = v[0]
        seq_dict[seq] = pred_dict


    save_file = osp.dirname(save_dir) + ".pkl"
    with open(save_file, "wb") as f:    
        pickle.dump(seq_dict, f)
    return 




def set_test_cfg(opt, model_cfg):
    # resolve model_cfg first 
    OmegaConf.resolve(model_cfg)
    model_cfg.guide.hint = opt.guide.hint
    model_cfg.sample = opt.sample
    model_cfg.post_guidance = opt.post_guidance

    model_cfg.datasets.window = opt.datasets.window
    model_cfg.datasets.use_cache = opt.datasets.use_cache
    model_cfg.datasets.save_cache = opt.datasets.save_cache
    model_cfg.dyn_only = opt.dyn_only
    model_cfg.oracle_cond = opt.oracle_cond


@hydra.main(config_path="../../config", config_name="test", version_base=None)
@slurm_engine()
def main(opt):

    if opt.test_mode == "guided":
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
        print(opt.sample,)
        test_guided_generation(diffusion_model, dl, opt)

    elif opt.test_mode == 'aggregate':
        aggregate_prediction(opt)

    return



if __name__ == "__main__":
    main()
