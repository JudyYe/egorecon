import os
import os.path as osp
import pickle
from copy import deepcopy
from glob import glob
from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
from jutils import geom_utils, model_utils
from move_utils import slurm_utils
from omegaconf import OmegaConf
from PIL import Image
from pytorch3d.structures import Meshes
from tqdm import tqdm
from eval.eval_hoi import eval_hotclip_pose6d, eval_hotclip_joints
from ..manip.data import build_dataloader
from ..manip.model import build_model, fncmano_jax
from ..manip.model.guidance_optimizer_hoi_jax import (
    do_guidance_optimization,
)
from ..manip.model.transformer_hand_to_object_diffusion_model import (
    CondGaussianDiffusion,
)
from ..utils.motion_repr import HandWrapper
from ..visualization.pt3d_visualizer import Pt3dVisualizer
from .trainer_hoi import (
    vis_gen_process,
    vis_one_from_cam_side,
)

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


def extract_hoi(
    demo_id,
    fname,
):
    hot3d_dir = f"data/HOT3D-CLIP/extract_images-rot90/clip-{demo_id}"
    image_paths = glob(osp.join(hot3d_dir, "*.jpg"))
    print(f"query {osp.join(hot3d_dir, '*.jpg')}, found {len(image_paths)} images")
    image_paths = sorted(image_paths)
    images = [Image.open(image_path) for image_path in image_paths]
    images = [image.convert("RGB") for image in images]
    images = [image.resize((360, 360)) for image in images]
    os.makedirs(osp.dirname(fname), exist_ok=True)
    imageio.mimwrite(fname, images, fps=30)


@torch.no_grad()
def patch_wTc(diffusion_model: CondGaussianDiffusion, dl, opt):
    model_cfg = diffusion_model.opt

    # add guidance
    for b, batch in enumerate(tqdm(dl)):
        if opt.test_num > 0 and b >= opt.test_num:
            break
        b = batch["ind"][0]
        # if b.item() != 484:
        #     continue
        index = f"{batch['demo_id'][0]}_{batch['object_id'][0]}"
        save_file = osp.join(model_cfg.exp_dir, opt.test_folder, "post", f"{index}.pkl")
        with open(save_file, "rb") as f:
            pred_dict = pickle.load(f)

        pred_dict["wTc"] = batch["wTc"][0].cpu().numpy()

        with open(save_file, "wb") as f:
            pickle.dump(pred_dict, f)


def get_pred_obs_from_fp(sample, model_cfg):
    obs = CondGaussianDiffusion.get_obs(
        model_cfg, guide=True, batch=sample, shape=sample["target"].shape
    )
    device = sample["target"].device
    B, T = sample["condition"].shape[:2]

    pred = {
        "wTo": sample["wTo_shelf"],
        "left_hand_params": sample["noisy_left_hand_params"],
        "right_hand_params": sample["noisy_right_hand_params"],
        "contact": torch.zeros(
            [B, T, 2], device=device
        ),  # nothing just for consistency
    }
    return pred, obs


def test_fp(dl, opt, model_cfg):
    hand_wrapper = HandWrapper(opt.paths.mano_dir).to(device)
    viz_off = Pt3dVisualizer(
        exp_name=opt.expname,
        save_dir=osp.join(opt.exp_dir, opt.test_folder),
        mano_models_dir=opt.paths.mano_dir,
        object_mesh_dir=opt.paths.object_mesh_dir,
    )
    left_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="left")
    right_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="right")

    for b, batch in enumerate(tqdm(dl)):
        if opt.test_num > 0 and b >= opt.test_num:
            break
        b = batch["ind"][0]
        # if b.item() != 484:
        #     continue
        index = f"{batch['demo_id'][0]}_{batch['object_id'][0]}"
        B, T = batch["condition"].shape[:2]

        sample = batch = model_utils.to_cuda(batch)
        # guide FP !
        pred_dict, obs = get_pred_obs_from_fp(sample, model_cfg)

        # vis GT here:
        if "wTo" in sample:
            wTo = sample["wTo"][0]
        elif "target_raw" in sample:
            wTo = sample["target_raw"][0]
        else:
            wTo = None
        
        # gt_dict = {
        #     "wTo": wTo,
        #     "left_hand_params": sample["left_hand_params"][0],
        #     "right_hand_params": sample["right_hand_params"][0],
        #     "contact": sample["contact"][0] if "contact" in sample else None,
        #     "newMesh": sample["newMesh"],
        # }
        if opt.vis_x:
            verts, faces, joints = hand_wrapper.hand_para2verts_faces_joints(
                sample["left_hand_params"][0], side="left"
            )
            left_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
            verts, faces, joints = hand_wrapper.hand_para2verts_faces_joints(
                sample["right_hand_params"][0], side="right"
            )
            right_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
            gt_hand_meshes = [left_hand_meshes, right_hand_meshes]
            
            vis_one_from_cam_side(
                None,
                viz_off,
                model_cfg,
                step=0,
                output=sample,
                camera={"intr": sample["intr"][0], "wTc": sample["wTc"][0]},
                name=f"{index}_gt",
                hand_meshes=gt_hand_meshes,
            )
        
        # vis input here:
        fname = osp.join(
            model_cfg.exp_dir, opt.test_folder, "log", f"{index}_input.mp4"
        )
        extract_hoi(sample["demo_id"][0], fname)

        post_dict, debug_info = do_guidance_optimization(
            pred_dict=pred_dict,
            obs=obs,
            left_mano_model=left_mano_model,
            right_mano_model=right_mano_model,
            guidance_mode=opt.guide.hint,
            phase="post",
            verbose=True,
        )
        info_per_time = get_info_str(debug_info, B, T)
        save_dict = deepcopy(post_dict)  # for now
        if opt.vis_x:
            post_dict["newMesh"] = sample["newMesh"]
            verts, faces, joints = hand_wrapper.hand_para2verts_faces_joints(
                post_dict["left_hand_params"][0], side="left"
            )
            left_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
            verts, faces, joints = hand_wrapper.hand_para2verts_faces_joints(
                post_dict["right_hand_params"][0], side="right"
            )
            right_hand_meshes = Meshes(verts=verts, faces=faces).to(device)
            hand_meshes = [left_hand_meshes, right_hand_meshes]

            vis_one_from_cam_side(
                None,
                viz_off,
                model_cfg,
                step=0,
                output=post_dict,
                camera={"intr": sample["intr"][0], "wTc": sample["wTc"][0]},
                name=f"{index}_fp",
                debug_info=info_per_time,
                hand_meshes=hand_meshes,
            )
            save_prediction(
                save_dict,
                index,
                osp.join(model_cfg.exp_dir, opt.test_folder, "post", f"{index}.pkl"),
                wTc=sample["wTc"][0],
            )


def get_info_str(info, B, T):
    # B, T = sample["condition"].shape[:2]
    info_per_time = [[] for _ in range(T)]
    for t in range(T):
        for b in range(B):
            info_str = f"t={t:04d} x1000\n"
            for k, v in info.items():
                info_str += f"  {k}={v[b, t] * 1000:.4f} \n"
            info_per_time[t].append(info_str)

    return info_per_time


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
        # if b.item() != 484:
        #     continue
        index = f"{batch['demo_id'][0]}_{batch['object_id'][0]}"
        B, T = batch["condition"].shape[:2]

        sample = batch = model_utils.to_cuda(batch)

        seq_len = torch.tensor([sample["condition"].shape[1]]).to(device)
        if model_cfg.get("legacy_mask", True):
            actual_seq_len = seq_len + 2
            tmp_mask = torch.arange(model_cfg.model.window + 2, device=device).expand(
                1, model_cfg.model.window + 2
            ) < actual_seq_len[:, None].repeat(1, model_cfg.model.window + 2)
            padding_mask = tmp_mask[:, None, :]
        else:
            padding_mask = None
        padding_mask = None

        # visualize real RGB
        fname = osp.join(
            model_cfg.exp_dir, opt.test_folder, "log", f"{index}_input.mp4"
        )
        extract_hoi(sample["demo_id"][0], fname)

        # visualize gt
        gt = diffusion_model.decode_dict(
            diffusion_model.denormalize_data(sample["target"])
        )
        gt["newMesh"] = sample["newMesh"]
        gt["contact"] = sample["contact"]
        vis_one_from_cam_side(
            diffusion_model,
            viz_off,
            model_cfg,
            step=0,
            output=gt,
            camera={"intr": sample["intr"][0], "wTc": sample["wTc"][0]},
            name=f"{index}_gt",
        )

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
                vis_one_from_cam_side(
                    diffusion_model,
                    viz_off,
                    model_cfg,
                    step=0,
                    output=pred_dict,
                    camera={"intr": sample["intr"][0], "wTc": sample["wTc"][0]},
                    name=f"{index}_{i}_sample",
                )
            save_prediction(
                pred_dict,
                index,
                osp.join(model_cfg.exp_dir, opt.test_folder, "sample", f"{index}.pkl"),
                wTc=sample["wTc"][0],
            )

        if opt.inner_guidance:
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
            save_prediction(
                diffusion_model.decode_dict(guided_object_pred_raw),
                index,
                osp.join(model_cfg.exp_dir, opt.test_folder, "guided", f"{index}.pkl"),
                wTc=sample["wTc"][0],
            )

            if opt.vis_x:
                info_per_time = get_info_str(info["info_inner"][-1], B, T)
                pred_dict = diffusion_model.decode_dict(guided_object_pred_raw)
                pred_dict["newMesh"] = sample["newMesh"]
                vis_one_from_cam_side(
                    diffusion_model,
                    viz_off,
                    model_cfg,
                    step=0,
                    output=pred_dict,
                    camera={"intr": sample["intr"][0], "wTc": sample["wTc"][0]},
                    name=f"{index}_guided",
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
                    pref=f"{index}_x0",
                )

        left_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="left")
        right_mano_model = fncmano_jax.MANOModel.load(Path("assets/mano"), side="right")

        if opt.post_guidance:
            # directly guide object_pred_raw
            obs = diffusion_model.get_obs(
                model_cfg,
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
            info_per_time = get_info_str(debug_info, B, T)
            post_object_pred_raw = diffusion_model.encode_dict_to_params(post_pred_dict)
            pred_dict = diffusion_model.decode_dict(post_object_pred_raw)
            if opt.vis_x:
                pred_dict["newMesh"] = sample["newMesh"]
                vis_one_from_cam_side(
                    diffusion_model,
                    viz_off,
                    model_cfg,
                    step=0,
                    output=pred_dict,
                    camera={"intr": sample["intr"][0], "wTc": sample["wTc"][0]},
                    name=f"{index}_post",
                    debug_info=info_per_time,
                )
            save_prediction(
                post_pred_dict,
                index,
                osp.join(model_cfg.exp_dir, opt.test_folder, "post", f"{index}.pkl"),
                wTc=sample["wTc"][0],
            )


def save_prediction(pred_dict, index, fname, wTc):
    os.makedirs(osp.dirname(fname), exist_ok=True)
    to_save = {}
    for key in [
        "wTo",
        "left_hand_params",
        "right_hand_params",
        "contact",
        "left_joints",
        "right_joints",
    ]:
        value = pred_dict.get(key, None)
        if value is not None:
            to_save[key] = value.detach().cpu().numpy()
    to_save["index"] = index
    to_save["wTc"] = wTc.detach().cpu().numpy()  # GT

    with open(fname, "wb") as f:
        pickle.dump(to_save, f)
    print(f"Saved to {fname}")


def aggregate_prediction(opt):
    if opt.dir is None:
        save_dir = osp.join(opt.exp_dir, opt.test_folder, "post")
    else:
        save_dir = opt.dir
    pred_list = glob(osp.join(save_dir, "*.pkl"))
    # for hand, ignore object id
    seq_dict = {}
    # for object evaluation, save with object id
    seq_obj_dict = {}

    for pred_file in pred_list:
        with open(pred_file, "rb") as f:
            pred_dict = pickle.load(f)
        index = pred_dict["index"]
        seq = index.split("_")[0]
        obj_id = index.split("_")[1]

        for k, v in pred_dict.items():
            if isinstance(v, np.ndarray) and v.shape[0] == 1:
                pred_dict[k] = v[0]
        seq_dict[seq] = pred_dict
        # Aggregate for object evaluation (with object id)

        if seq not in seq_obj_dict:
            seq_obj_dict[seq] = {}
        # Save wTo with object ID as key
        wTo = (
            geom_utils.se3_to_matrix_v2(torch.FloatTensor(pred_dict["wTo"]))
            .cpu()
            .numpy()
        )
        seq_obj_dict[seq][f"obj_{obj_id}_wTo"] = wTo  # (T, 4,)
        seq_obj_dict[seq]["wTc"] = pred_dict["wTc"]

    # Save hand evaluation format
    save_file = save_dir.rstrip("/") + ".pkl"
    print(f"Saving hand predictions to {save_file}")
    with open(save_file, "wb") as f:
        pickle.dump(seq_dict, f)

    # Save object evaluation format
    save_file_obj = save_dir.rstrip("/") + "_objects.pkl"
    print(f"Saving object predictions to {save_file_obj}")
    with open(save_file_obj, "wb") as f:
        pickle.dump(seq_obj_dict, f)

    return save_file, save_file_obj


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
@slurm_utils.slurm_engine()
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
            load_obs=True,
        )
        test_guided_generation(diffusion_model, dl, opt)
    elif opt.test_mode == "patch":
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
        patch_wTc(diffusion_model, dl, opt)
    elif opt.test_mode == "fp":
        model_cfg = OmegaConf.load(opt.dir)
        set_test_cfg(opt, model_cfg)
        dl, ds = build_dataloader(
            opt.testdata,
            model_cfg,
            is_train=False,
            shuffle=True,
            batch_size=1,
            num_workers=1,
            load_obs=True,
        )
        test_fp(dl, opt, model_cfg)

    hand_file, obj_file = aggregate_prediction(opt)
    
    eval_hotclip_pose6d(pred_file=obj_file, split=opt.testdata.testsplit, skip_not_there=True, )
    eval_hotclip_joints(pred_file=hand_file, split=opt.testdata.testsplit, skip_not_there=True, )

    return


if __name__ == "__main__":
    main()
