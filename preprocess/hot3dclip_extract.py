import argparse
import json
import os
import os.path as osp
import pickle
import tarfile
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rerun as rr
import torch
from hand_tracking_toolkit.dataset import HandShapeCollection, warp_image
from jutils import hand_utils
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import preprocess.hot3dclip_util as clip_util  # pyre-ignore
from hot3d.hot3d.data_loaders.loader_hand_poses import Handedness

# data format:
# seq:
# "objects": ['uid1', 'uid2', ...],
# "left_hand_theta": [T, 3+3+15], 3aa+3tsl+pcax15
# "left_hand_shape": [T, 10],
# "right_hand_theta": [T, 3+3+15],
# "right_hand_shape": [T, 10],
# "wTc": [T, 4, 4],
# intrinsic: [3, 3],
# "obj_{i}_cTo_shelf": [T, 4, 4],
# "obj_{i}_shelf_valid": [T, ],
# "obj_{i}_wTo": [T, 4, 4],
# "obj_{i}_wTo_valid": [T, ],


def get_one_clip(clip_path, vis=False):
    tar = tarfile.open(clip_path, mode="r")
    clip_name = os.path.basename(clip_path).split(".tar")[0]
    clip_output_path = os.path.join(vis_dir, clip_name)
    os.makedirs(clip_output_path, exist_ok=True)
    hand_shape: Optional[HandShapeCollection] = clip_util.load_hand_shape(tar)

    anno = defaultdict(list)
    #
    # anno = {
    #     "objects": [],
    #     "left_hand_theta": [],
    #     "left_hand_shape": [],
    #     "right_hand_theta": [],
    #     "right_hand_shape": [],
    #     "wTc": [],
    #     "intrinsic": [],
    #     "uid{i}_wTo": [],
    #     "uid{i}_wTo_valid": [T, ],
    # }
    T = clip_util.get_number_of_frames(tar)
    anno["wTc"] = np.tile(np.eye(4)[None], (T, 1, 1))
    anno["left_hand_shape"] = [hand_shape.mano_beta.cpu().numpy()] * T
    anno["right_hand_shape"] = [hand_shape.mano_beta.cpu().numpy()] * T
    anno["left_hand_theta"] = np.zeros((T, 3 + 3 + 15))
    anno["right_hand_theta"] = np.zeros((T, 3 + 3 + 15))
    image_list = []

    for frame_id in tqdm(range(clip_util.get_number_of_frames(tar))):
        frame_key = f"{frame_id:06d}"

        # Load camera parameters.
        cameras, _ = clip_util.load_cameras(tar, frame_key)
        image_streams = sorted(cameras.keys(), key=lambda x: int(x.split("-")[0]))

        # Load hand and object annotations.
        hands: Optional[Dict[str, Any]] = clip_util.load_hand_annotations(
            tar, frame_key
        )

        for hand_side in ["left", "right"]:
            if hand_side not in hands:
                print("Why no hand????", clip_name, frame_id, hand_side)
                assert False
            theta = np.array(hands[hand_side]["mano_pose"]["thetas"])  # 15
            xform = np.array(hands[hand_side]["mano_pose"]["wrist_xform"])  # 6
            global_orient = xform[:3]
            transl = xform[3:]
            thetas = np.concatenate([global_orient, transl, theta], axis=-1)
            anno[f"{hand_side}_hand_theta"][frame_id] = thetas

        objects: Optional[Dict[str, Any]] = clip_util.load_object_annotations(
            tar, frame_key
        )

        stream_id = image_streams[0]

        # Camera parameters of the current image.
        camera_model = cameras[stream_id]
        camera_model_orig = camera_model
        camera_model = clip_util.convert_to_pinhole_camera(camera_model)
        if vis:
            stream_key = str(stream_id)

            # Load the image.
            image: np.ndarray = clip_util.load_image(tar, frame_key, stream_key)

            # Make sure the image has 3 channels.
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)

            image = warp_image(
                src_camera=camera_model_orig,
                dst_camera=camera_model,
                src_image=image,
            )
            image_list.append(image)

        intrinsic = camera_model.uv_to_window_matrix()
        wTc = camera_model.T_world_from_eye
        rot = np.array(
            [
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        anno["wTc"][frame_id] = wTc @ rot

        if objects is None:
            print("Why????", clip_name, frame_id)
            assert False

        for instance_list in objects.values():
            for instance in instance_list:
                bop_id = int(instance["object_bop_id"])
                bop_id = f"{bop_id:06d}"
                wTo = clip_util.se3_from_dict(instance["T_world_from_object"])
                if f"obj_{bop_id}_wTo" not in anno:
                    anno[f"obj_{bop_id}_wTo"] = np.tile(np.eye(4)[None], (T, 1, 1))
                    anno[f"obj_{bop_id}_wTo_valid"] = np.zeros(T).astype(bool)

                anno[f"obj_{bop_id}_wTo"][frame_id] = wTo
                anno[f"obj_{bop_id}_wTo_valid"][frame_id] = True
                anno["objects"].append(bop_id)

    # get all bop_id
    bop_id_list = list(set(anno["objects"]))
    anno["objects"] = bop_id_list
    anno["intrinsic"] = intrinsic

    for key, value in anno.items():
        anno[key] = np.array(value)

    if vis:
        vis_meta(anno, image_list, clip_output_path)

    out_file = osp.join(args.preprocess_dir, clip_name + ".npz")
    np.savez_compressed(out_file, **anno)
    return anno, image_list


def vis_meta(meta, image_list, vis_dir, num=-1):
    intrinsic = meta["intrinsic"]

    T_w_c = meta["wTc"]

    H, W = image_list[0].shape[:2]

    rr.init("vis_meta")
    rr.save(osp.join(vis_dir, f"vis_meta.rrd"))
    print("Saved rrd to", osp.join(vis_dir, f"vis_meta.rrd"))
    print(meta["objects"])

    sided_wrapper = {}
    for side in ["left", "right"]:
        sided_wrapper[side] = hand_utils.ManopthWrapper(
            args.mano_model_dir,
            side=side,
        )

    if num < 0:
        num = len(T_w_c)

    for t in range(num):
        rr.set_time_sequence("frame", t)
        # log image
        image = image_list[t]
        # resacle to maxium 256
        image = cv2.resize(image, (256, 256))
        image = np.rot90(image, k=3)
        rr.log("world/image", rr.Image(image))
        pose = T_w_c[t]
        focal_length = intrinsic[0, 0]
        if t == 0:
            rr.log(
                "world/camera",
                rr.Pinhole(
                    width=W,
                    height=H,
                    focal_length=float(focal_length),
                ),
            )
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=pose[:3, 3],
                rotation=rr.Quaternion(
                    xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat()
                ),
            ),
        )
        for uid in meta["objects"]:
            T_w_o = meta[f"obj_{uid}_wTo"][t]
            mesh_file = osp.join(args.object_models_dir, f"obj_{uid}.glb")

            if t == 0:
                rr.log(
                    f"world/object_pose/{uid}",
                    rr.Asset3D(
                        path=mesh_file,
                    ),
                )
            rr.log(
                f"world/object_pose/{uid}",
                rr.Transform3D(
                    translation=T_w_o[:3, 3],
                    rotation=rr.Quaternion(
                        xyzw=Rotation.from_matrix(T_w_o[:3, :3]).as_quat()
                    ),
                ),
            )

        # hand pose
        for handedness in ["left", "right"]:
            hand_pose = meta[f"{handedness}_hand_theta"][t]
            shape = meta[f"{handedness}_hand_shape"][t]
            if handedness == "left":
                side = Handedness.Left
            else:
                side = Handedness.Right
            # faces = hand_data_provider.get_hand_mesh_faces_and_normals(side)[0]

            rot, tsl, hA = hand_pose[..., :3], hand_pose[..., 3:6], hand_pose[..., 6:]

            rot = torch.FloatTensor(rot)[None]
            tsl = torch.FloatTensor(tsl)[None]
            hA = torch.FloatTensor(hA)[None]
            shape = torch.FloatTensor(shape)[None]
            hA = sided_wrapper[handedness].pca_to_pose(hA)

            mesh, _ = sided_wrapper[handedness](
                None, hA, axisang=rot, trans=tsl, th_betas=shape
            )
            faces = sided_wrapper[handedness].hand_faces.cpu().numpy()[0]
            vertices = mesh.verts_packed().cpu().numpy()
            rr.log(
                f"world/hand/{handedness}",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                ),
            )
    return


def batch_preprocess():
    # clips = sorted([p for p in os.listdir(args.clips_dir, 'train_aria') if p.endswith(".tar")])
    clips_aria = sorted(glob(os.path.join(args.clips_dir, "train_aria", "*.tar")))
    clips_quest = sorted(glob(os.path.join(args.clips_dir, "train_quest3", "*.tar")))
    clips = clips_aria + clips_quest
    for clip in tqdm(clips):
        lock_file = osp.join(args.preprocess_dir, "lock", os.path.basename(clip))
        done_file = osp.join(args.preprocess_dir, "done", os.path.basename(clip))

        if osp.exists(done_file):
            continue
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            continue

        get_one_clip(clip, vis=False)

        os.makedirs(done_file)
        os.rmdir(lock_file)


def merge_preprocess():
    preprocess_dir = args.preprocess_dir
    meta_list = glob(osp.join(preprocess_dir, "*.npz"))
    data = {}
    for meta_file in meta_list:
        seq = os.path.basename(meta_file).split(".")[0].split("-")[-1]
        # Use context manager to ensure file is properly closed
        with np.load(meta_file) as npz_file:
            meta = dict(npz_file)
        # change uid* to obj_*
        new_meta = {}
        for k in meta.keys():
            # if k.startswith("uid"):
            new_k = k.replace("uid", "obj_")
                # _gt_valid -> _wTo_valid
            new_k = new_k.replace("gt_valid", "wTo_valid")
            new_meta[new_k] = meta[k]
        
        data[seq] = new_meta

    with open(osp.join(preprocess_dir, "dataset.pkl"), "wb") as f:
        pickle.dump(data, f)
    return

def make_own_split():
    orig_split_file = osp.join(args.clips_dir, "clip_splits.json")
    with open(orig_split_file, "r") as f:
        orig_split = json.load(f)
    all_clips  = orig_split['train']['Aria'] + orig_split['train']['Quest3']
    clip_definition_file = osp.join(args.clips_dir, "clip_definitions.json")
    with open(clip_definition_file, "r") as f:
        clip_def = json.load(f)
    clipid2seqid = {int(k): v['sequence_id'] for k, v in clip_def.items()}

    from preprocess.hot3d import hawor_eval_seqs
    test_seqs = set(hawor_eval_seqs)


    split = {
        'train': [],
        'test': [],
    }
    for clip in all_clips:
        seqid = clipid2seqid[clip]
        if seqid in test_seqs:
            split['test'].append(f"{clip:06d}")
        else:
            split['train'].append(f"{clip:06d}")
    print('Total clips:', len(all_clips), '-->', len(split['train']), len(split['test']))
    set_dir = osp.join(args.clips_dir, "sets")
    os.makedirs(set_dir, exist_ok=True)
    with open(osp.join(set_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=4)
    return 


mano_model_folder = "hot3d/hot3d/mano_v1_2/models"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clips_dir",
        type=str,
        help="Path to a folder with clips.",
        default="data/HOT3D-CLIP",
    )
    parser.add_argument(
        "--object_models_dir",
        type=str,
        help="Path to a folder with 3D object models.",
        default="data/HOT3D-CLIP/object_models_eval/",
    )
    parser.add_argument(
        "--mano_model_dir",
        type=str,
        help="Path to a folder with MANO model (MANO_RIGHT/LEFT.pkl files).",
        default="assets/mano/",
    )
    parser.add_argument(
        "--preprocess_dir",
        type=str,
        default="data/HOT3D-CLIP/preprocess/",
    )
    args = parser.parse_args()
    vis_dir = "outputs/vis_hot3d_clips_preprocess/"

    # batch_preprocess()

    merge_preprocess()

    # make_own_split()
