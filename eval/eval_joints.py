from jutils import mesh_utils, geom_utils, image_utils, plot_utils
from pytorch3d.structures import Meshes
from fire import Fire
import os
import os.path as osp
import json
from collections import defaultdict
import torch
import pandas as pd

import pickle
import numpy as np
from eval.eval_utils import eval_jts
from egorecon.utils.motion_repr import HandWrapper
from eval.eval_utils import eval_pose
from egorecon.visualization.pt3d_visualizer import Pt3dVisualizer
from pytorch3d.structures import Meshes
from jutils import mesh_utils

device = "cuda"
data_dir = "/move/u/yufeiy2/data/HOT3D-CLIP"

mano_model_path = "assets/mano"



def get_hand_joints(seq, hand_wrapper: HandWrapper, force_fk=True):
    if "left_joints" not in seq or force_fk:
        if "left_hand_theta" not in seq:
            seq["left_hand_theta"], seq["left_hand_shape"] = seq["left_hand_params"][..., :-10], seq["left_hand_params"][..., -10:]

        left_theta = torch.FloatTensor(seq["left_hand_theta"]).to(device)
        left_shape = torch.FloatTensor(seq["left_hand_shape"]).to(device)
        _, _, left_joints = hand_wrapper.hand_para2verts_faces_joints(
            left_theta, left_shape, side="left"
        )
    else:
        left_joints = seq["left_joints"]
    if "right_joints" not in seq or force_fk:
        if "right_hand_theta" not in seq:
            seq["right_hand_theta"], seq["right_hand_shape"] = seq["right_hand_params"][..., :-10], seq["right_hand_params"][..., -10:]
            
        right_theta = torch.FloatTensor(seq["right_hand_theta"]).to(device)
        right_shape = torch.FloatTensor(seq["right_hand_shape"]).to(device)
        _, _, right_joints = hand_wrapper.hand_para2verts_faces_joints(
            right_theta, right_shape, side="right"
        )
    else:
        right_joints = seq["right_joints"]
    return left_joints, right_joints


def get_hand_meshes(seq, hand_wrapper: HandWrapper):

    if "left_hand_theta" not in seq:
        seq["left_hand_theta"], seq["left_hand_shape"] = seq["left_hand_params"][..., :-10], seq["left_hand_params"][..., -10:]

    left_theta = torch.FloatTensor(seq["left_hand_theta"]).to(device)
    left_shape = torch.FloatTensor(seq["left_hand_shape"]).to(device)
    left_verts, left_faces, _ = hand_wrapper.hand_para2verts_faces_joints(
        left_theta, left_shape, side="left"
    )
    
    if "right_hand_theta" not in seq:
        seq["right_hand_theta"], seq["right_hand_shape"] = seq["right_hand_params"][..., :-10], seq["right_hand_params"][..., -10:]
        
    right_theta = torch.FloatTensor(seq["right_hand_theta"]).to(device)
    right_shape = torch.FloatTensor(seq["right_hand_shape"]).to(device)
    right_verts, right_faces, _ = hand_wrapper.hand_para2verts_faces_joints(
        right_theta, right_shape, side="right"
    )
    
    left_meshes = Meshes(verts=left_verts, faces=left_faces).to(device)
    right_meshes = Meshes(verts=right_verts, faces=right_faces).to(device)
    return left_meshes, right_meshes


def get_hotclip_split(split="test"):
    split_file = osp.join(data_dir, "sets", "split.json")
    with open(split_file, "r") as f:
        split2sesq = json.load(f)
    seqs = split2sesq[split]
    return seqs



@torch.no_grad()
def vis_hotclip(
    pred_file="pred_joints.pkl",
    gt_file="/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
    side="right",
    save_dir=None,
    split="test",
    skip_not_there=True,
    chunk_length=-1,
):

    seqs = get_hotclip_split(split)
    if isinstance(pred_file, str):
        with open(pred_file, "rb") as f:
            pred_data = pickle.load(f)
    else:
        pred_data = pred_file
    if isinstance(gt_file, str):
        with open(gt_file, "rb") as f:
            gt_data = pickle.load(f)
    else:
        gt_data = gt_file

    hand_wrapper = HandWrapper(mano_model_path).to(device)
    if save_dir is None:
        save_dir = osp.join(osp.dirname(pred_file), "vis")
    os.makedirs(save_dir, exist_ok=True)

    for seq in seqs:
        if skip_not_there and (seq not in pred_data or seq not in gt_data):
            print(f"Seq {seq} not in pred_data or gt_data")
            continue
        
        pred_seq = pred_data[seq]
        gt_seq = gt_data[seq]
        left_pred, right_pred = get_hand_meshes(pred_seq, hand_wrapper)
        pred_hands = mesh_utils.join_scene([left_pred, right_pred])
        pred_hands.textures = mesh_utils.pad_texture(pred_hands, 'blue')
        left_gt, right_gt = get_hand_meshes(gt_seq, hand_wrapper)
        gt_hands = mesh_utils.join_scene([left_gt, right_gt])
        gt_hands.textures = mesh_utils.pad_texture(gt_hands, 'red')

        scene = mesh_utils.join_scene([pred_hands, gt_hands])
        nTw = mesh_utils.get_nTw(scene.verts_packed()[None], new_scale=1.2)

        image_list = mesh_utils.render_geom_rot_v2(scene, nTw=nTw, time_len=1)  # (time_len, B, H, W, 3)
        H, W = image_list[0].shape[-2:]
        image_list = torch.stack(image_list, axis=0).reshape(-1, 1, 3, H, W)
        image_utils.save_gif(image_list, osp.join(save_dir, f"{seq}"), fps=30, ext=".mp4")



def eval_hotclip_pose6d(
    pred_file="pred_points.pkl",
    gt_file="/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
    save_dir=None,
    split="test",
    skip_not_there=True,
    chunk_length=-1,
):
    """
    Evaluate object pose using ADD, ADI, APD metrics.
    
    Args:
        pred_file: Prediction file with wTo per object (e.g., aggregated_prediction.pkl)
        gt_file: Ground truth file
        save_dir: Directory to save results
        split: Dataset split
        skip_not_there: Skip sequences not in pred/gt
        chunk_length: Chunk length (-1 for full sequence)
    """
    
    seqs = get_hotclip_split(split)
    
    # Load mesh library
    object_mesh_dir = osp.join(data_dir, "object_models_eval")
    object_library = Pt3dVisualizer.setup_template(object_mesh_dir, lib="hotclip", load_mesh=True)
    
    if isinstance(pred_file, str):
        with open(pred_file, "rb") as f:
            pred_data = pickle.load(f)
    else:
        pred_data = pred_file
    if isinstance(gt_file, str):
        with open(gt_file, "rb") as f:
            gt_data = pickle.load(f)
    else:
        gt_data = gt_file
    
    metric_list = defaultdict(list)
    
    for seq in seqs:
        if skip_not_there and (seq not in pred_data or seq not in gt_data):
            continue
        
        pred_seq = pred_data[seq]
        gt_seq = gt_data[seq]
        
        gt_wTc = torch.FloatTensor(gt_seq["wTc"])[0]
        pred_wTc = torch.FloatTensor(pred_seq["wTc"])[0]
        T = len(gt_seq["wTc"])
        
        # align by the 1st frame camera
        wpTc = geom_utils.se3_to_matrix_v2(pred_wTc) # (4, 4)
        wTwp = gt_wTc @ geom_utils.inverse_rt_v2(wpTc)
        wTwp = wTwp[None].repeat(T, 1, 1)

        # Get object IDs from pred_seq (format: obj_{obj_id}_wTo or similar)
        # Or from gt_seq which should have objects list
        if "objects" in gt_seq:
            object_ids = gt_seq["objects"]
        else:
            # Try to extract from keys in pred_seq
            object_ids = []
            for key in pred_seq.keys():
                if "wTo" in key and key.startswith("obj_"):
                    obj_id = key.split("_")[1]  # Extract object ID
                    if obj_id not in object_ids:
                        object_ids.append(obj_id)
        
        for obj_id in object_ids:
            # Get wTo for this object
            pred_wTo_key = f"obj_{obj_id}_wTo"
            gt_wTo_key = f"obj_{obj_id}_wTo"
            
            # Try alternative key formats
            if pred_wTo_key not in pred_seq:
                # Try without prefix
                pred_wTo_key = f"{obj_id}_wTo"
            if gt_wTo_key not in gt_seq:
                gt_wTo_key = f"{obj_id}_wTo"
            
            if pred_wTo_key not in pred_seq or gt_wTo_key not in gt_seq:
                # print(f"Warning: {pred_wTo_key} or {gt_wTo_key} not found for seq {seq}, obj {obj_id}")
                continue
            
            pred_wpTo = torch.FloatTensor(pred_seq[pred_wTo_key])
            pred_wTo = wTwp @ pred_wpTo
            gt_wTo = torch.FloatTensor(gt_seq[gt_wTo_key])
            
            # Get mesh for this object
            # Try different formats for object ID
            mesh_obj = object_library[obj_id]
            
            # Create mesh list for each frame (same mesh for all frames)
            T = pred_wTo.shape[0]

            metrics = eval_pose(
                gt_wTo,
                pred_wTo,
                mesh_obj,
                metric_names=["add", "adi", "apd", "center"]
            )
            for name, values in metrics.items():
                # values is array of per-frame metrics, take mean
                metric_list[name].append(np.mean(values))
            metric_list["index"].append(f"{seq}_{obj_id}")

    # Compute mean metrics
    mean_metrics = {
        name: np.mean(value) for name, value in metric_list.items() if name != "index"
    }
    mean_metrics["index"] = "Mean"
    
    # Print mean metrics
    print("Mean object pose metrics:")
    for name, value in mean_metrics.items():
        if name != "index":
            print(f"{name}: {value:.4f}")
    print("-" * 100)
    
    # Save to CSV
    if save_dir is None:
        save_dir = osp.join(osp.dirname(pred_file))
    os.makedirs(save_dir, exist_ok=True)
    csv_file = osp.join(save_dir, f"object_pose_metrics_chunk_{chunk_length}.csv")
    
    metrics_names = [name for name in metric_list.keys() if name != "index"]
    data = []
    # Add mean_metrics as first row
    row = [mean_metrics["index"]] + [mean_metrics[name] for name in metrics_names]
    data.append(row)
    # Add individual metrics
    for i in range(len(metric_list["index"])):
        row = [metric_list["index"][i]] + [
            metric_list[name][i] for name in metrics_names
        ]
        data.append(row)
    
    df = pd.DataFrame(data, columns=["index"] + metrics_names)
    df.to_csv(csv_file, index=False)
    print("Saved metrics to csv file: ", csv_file)
    return 

def eval_hotclip_joints(
    pred_file="pred_joints.pkl",
    gt_file="/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
    side="both",
    save_dir=None,
    split="test",
    skip_not_there=True,
    chunk_length=-1,
):
    seqs = get_hotclip_split(split)

    hand_wrapper = HandWrapper(mano_model_path).to(device)
    if isinstance(pred_file, str):
        with open(pred_file, "rb") as f:
            pred_data = pickle.load(f)
    else:
        pred_data = pred_file
    if isinstance(gt_file, str):
        with open(gt_file, "rb") as f:
            gt_data = pickle.load(f)
    else:
        gt_data = gt_file

    metric_list = defaultdict(list)
    # for seq in gt_data.keys():
    for seq in seqs:
        if skip_not_there and (seq not in pred_data or seq not in gt_data):
            print(f"Seq {seq} not in pred_data or gt_data")
            continue
        pred_seq = pred_data[seq]
        gt_seq = gt_data[seq]

        left_pred, right_pred = get_hand_joints(pred_seq, hand_wrapper)  # (T, J, 3)
        left_gt, right_gt = get_hand_joints(gt_seq, hand_wrapper)

        if side == "right":
            pred = right_pred
            gt = right_gt
        elif side == "left":
            pred = left_pred
            gt = left_gt
        elif side == "both":
            pred = torch.cat([left_pred, right_pred], 0)
            gt = torch.cat([left_gt, right_gt], 0)

        if chunk_length > 0:
            T = pred.shape[0]
            for start in range(0, T, chunk_length):
                end = min(T, start + chunk_length)
                pred_chunk = pred[start:end]
                gt_chunk = gt[start:end]
                metrics = eval_jts(
                    gt_chunk,
                    pred_chunk,
                    metric_names=[
                        "ga_jmse",
                        "fa_jmse",
                        "acc_norm",
                        "pampjpe",
                    ],
                )
                for name, value in metrics.items():
                    metric_list[name].append(value)
                metric_list["index"].append(f"{seq}_{start}_{end}")
        else:
            metrics = eval_jts(
                gt,
                pred,
                metric_names=[
                    "ga_jmse",
                    "fa_jmse",
                    "acc_norm",
                    "pampjpe",
                ],
            )
            for name, value in metrics.items():
                metric_list[name].append(value)
            metric_list["index"].append(seq)
        # print(seq, metrics)
        # for name, value in metrics.items():
        #     metric_list[name].append(value)

    # save metrics as csv, column: metrics, rows: 1st Mean, 2nd to last : each indivisual.

    mean_metrics = {
        name: np.mean(value) for name, value in metric_list.items() if name != "index"
    }
    mean_metrics["index"] = "Mean"

    # pretty print mean
    print("Mean metrics:")
    for name, value in mean_metrics.items():
        if name != "index":
            print(f"{name}: {value:.4f}")
    print("-" * 100)

    # column: index, metrics1, metrics2, ... rows: each individual sequence + mean
    if save_dir is None:
        save_dir = osp.join(osp.dirname(pred_file))
    os.makedirs(save_dir, exist_ok=True)
    csv_file = osp.join(save_dir, f"hand_metrics_chunk_{chunk_length}.csv")
    df = pd.DataFrame()
    # Create a DataFrame where the first row is the mean metrics, followed by each sequence/chunk metrics
    metrics_names = [name for name in metric_list.keys() if name != "index"]
    data = []
    # Add mean_metrics as the first row
    row = [mean_metrics["index"]] + [mean_metrics[name] for name in metrics_names]
    data.append(row)
    # Add individual sequence/chunk metrics
    for i in range(len(metric_list["index"])):
        row = [metric_list["index"][i]] + [
            metric_list[name][i] for name in metrics_names
        ]
        data.append(row)
    df = pd.DataFrame(data, columns=["index"] + metrics_names)
    df.to_csv(csv_file, index=False)
    print("Saved metrics to csv file: ", csv_file)
    return



def main(mode='quant',  **kwargs):
    if mode == 'quant':
        eval_hotclip_joints(**kwargs)
    elif mode == 'pose6d':
        eval_hotclip_pose6d(**kwargs)
    elif mode == 'vis':
        vis_hotclip(**kwargs)


if __name__ == "__main__":
    split = "test"
    gt_file = "dataset_contact.pkl"
    pred_file = "pred_joints.pkl"
    # eval_hotclip(pred_file, gt_file)
    Fire(main)
