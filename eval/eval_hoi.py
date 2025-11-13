from tqdm import tqdm
from jutils import mesh_utils, geom_utils, image_utils
from pytorch3d.structures import Meshes
from fire import Fire
import os
import os.path as osp
import json
from collections import defaultdict
import math
import torch
import pandas as pd

import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eval.eval_utils import eval_jts
from egorecon.utils.motion_repr import HandWrapper
from eval.eval_utils import eval_pose, csv_to_auc
from egorecon.visualization.pt3d_visualizer import Pt3dVisualizer
from eval.pcl import align_pcl

device = "cuda"
data_dir = "/move/u/yufeiy2/data/HOT3D-CLIP"

mano_model_path = "assets/mano"


def _plot_add_curves(
    per_time_df: pd.DataFrame,
    subset_specs,
    png_path: str,
    add_column: str = "add",
    max_val: float = 0.3,
    step: float = 0.001,
) -> None:
    if add_column not in per_time_df.columns:
        print(f"[AUC] Column '{add_column}' not found. Skip plotting curves.")
        return

    df = per_time_df.copy()
    if "index" in df.columns:
        df = df[df["index"] != "Mean"]

    if df.empty:
        print("[AUC] No rows available for curve plotting.")
        return

    def _compute_curve(errors: np.ndarray):
        if errors.size == 0:
            return None
        xs = np.arange(0.0, max_val + step, step)
        sorted_errs = np.sort(errors)
        counts = np.searchsorted(sorted_errs, xs, side="right")
        ys = counts.astype(np.float64) / float(sorted_errs.size)
        ys = np.clip(ys, 0.0, 1.0)
        auc = np.trapz(ys, xs) / max(max_val, np.finfo(float).eps)
        return xs, ys, auc

    curve_entries = []

    overall_errors = df[add_column].dropna().to_numpy()
    curve = _compute_curve(overall_errors)
    if curve is not None:
        curve_entries.append(("all-test", curve))

    for subset_key, label, _ in subset_specs:
        if subset_key not in df.columns:
            continue
        subset_errors = df[df[subset_key]][add_column].dropna().to_numpy()
        curve = _compute_curve(subset_errors)
        if curve is not None:
            curve_entries.append((label, curve))

    if not curve_entries:
        print("[AUC] No valid curves to plot.")
        return

    num_curves = len(curve_entries)
    cols = min(3, num_curves)
    rows = math.ceil(num_curves / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.2), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (label, (xs, ys, auc)) in enumerate(curve_entries):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.set_visible(True)
        ax.plot(xs, ys, label="ADD curve", color="tab:blue")
        ax.set_xlim(0.0, max_val)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Threshold (m)")
        ax.set_ylabel("Success rate")
        ax.set_title(f"{label} (AUC={auc:.3f})")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved ADD curves to {png_path}")


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


def get_hotclip_split(split="test50obj"):
    split_file = osp.join(data_dir, "sets", "split.json")
    with open(split_file, "r") as f:
        split2sesq = json.load(f)
    seqs = split2sesq[split]
    if "_" in seqs[0]:
        seqs = [seq.split("_")[0] for seq in seqs]
        seqs = list(set(seqs))
    return seqs



@torch.no_grad()
def vis_hotclip(
    pred_file="pred_joints.pkl",
    gt_file="/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
    side="right",
    save_dir=None,
    split="test50obj",
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
        print("seq=", seq)
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


def eval_hotclip_hoi(
    obj_pred_file="", 
    hand_pred_file="",
    gt_file="/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
    save_dir=None,
    split="test50obj",
    skip_not_there=True,
    chunk_length=-1,
    per_traj_align=True,
    ):
    """
    Evaluate HOI by aligning predicted object poses with ground-truth poses
    using hand joints as reference for each frame.
    """
    print("Evaluating HOI with hand-aligned object poses...")

    seqs = get_hotclip_split(split)

    object_mesh_dir = osp.join(data_dir, "object_models_eval")
    object_library = Pt3dVisualizer.setup_template(object_mesh_dir, lib="hotclip", load_mesh=True)

    frame_subsplit_file = osp.join(data_dir, "sets", f"frame_subsplit_{split}.pkl")
    with open(frame_subsplit_file, "rb") as f:
        frame_subsplit = pickle.load(f)

    if isinstance(obj_pred_file, str):
        with open(obj_pred_file, "rb") as f:
            obj_pred_data = pickle.load(f)
    else:
        obj_pred_data = obj_pred_file

    if isinstance(hand_pred_file, str):
        with open(hand_pred_file, "rb") as f:
            hand_pred_data = pickle.load(f)
    else:
        hand_pred_data = hand_pred_file

    if isinstance(gt_file, str):
        with open(gt_file, "rb") as f:
            gt_data = pickle.load(f)
    else:
        gt_data = gt_file

    hand_wrapper = HandWrapper(mano_model_path).to(device)

    metric_list = defaultdict(list)
    per_time_metric_list = defaultdict(list)

    def apply_transform_to_points(transform, points):
        """
        Apply a batch of rigid transforms to point sets.

        Args:
            transform: (T, 4, 4)
            points: (T, N, 3)
        Returns:
            transformed points with shape (T, N, 3)
        """
        rot = transform[:, :3, :3]
        trans = transform[:, :3, 3]
        return torch.einsum("tij,tnj->tni", rot, points) + trans[:, None, :]

    for seq in tqdm(seqs):
        if skip_not_there and (
            seq not in obj_pred_data or seq not in hand_pred_data or seq not in gt_data
        ):
            continue

        obj_pred_seq = obj_pred_data[seq]
        hand_pred_seq = hand_pred_data[seq]
        gt_seq = gt_data[seq]

        left_pred, right_pred = get_hand_joints(hand_pred_seq, hand_wrapper)
        left_gt, right_gt = get_hand_joints(gt_seq, hand_wrapper)

        pred_hands = torch.cat([left_pred, right_pred], dim=1).float()
        gt_hands = torch.cat([left_gt, right_gt], dim=1).float()

        T = gt_hands.shape[0]
        if per_traj_align:
            s_align, R_align, t_align = align_pcl(
                gt_hands.reshape(1, -1, 3), pred_hands.reshape(1, -1, 3), fixed_scale=True
            )
            R_align = R_align.repeat(T, 1, 1)
            t_align = t_align.repeat(T, 1)
        else:
            s_align, R_align, t_align = align_pcl(
                gt_hands, pred_hands, fixed_scale=True
            )

        T_align = torch.eye(4, device=pred_hands.device)[None].repeat(
            T, 1, 1
        )
        T_align[:, :3, :3] = R_align
        T_align[:, :3, 3] = t_align

        total_transform = T_align #  torch.matmul(T_align, wTwp)

        if "objects" in gt_seq:
            object_ids = gt_seq["objects"]
        else:
            object_ids = []
            for key in obj_pred_seq.keys():
                if "wTo" in key and key.startswith("obj_"):
                    obj_id = key.split("_")[1]
                    if obj_id not in object_ids:
                        object_ids.append(obj_id)

        for obj_id in object_ids:
            pred_wTo_key = f"obj_{obj_id}_wTo"
            gt_wTo_key = f"obj_{obj_id}_wTo"

            if pred_wTo_key not in obj_pred_seq:
                pred_wTo_key = f"{obj_id}_wTo"
            if gt_wTo_key not in gt_seq:
                gt_wTo_key = f"{obj_id}_wTo"

            if pred_wTo_key not in obj_pred_seq or gt_wTo_key not in gt_seq:
                continue

            pred_wpTo = torch.FloatTensor(obj_pred_seq[pred_wTo_key]).to(
                pred_hands.device
            )
            gt_wTo = torch.FloatTensor(gt_seq[gt_wTo_key]).to(pred_hands.device)

            frame_count = min(
                pred_wpTo.shape[0], gt_wTo.shape[0], total_transform.shape[0]
            )
            if frame_count == 0:
                continue

            pred_wpTo = pred_wpTo[:frame_count]
            gt_wTo = gt_wTo[:frame_count]
            transform = total_transform[:frame_count]

            pred_wTo = torch.matmul(transform, pred_wpTo)

            mesh_obj = object_library[obj_id]

            metrics = eval_pose(
                gt_wTo.cpu(),
                pred_wTo.cpu(),
                mesh_obj,
                metric_names=["add", "adi", "apd", "center", "acc-norm"],
            )

            for name, values in metrics.items():
                metric_list[name].append(np.mean(values))
                if name == "acc-norm":
                    continue
                per_time_metric_list[name].append(values)

            metric_list["index"].append(f"{seq}_{obj_id}")
            for t in range(frame_count):
                per_time_metric_list["index"].append(f"{seq}_{obj_id}_{t:04d}")

    if len(metric_list["index"]) == 0:
        print("No valid sequences found for evaluation.")
        return

    mean_metrics = {
        name: np.mean(value) for name, value in metric_list.items() if name != "index"
    }
    mean_metrics["index"] = "Mean"

    per_time_metric_cat = {
        name: np.concatenate(values, 0)
        for name, values in per_time_metric_list.items()
        if name != "index"
    }
    index = per_time_metric_list["index"]
    per_time_metric_list = per_time_metric_cat

    mean_per_time_metrics = {
        name: np.mean(values, 0) for name, values in per_time_metric_cat.items()
    }
    mean_per_time_metrics["index"] = "Mean"
    per_time_metric_list["index"] = index

    if save_dir is None:
        if isinstance(obj_pred_file, str):
            save_dir = osp.join(osp.dirname(obj_pred_file))
        else:
            save_dir = osp.join(os.getcwd(), "eval_results")

    os.makedirs(save_dir, exist_ok=True)

    per_time_csv_file = osp.join(
        save_dir, f"hoi_metrics_per_time_chunk_{chunk_length}.csv"
    )

    per_time_metrics_names = [
        name for name in per_time_metric_list.keys() if name != "index"
    ]
    per_time_data = []
    per_time_row = [mean_per_time_metrics["index"]] + [
        mean_per_time_metrics[name] for name in per_time_metrics_names
    ]
    per_time_data.append(per_time_row)
    for i in range(len(per_time_metric_list["index"])):
        per_time_row = [per_time_metric_list["index"][i]] + [
            per_time_metric_list[name][i] for name in per_time_metrics_names
        ]
        per_time_data.append(per_time_row)

    per_time_df = pd.DataFrame(per_time_data, columns=["index"] + per_time_metrics_names)

    subset_specs = [
        ("subset_contact", "contact", lambda contact, trunc: contact == 1),
        ("subset_truncate_1", "truncate_1", lambda contact, trunc: trunc == 1),
        ("subset_truncate_2", "truncate_2", lambda contact, trunc: trunc == 2),
        ("subset_truncate_pos", "truncate_pos", lambda contact, trunc: trunc > 0),
    ]

    subset_masks = {
        spec[0]: np.zeros(len(per_time_df), dtype=bool) for spec in subset_specs
    }

    for row_idx, frame_name in enumerate(per_time_df["index"]):
        if frame_name == "Mean":
            continue
        seq, obj_id, frame_str = frame_name.split("_")
        frame_idx = int(frame_str)
        seq_obj_key = f"{seq}_{obj_id}"
        entry = frame_subsplit[seq_obj_key]
        contact_flag = entry["contact"][frame_idx]
        truncate_flag = entry["truncate"][frame_idx]
        for subset_key, _, predicate in subset_specs:
            if predicate(contact_flag, truncate_flag):
                subset_masks[subset_key][row_idx] = True

    for subset_key, _, _ in subset_specs:
        per_time_df[subset_key] = subset_masks[subset_key]

    per_time_df.to_csv(per_time_csv_file, index=False)
    key_list = ["add", "adi", "center"]
    print("Saved per-time ***HOI*** metrics to csv file:", per_time_csv_file)
    print("Computing AUC per time...")
    print("-" * 30, "HOI", "-" * 30)
    overall_auc = csv_to_auc(per_time_df, silence=True)
    if overall_auc:
        keys = ",".join([key.upper() for key in overall_auc.keys()])
        values = ",".join([f"{overall_auc[key]:.6f}" for key in overall_auc.keys()])
        print(keys)
        print(values)

    print("-" * 30, "HOI Subset AUC", "-" * 30)
    frame_rows = per_time_df["index"] != "Mean"
    subset_rows = [
        ["all-test"]
        + [overall_auc.get(k, np.nan) if overall_auc else np.nan for k in key_list]
        + [float(mean_metrics["acc-norm"])]
    ]
    for subset_key, label, _ in subset_specs:
        subset_df = per_time_df[per_time_df[subset_key] & frame_rows]
        frame_count = len(subset_df)
        print(f"{label} (frames={frame_count})")
        auc_dict = csv_to_auc(subset_df, silence=True)
        if auc_dict:
            keys = ",".join([key.upper() for key in auc_dict.keys()])
            values = ",".join([f"{auc_dict[key]:.6f}" for key in auc_dict.keys()])
            print(keys)
            print(values)
        subset_rows.append(
            [label] + [auc_dict.get(k, np.nan) for k in key_list] + ["NA"]
        )

    subset_columns = ["subset"] + [f"{k}_auc" for k in key_list] + ["acc-norm"]
    subset_csv = pd.DataFrame(subset_rows, columns=subset_columns)
    subset_csv_file = osp.join(save_dir, "hoi_auc.csv")
    subset_csv.to_csv(subset_csv_file, index=False)
    print("Saved HOI subset AUC to csv file:", subset_csv_file)
    _plot_add_curves(per_time_df, subset_specs, subset_csv_file.replace(".csv", ".png"))
    # also print mean acc-norm
    print(f"Mean acc-norm: {mean_metrics['acc-norm']:.6f}")
    return


def eval_hotclip_pose6d(
    pred_file="pred_points.pkl",
    gt_file="/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
    save_dir=None,
    split="test50obj",
    skip_not_there=True,
    chunk_length=-1,
    aligned=False,
    align_rt=False,
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
    print("Evaluating object pose using ADD, ADI, APD metrics...")

    seqs = get_hotclip_split(split)
    
    # Load mesh library
    object_mesh_dir = osp.join(data_dir, "object_models_eval")
    object_library = Pt3dVisualizer.setup_template(object_mesh_dir, lib="hotclip", load_mesh=True)
    
    frame_subsplit_file = osp.join(data_dir, "sets", f"frame_subsplit_{split}.pkl")
    with open(frame_subsplit_file, "rb") as f:
        frame_subsplit = pickle.load(f)

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
    per_time_metric_list = defaultdict(list)
    
    for seq in tqdm(seqs):
        if skip_not_there and (seq not in pred_data or seq not in gt_data):
            continue
        
        pred_seq = pred_data[seq]
        gt_seq = gt_data[seq]

        if not aligned:
            gt_wTc = torch.FloatTensor(gt_seq["wTc"])[0]
            pred_wTc = torch.FloatTensor(pred_seq["wTc"])[0]
            T = len(gt_seq["wTc"])
            
            # align by the 1st frame camera
            wpTc = geom_utils.se3_to_matrix_v2(pred_wTc) # (4, 4)
            wTwp = gt_wTc @ geom_utils.inverse_rt_v2(wpTc)
            wTwp = wTwp[None].repeat(T, 1, 1)
        else:
            T = len(gt_seq["wTc"])
            wTwp = torch.eye(4)[None].repeat(T, 1, 1)

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
                # print(f"Object {obj_id} {pred_wTo_key} not found in pred_seq or gt_seq", pred_wTo_key in pred_seq, gt_wTo_key in gt_seq)
                # print(pred_seq.keys())
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
                metric_names=["add", "adi", "apd", "center", "acc-norm"]
            )
            for name, values in metrics.items():
                # values is array of per-frame metrics, take mean
                metric_list[name].append(np.mean(values))
                
                if name == "acc-norm":
                    continue
                per_time_metric_list[name].append(values)
            metric_list["index"].append(f"{seq}_{obj_id}")
            for t in range(T):
                per_time_metric_list["index"].append(f"{seq}_{obj_id}_{t:04d}")

    # Compute mean metrics
    mean_metrics = {
        name: np.mean(value) for name, value in metric_list.items() if name != "index"
    }
    mean_metrics["index"] = "Mean"

    # firstly concat per_time_metric_list into a single array
    per_time_metric_cat = {name: np.concatenate(values, 0) for name, values in per_time_metric_list.items() if name != "index"}
    index = per_time_metric_list["index"]
    per_time_metric_list = per_time_metric_cat

    mean_per_time_metrics = {name: np.mean(values, 0) for name, values in per_time_metric_cat.items()}
    mean_per_time_metrics["index"] = "Mean"
    per_time_metric_list["index"] = index
    
    # Save to CSV
    if save_dir is None:
        save_dir = osp.join(osp.dirname(pred_file))
    os.makedirs(save_dir, exist_ok=True)
    
    # Save per-time metrics to CSV
    per_time_csv_file = osp.join(save_dir, f"object_pose_metrics_per_time_chunk_{chunk_length}.csv")
    
    per_time_metrics_names = [name for name in per_time_metric_list.keys() if name != "index"]
    per_time_data = []
    # Add mean_per_time_metrics as first row
    per_time_row = [mean_per_time_metrics["index"]] + [mean_per_time_metrics[name] for name in per_time_metrics_names]
    per_time_data.append(per_time_row)
    # Add individual per-time metrics
    for i in range(len(per_time_metric_list["index"])):
        per_time_row = [per_time_metric_list["index"][i]] + [
            per_time_metric_list[name][i] for name in per_time_metrics_names
        ]
        per_time_data.append(per_time_row)
    
    per_time_df = pd.DataFrame(per_time_data, columns=["index"] + per_time_metrics_names)

    subset_specs = [
        ("subset_contact", "contact", lambda contact, trunc: contact == 1),
        ("subset_truncate_1", "truncate_1", lambda contact, trunc: trunc == 1),
        ("subset_truncate_2", "truncate_2", lambda contact, trunc: trunc == 2),
        ("subset_truncate_pos", "truncate_pos", lambda contact, trunc: trunc > 0),
    ]

    subset_masks = {
        spec[0]: np.zeros(len(per_time_df), dtype=bool) for spec in subset_specs
    }

    for row_idx, frame_name in enumerate(per_time_df["index"]):
        if frame_name == "Mean":
            continue
        seq, obj_id, frame_str = frame_name.split("_")
        frame_idx = int(frame_str)
        seq_obj_key = f"{seq}_{obj_id}"
        entry = frame_subsplit[seq_obj_key]
        contact_flag = entry["contact"][frame_idx]
        truncate_flag = entry["truncate"][frame_idx]
        for subset_key, _, predicate in subset_specs:
            if predicate(contact_flag, truncate_flag):
                subset_masks[subset_key][row_idx] = True

    for subset_key, _, _ in subset_specs:
        per_time_df[subset_key] = subset_masks[subset_key]

    per_time_df.to_csv(per_time_csv_file, index=False)
    key_list = ["add", "adi", "center"]
    print("Saved per-time metrics to csv file: ", per_time_csv_file)
    print("Computing AUC per time...")
    print("-" * 30, "Object Pose", "-" * 30)
    overall_auc = csv_to_auc(per_time_df, silence=True)
    if overall_auc:
        keys = ",".join([key.upper() for key in overall_auc.keys()])
        values = ",".join([f"{overall_auc[key]:.6f}" for key in overall_auc.keys()])
        print(keys)
        print(values)

    print("-" * 30, "Object Pose Subset AUC", "-" * 30)
    frame_rows = per_time_df["index"] != "Mean"
    subset_rows = [
        ["all-test"]
        + [overall_auc.get(k, np.nan) if overall_auc else np.nan for k in key_list]
        + [float(mean_metrics["acc-norm"])]
    ]
    for subset_key, label, _ in subset_specs:
        subset_df = per_time_df[per_time_df[subset_key] & frame_rows]
        frame_count = len(subset_df)
        print(f"{label} (frames={frame_count})")
        if frame_count == 0:
            subset_rows.append([label] + [np.nan for _ in key_list] + ["NA"])
            continue
        auc_dict = csv_to_auc(subset_df, silence=True)
        if auc_dict:
            keys = ",".join([key.upper() for key in auc_dict.keys()])
            values = ",".join([f"{auc_dict[key]:.6f}" for key in auc_dict.keys()])
            print(keys)
            print(values)
        subset_rows.append(
            [label] + [auc_dict.get(k, np.nan) for k in key_list] + ["NA"]
        )

    subset_columns = ["subset"] + [f"{k}_auc" for k in key_list] + ["acc-norm"]
    subset_csv = pd.DataFrame(subset_rows, columns=subset_columns)
    subset_csv_file = osp.join(save_dir, "pose_auc.csv")
    subset_csv.to_csv(subset_csv_file, index=False)
    print("Saved Object Pose subset AUC to csv file:", subset_csv_file)
    _plot_add_curves(per_time_df, subset_specs, subset_csv_file.replace(".csv", ".png"))
    print(f"Mean acc-norm: {mean_metrics['acc-norm']:.6f}")


def eval_hotclip_joints(
    pred_file="pred_joints.pkl",
    gt_file="/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
    side="both",
    save_dir=None,
    split="test50obj",
    skip_not_there=True,
    chunk_length=-1,
    force_fk=True,
):
    seqs = get_hotclip_split(split)
    device = "cuda:0"

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
            continue
        pred_seq = pred_data[seq]
        gt_seq = gt_data[seq]

        left_gt, right_gt = get_hand_joints(gt_seq, hand_wrapper, force_fk=force_fk)
        left_pred, right_pred = get_hand_joints(pred_seq, hand_wrapper, force_fk=force_fk   )  # (T, J, 3)
        print("left pred", left_pred.device, left_gt.device)
        left_pred = left_pred.to(device)
        right_pred = right_pred.to(device)
        left_gt = left_gt.to(device)
        right_gt = right_gt.to(device)

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
    mean_metrics = {
        name: np.mean(value) for name, value in metric_list.items() if name != "index"
    }
    mean_metrics["index"] = "Mean"

    # pretty print mean
    print("Mean metrics:")
    key_list = ["pampjpe", "fa_jmse", "ga_jmse", "acc_norm"]
    key_str = ",".join(key_list)
    value_str = ",".join([f"{mean_metrics[name]:f}" for name in key_list])
    print(key_str)
    print(value_str)
    print("-" * 30)

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



def main(mode='hand',  **kwargs):
    if mode == 'hand':
        eval_hotclip_joints(**kwargs)
    elif mode == 'pose6d':
        eval_hotclip_pose6d(**kwargs)
    elif mode == 'hoi':
        eval_hotclip_hoi(**kwargs)
    elif mode == 'vis':
        vis_hotclip(**kwargs)


if __name__ == "__main__":
    split = "test"
    gt_file = "dataset_contact.pkl"
    pred_file = "pred_joints.pkl"
    # eval_hotclip(pred_file, gt_file)
    Fire(main)
