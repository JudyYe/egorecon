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

device = "cuda"
data_dir = "/move/u/yufeiy2/data/HOT3D-CLIP"

mano_model_path = "assets/mano"
def get_hand_joints(seq, hand_wrapper: HandWrapper):
    if "left_joints" not in seq:
        left_theta = torch.FloatTensor(seq["left_hand_theta"]).to(device)
        left_shape = torch.FloatTensor(seq["left_hand_shape"]).to(device)
        _, _, left_joints = hand_wrapper.hand_para2verts_faces_joints(left_theta, left_shape, side="left")
    else:
        left_joints = seq["left_joints"]
    if "right_joints" not in seq:
        right_theta = torch.FloatTensor(seq["right_hand_theta"]).to(device)
        right_shape = torch.FloatTensor(seq["right_hand_shape"]).to(device)
        _, _, right_joints = hand_wrapper.hand_para2verts_faces_joints(right_theta, right_shape, side="right")
    else:
        right_joints = seq["right_joints"]
    return left_joints, right_joints

def get_hotclip_split(split="test"):
    split_file = osp.join(data_dir, "sets", "split.json")
    with open(split_file, "r") as f:
        split2sesq = json.load(f)
    seqs = split2sesq[split]
    return seqs

def eval_hotclip(pred_file, gt_file, side="right", save_dir="outputs/", split="test", skip_not_there=False, chunk_length=100):
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
                metrics = eval_jts(gt_chunk, pred_chunk, metric_names=["ga_jmse", "fa_jmse", "acc_norm", "pampjpe",])
                for name, value in metrics.items():
                    metric_list[name].append(value)
                metric_list["index"].append(f"{seq}_{start}_{end}")
        else:
            metrics = eval_jts(gt, pred, metric_names=["ga_jmse","fa_jmse", "acc_norm", "pampjpe",])
            for name, value in metrics.items():
                metric_list[name].append(value)
            metric_list["index"].append(seq)
        # print(seq, metrics)
        # for name, value in metrics.items():
        #     metric_list[name].append(value)

    # save metrics as csv, column: metrics, rows: 1st Mean, 2nd to last : each indivisual. 

    mean_metrics = {name: np.mean(value) for name, value in metric_list.items() if name != "index"}
    mean_metrics["index"] = "Mean"

    
    # pretty print mean
    print("Mean metrics:")
    for name, value in mean_metrics.items():
        if name != "index":
            print(f"{name}: {value:.4f}")
    print("-" * 100)
    
    # column: index, metrics1, metrics2, ... rows: each individual sequence + mean
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
        row = [metric_list["index"][i]] + [metric_list[name][i] for name in metrics_names]
        data.append(row)
    df = pd.DataFrame(data, columns=["index"] + metrics_names)
    df.to_csv(csv_file, index=False)
    print("Saved metrics to csv file: ", csv_file)
    return 

if __name__ == "__main__":
    split = "test"
    gt_file = "dataset_contact.pkl"
    pred_file = "pred_joints.pkl"
    eval_hotclip(pred_file, gt_file)
