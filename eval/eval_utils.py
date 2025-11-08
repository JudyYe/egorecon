# from slamhr and haptic

import json
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from fire import Fire
from .pcl import align_pcl
from . import bop_utils as misc
from scipy import spatial
from jutils import mesh_utils
def w_rje(gt, pred):
    """world relative joint error

    :param pred: (T, 3)
    :param gt: (T, 3)
    """
    return 


def wa_rje(gt, pred):
    """world aligned trajectory

    :param pred: tensor in shape of (T, 3) 
    :param gt: tensor in shape of (T, 3)
    """
    aligned_pred = global_align_joints(gt[:, None], pred[:, None])
    aligned_pred = aligned_pred.squeeze(1)
    res = torch.linalg.norm(gt - pred, dim=-1).mean()
    return res


def apd(R_est, t_est, R_gt, t_gt, pts, thre_list=[0.01, 0.02, 0.05, 0.1]):
    """(Average percent of Points within Delta (APD))
    :return: dict with {thre: value} for each threshold
    """
    apd_dict = {}
    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)
    for thre in thre_list:
        dist = np.linalg.norm(pts_est - pts_gt, axis=1)
        num_points = dist < thre
        num_points = num_points.sum()
        # Convert to percentage (fraction of points within threshold)
        apd_value = num_points / len(pts) if len(pts) > 0 else 0.0
        apd_dict[thre] = apd_value
    
    return apd_dict



def compute_auc_sklearn(errs, max_val=0.1, step=0.001):
  from sklearn import metrics
  errs = np.sort(np.array(errs))
  X = np.arange(0, max_val+step, step)
  Y = np.ones(len(X))
  for i,x in enumerate(X):
    y = (errs<=x).sum()/len(errs)
    Y[i] = y
    if y>=1:
      break
  auc = metrics.auc(X, Y) / (max_val*1)
  return auc



def add(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with no indistinguishable
    views - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

def center(R_est, t_est, R_gt, t_gt, pts):

    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)  # (P, 3)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)
    
    pts_mean = np.mean(pts_est, axis=0) # (3, )
    pts_gt_mean = np.mean(pts_gt, axis=0)
    e = np.linalg.norm(pts_mean - pts_gt_mean)
    return e

def adi(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e

def eval_pose(gt_wTo, pred_wTo, mesh, metric_names=["add", "adi", "apd", "center", "acc-norm"]):
    # Handle apd separately since it returns multiple thresholds
    apd_thre_list = [0.01, 0.02, 0.05, 0.1]  # Default thresholds for apd
    apd_metric_names = [f"apd@{thre}" for thre in apd_thre_list] if "apd" in metric_names else []
    
    # Initialize metrics - separate entries for each apd threshold
    all_metric_names = [name for name in metric_names if name != "apd"] + apd_metric_names
    cur_metrics = {name: [] for name in all_metric_names}


    if "acc-norm" in metric_names:
        points = mesh.verts_padded()
        T = gt_wTo.shape[0]
        points_exp = points.repeat(T, 1, 1)

        gt_wPoints = mesh_utils.apply_transform(points_exp, gt_wTo)
        pred_wPoints = mesh_utils.apply_transform(points_exp, pred_wTo)  # (T, P, 3)

        # use slamhr's compute_accel_norm
        gt_accel = compute_accel_norm(gt_wPoints)  # (T - 2, P, )
        pred_accel = compute_accel_norm(pred_wPoints)
        score = torch.linalg.norm(gt_accel - pred_accel, dim=-1)
        cur_metrics["acc-norm"].append(score.cpu().numpy())  

    gt_wTo = gt_wTo.cpu().numpy()
    pred_wTo = pred_wTo.cpu().numpy()

    gt_wTo_rot, gt_wTo_t = gt_wTo[..., :3, :3], gt_wTo[..., :3, 3:4]
    pred_wTo_rot, pred_wTo_t = pred_wTo[..., :3, :3], pred_wTo[..., :3, 3:4]

    for t in range(gt_wTo.shape[0]):
        points = mesh.verts_packed().detach().cpu().numpy()  # (P, 3)
        
        for name in metric_names:
            if name == "add":
                score = add(gt_wTo_rot[t], gt_wTo_t[t], pred_wTo_rot[t], pred_wTo_t[t], points)
                cur_metrics[name].append(score)
            elif name == "adi":
                score = adi(gt_wTo_rot[t], gt_wTo_t[t], pred_wTo_rot[t], pred_wTo_t[t], points)
                cur_metrics[name].append(score)
            elif name == "center":
                score = center(gt_wTo_rot[t], gt_wTo_t[t], pred_wTo_rot[t], pred_wTo_t[t], points)
                cur_metrics[name].append(score)
            elif name == "apd":
                # apd returns a dict with thresholds as keys: {thre: value}
                apd_dict = apd(gt_wTo_rot[t], gt_wTo_t[t], pred_wTo_rot[t], pred_wTo_t[t], points, thre_list=apd_thre_list)
                # Convert dict to separate metric entries
                for thre in apd_thre_list:
                    apd_key = f"apd@{thre}"
                    value = apd_dict[thre]  # apd_dict[thre] is now a scalar value
                    cur_metrics[apd_key].append(value)
            elif name == "acc-norm":
                continue
            else:
                raise NotImplementedError(f"metric {name} not implemented")
    
    # Convert lists to numpy arrays
    cur_metrics = {name: np.array(cur_metrics[name]) for name in all_metric_names}
    return cur_metrics


def eval_jts(gt_seq_joints, res_joints, metric_names=["ga_jmse", "ga_rmse", "fa_jmse", "acc_norm", "pampjpe",]):
    """
    :param gt_seq_joints: (T, J, 3)
    :param res_joints: (T, J, 3)
    :param metric_names: list of metric names
    :return: dictionary of metrics
    """
    cur_metrics = {name: np.nan for name in metric_names}
    for name in metric_names:
        if name == "acc_norm":
            # this is from slamhr
            target = compute_accel_norm(gt_seq_joints)  # (T-2, J)
            pred = compute_accel_norm(res_joints)
        else:
            target = gt_seq_joints
            if name == "pampjpe":
                pred = local_align_joints(gt_seq_joints, res_joints)
            elif name == "ga_jmse":
                pred = global_align_joints(gt_seq_joints, res_joints)
            elif name == "fa_jmse":
                pred = first_align_joints(gt_seq_joints, res_joints)
            elif name == "ga_rmse":
                pred = global_align_joints(gt_seq_joints, res_joints)
                pred = pred[..., 5, :]
                target = target[..., 5, :]
            elif name == 'jmse':
                pred = res_joints
            else:
                raise NotImplementedError(f"metric {name} not implemented")
        cur_metrics[name] = torch.linalg.norm(target - pred, dim=-1).mean().item()
    return cur_metrics

def compute_accel_norm(joints):
    """
    :param joints (T, J, 3)
    """
    vel = joints[1:] - joints[:-1]  # (T-1, J, 3)
    acc = vel[1:] - vel[:-1]  # (T-2, J, 3)
    return torch.linalg.norm(acc, dim=-1)



def compute_error_accel(joints_gt, joints_pred, vis=None):
    
    """
    from hawor? 
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)  # (T, xxx)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)



def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(
        gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3)
    )
    pred_glob = (
        s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    )  # (T, J, 3)
    return pred_glob


def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(
        gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3)
    )
    pred_first = (
        s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    )
    return pred_first


def local_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_loc, R_loc, t_loc = align_pcl(gt_joints, pred_joints)
    pred_loc = (
        s_loc[:, None] * torch.einsum("tij,tnj->tni", R_loc, pred_joints)
        + t_loc[:, None]
    )
    return pred_loc


def csv_to_auc(csv_file='/move/u/yufeiy2/egorecon/outputs/ready/ours/eval_hoi_contact_ddim_long_shelf/object_pose_metrics_chunk_-1.csv', max_val=0.3, step=0.001, silence=False):
    """
    Compute AUC for ADD, ADI, and center metrics from CSV file.
    
    Args:
        csv_file: Path to CSV file with metrics
        max_val: Maximum value for AUC computation (default: 0.1)
        step: Step size for AUC computation (default: 0.001)
    """
    # Read CSV file
    dataframe_input = not isinstance(csv_file, str)
    if not dataframe_input:
        df = pd.read_csv(csv_file)
    else:
        df = csv_file
    
    # Set index column if it exists
    if 'index' in df.columns:
        df = df.set_index('index')
    
    # Remove "Mean" row if it exists
    if 'Mean' in df.index:
        df = df.drop('Mean')
    
    # Extract error values for add, adi, center
    metrics_to_compute = ['add', 'adi', 'center']
    auc_results = {}
    
    for metric_name in metrics_to_compute:
        if metric_name not in df.columns:
            print(f"Warning: {metric_name} column not found in CSV file")
            continue
        
        # Extract error values (ignore NaN values)
        errors = df[metric_name].dropna().values
        
        if len(errors) == 0:
            print(f"Warning: No valid values found for {metric_name}")
            auc_results[metric_name] = 0.0
            continue
        
        # Compute AUC
        auc = compute_auc_sklearn(errors, max_val=max_val, step=step)
        auc_results[metric_name] = float(auc)
        
        # Print result
        # if not silence:
            # print(f"{metric_name.upper()} AUC: {auc:.6f} (from {len(errors)} samples)")
    
    # Print summary
    if not silence:
        print("\n" + "="*50)
        print("AUC Summary: ")
        # separate by comma
        keys = ",".join([key.upper() for key in auc_results.keys()])
        values = ",".join([str(auc_results[key]) for key in auc_results.keys()])
        print(keys)
        print(values)
        print("="*50)
    
    # # Save to JSON file
    # if not dataframe_input:
    #     csv_dir = osp.dirname(csv_file)
    #     csv_basename = osp.basename(csv_file)
    #     csv_name_without_ext = osp.splitext(csv_basename)[0]
    #     json_file = osp.join(csv_dir, f"{csv_name_without_ext}_auc.json")
        
    #     os.makedirs(csv_dir, exist_ok=True)
    #     with open(json_file, 'w') as f:
    #         json.dump(auc_results, f, indent=4)
        
    #     print(f"\nAUC results saved to: {json_file}")
    
    return auc_results
    




if __name__ == "__main__":
    Fire(csv_to_auc)