# from slamhr and haptic

import numpy as np
import torch
from .pcl import align_pcl



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
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
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

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

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
    )
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