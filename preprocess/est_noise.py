import pickle
import os
from collections import defaultdict
from tqdm import tqdm
import rerun as rr
import torch
import os.path as osp
from glob import glob
import numpy as np
from scipy.ndimage import gaussian_filter
from egorecon.utils.motion_repr import HandWrapper
from egorecon.manip.data.utils import load_pickle

mano_model_folder = "assets/mano"
device = "cuda:0"
sided_mano_model = HandWrapper(mano_model_folder).to(device)


def compute_trajectory_noise_stats(gt_vec, est_vec):
    """Compute trajectory-level and per-frame noise statistics.

    Args:
        gt_traj: (BS, T, D) ground truth value
        est_traj: (BS, T, D) estimated value

    Returns:
        traj_std: (9,) trajectory-level noise std [6D rot + 3D trans]
        frame_std: (9,) per-frame jittering noise std [6D rot + 3D trans]
    """
    # Convert to vector representation
    # gt_vec = se3_to_vec(gt_traj)  # (BS, T, 9)
    # est_vec = se3_to_vec(est_traj)  # (BS, T, 9)

    D = gt_vec.shape[-1]
    # Compute residual
    residual = est_vec - gt_vec  # (BS, T, 9)

    # Trajectory-level noise: mean across time, then std across batch
    traj_mean_residual = residual.mean(dim=1)  # (BS, 9)
    traj_std = traj_mean_residual.std(dim=0)  # (9,)

    # Per-frame noise: remove trajectory bias, then compute std
    frame_residual = residual - traj_mean_residual.unsqueeze(1)  # (BS, T, 9)

    frame_std = frame_residual.reshape(-1, D).std(dim=0)  # (D,)

    return traj_std, frame_std / 10


def generate_noisy_trajectory(clean_vec, traj_std, frame_std):
    """Generate synthetic noisy trajectories.

    Args:
        clean_vec: (BS, T, D) clean vector trajectories
        traj_std: (D,) trajectory-level noise std
        frame_std: (D,) per-frame noise std

    Returns:
        noisy_vec: (BS, T, D) noisy vector trajectories
    """
    if not isinstance(clean_vec, torch.Tensor):
        clean_vec = torch.from_numpy(clean_vec).float()
    BS, T, D = clean_vec.shape
    device = clean_vec.device
    # Generate trajectory-level noise (same for all frames in a trajectory)
    traj_noise = traj_std.to(device) * torch.randn(
        BS, 1, D, device=device
    )  # (BS, 1, D)

    # Generate per-frame noise
    frame_noise = frame_std.to(device) * torch.randn(
        BS, T, D, device=device
    )  # (BS, T, D)

    # Add noise
    noisy_vec = clean_vec + traj_noise + frame_noise  # (BS, T, 9)

    # smooth it by a gaussian filter size 5 along time axis
    # noisy_vec = gaussian_filter(noisy_vec.cpu().numpy(), sigma=1, axes=[1])
    # noisy_vec = torch.from_numpy(noisy_vec).to(device)
    
    return noisy_vec


def est_noise_use_merged_data(num=100):
    gt_dataset = load_pickle("data/HOT3D-CLIP/preprocess/dataset_contact.pkl")
    est_dataset = load_pickle(
        "data/HOT3D-CLIP/preprocess/dataset_contact_patched_hawor_gtcamFalse.pkl"
    )

    key_list = [
        "left_hand_theta",
        "left_hand_shape",
        "right_hand_theta",
        "right_hand_shape",
    ]

    std_dict = defaultdict(dict)
    gt_data = defaultdict(list)
    est_data = defaultdict(list)
    for seq in tqdm(np.random.permutation(list(est_dataset.keys()))[:num]):
        # select valid data
        if est_dataset[seq]["ready"]:
            for key in key_list:
                side, _, name = key.split("_")
                gt_d = gt_dataset[seq][f"{side}_hand"][name]
                est_d = est_dataset[seq][f"{side}_hand"][name]
                if np.isnan(est_d).any():
                    continue
                gt_data[key].append(gt_d)
                est_data[key].append(est_d)
    for key in key_list:
        gt_data[key] = np.stack(gt_data[key], axis=0)
        est_data[key] = np.stack(est_data[key], axis=0)
        gt_data[key] = torch.from_numpy(gt_data[key]).float()
        est_data[key] = torch.from_numpy(est_data[key]).float()

        # print(est_data[key][0:2,..., 3:7])

        traj_std, frame_std = compute_trajectory_noise_stats(
            gt_data[key], est_data[key]
        )
        std_dict[key] = {"traj_std": traj_std, "frame_std": frame_std}

    save_file = "data/cache/noise_stats_hand.pkl"
    pickle.dump(std_dict, open(save_file, "wb"))

    vis_gen(gt_data, est_data, std_dict, "outputs/est_noise_merged.rrd")




print("WHY TEHR IS NAN??? WHY some frame are black??? in w_motion")


def est_noise():
    est_dir = "data/HOT3D-CLIP/hawor_gtcamTrue/"
    # est_dir = 'data/HOT3D-CLIP/hawor_gtcamFalse/'
    gt_dir = "data/HOT3D-CLIP/preprocess/"
    np.random.seed(42)
    est_list = sorted(glob(osp.join(est_dir, "*.npz")))
    np.random.shuffle(est_list)

    std_dict = defaultdict(dict)
    gt_data = defaultdict(list)
    est_data = defaultdict(list)
    for est_file in tqdm(est_list[:100]):
        gt_file = osp.join(gt_dir, osp.basename(est_file))
        gt = np.load(gt_file)
        est = np.load(est_file)

        key_list = [
            "left_hand_theta",
            "left_hand_shape",
            "right_hand_theta",
            "right_hand_shape",
        ]

        for key in key_list:
            nan_mask = np.isnan(est[key])
            if nan_mask.any():
                mask_d = nan_mask.any(axis=-1)
                side = key.split("_")[0]
                continue
            gt_data[key].append(gt[key])
            est_data[key].append(est[key])

    for key in key_list:
        gt_data[key] = np.stack(gt_data[key], axis=0)
        est_data[key] = np.stack(est_data[key], axis=0)
        gt_data[key] = torch.from_numpy(gt_data[key]).float()
        est_data[key] = torch.from_numpy(est_data[key]).float()

        # print(est_data[key][0:2,..., 3:7])

        traj_std, frame_std = compute_trajectory_noise_stats(
            gt_data[key], est_data[key]
        )
        # print(key, traj_std.shape, frame_std.shape, )
        std_dict[key] = {"traj_std": traj_std, "frame_std": frame_std}

    save_file = "data/cache/noise_stats_hand.pkl"
    pickle.dump(std_dict, open(save_file, "wb"))


    vis_gen(gt_data, est_data, std_dict, "outputs/est_noise.rrd")


def vis_gen(gt_data, est_data, std_dict, save_file="outputs/est_noise.rrd"):
    T = 120
    time = 0

    rr.init("est_noise")
    rr.save(save_file)
    print("Saving to", save_file)
    for seq_idx in range(5):
        right_hand_theta = gt_data["right_hand_theta"][seq_idx][:T].to(device)
        right_hand_shape = gt_data["right_hand_shape"][seq_idx][:T].to(device)
        verts_clean, faces_clean, joints = (
            sided_mano_model.hand_para2verts_faces_joints(
                right_hand_theta, right_hand_shape, side="right"
            )
        )

        gen_theta = generate_noisy_trajectory(
            right_hand_theta[None],
            std_dict["right_hand_theta"]["traj_std"],
            std_dict["right_hand_theta"]["frame_std"],
        )[0]
        gen_shape = generate_noisy_trajectory(
            right_hand_shape[None],
            std_dict["right_hand_shape"]["traj_std"],
            std_dict["right_hand_shape"]["frame_std"],
        )[0]
        verts_gen, faces_gen, joints_gen = (
            sided_mano_model.hand_para2verts_faces_joints(
                gen_theta, gen_shape, side="right"
            )
        )

        right_theta_est = est_data["right_hand_theta"][seq_idx][:T].to(device)
        right_shape_est = est_data["right_hand_shape"][seq_idx][:T].to(device)
        verts_est, faces_est, joints_est = (
            sided_mano_model.hand_para2verts_faces_joints(
                right_theta_est, right_shape_est, side="right"
            )
        )

        V = verts_gen.shape[-2]
        for t in range(T):
            rr.set_time_sequence("frame", time)
            time += 1
            rr.log(
                "world/hand_clean",
                rr.Mesh3D(
                    vertex_positions=verts_clean[t].cpu().numpy(),
                    triangle_indices=faces_clean[t].cpu().numpy(),
                    vertex_colors=[[0, 200, 0] for _ in range(V)],
                ),
            )
            rr.log(
                "world/hand_gen",
                rr.Mesh3D(
                    vertex_positions=verts_gen[t].cpu().numpy(),
                    triangle_indices=faces_gen[t].cpu().numpy(),
                    vertex_colors=[[0, 0, 200] for _ in range(V)],
                ),
            )

            rr.log(
                "world/hand_est",
                rr.Mesh3D(
                    vertex_positions=verts_est[t].cpu().numpy(),
                    triangle_indices=faces_est[t].cpu().numpy(),
                    vertex_colors=[[200, 0, 0] for _ in range(V)],
                ),
            )

    return


if __name__ == "__main__":
    est_noise_use_merged_data()
    # est_noise()
