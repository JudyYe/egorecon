import pickle
import os
from collections import defaultdict
from tqdm import tqdm
import rerun as rr
import torch
import os.path as osp
from glob import glob
import numpy as np
gt_file = ""
est_file = ""


from egorecon.utils.motion_repr import HandWrapper

mano_model_folder = "assets/mano"
device = 'cuda:0'
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
    print('frame_residual', frame_residual.shape, )
    
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
    BS, T, D = clean_vec.shape
    device = clean_vec.device
    # Generate trajectory-level noise (same for all frames in a trajectory)
    traj_noise = traj_std.to(device) * torch.randn(BS, 1, D, device=device)  # (BS, 1, D)
    
    # Generate per-frame noise
    frame_noise = frame_std.to(device) * torch.randn(BS, T, D, device=device)  # (BS, T, D)
    
    # Add noise
    noisy_vec = clean_vec + traj_noise + frame_noise  # (BS, T, 9)
    
    return noisy_vec


def est_noise():
    # est_dir = 'data/HOT3D-CLIP/hawor_gtcamTrue/'
    est_dir = 'data/HOT3D-CLIP/hawor_gtcamFalse/'
    gt_dir = 'data/HOT3D-CLIP/preprocess/'
    np.random.seed(42)
    est_list = sorted(glob(osp.join(est_dir, '*.npz')))
    np.random.shuffle(est_list)

    std_dict = defaultdict(dict)
    gt_data = defaultdict(list)
    est_data = defaultdict(list)
    for est_file in tqdm(est_list[:100]):
        gt_file = osp.join(gt_dir, osp.basename(est_file))
        gt = np.load(gt_file)
        est = np.load(est_file)
        
        key_list = ['left_hand_theta', 'left_hand_shape', 'right_hand_theta', 'right_hand_shape']

        for key in key_list:
            nan_mask = np.isnan(est[key])
            if nan_mask.any():
                mask_d = nan_mask.any(axis=-1)  
                side = key.split('_')[0]
                print('any valid but nan', (est[f"{side}_hand_valid"][mask_d] != 1).any())
                continue
            gt_data[key].append(gt[key])
            est_data[key].append(est[key])

    for key in key_list:
        gt_data[key] = np.stack(gt_data[key], axis=0)
        est_data[key] = np.stack(est_data[key], axis=0)
        gt_data[key] = torch.from_numpy(gt_data[key]).float()
        est_data[key] = torch.from_numpy(est_data[key]).float()
        
        # print(est_data[key][0:2,..., 3:7])
        print('est', torch.isnan(est_data[key]).any())

        traj_std, frame_std = compute_trajectory_noise_stats(gt_data[key], est_data[key])
        # print(key, traj_std.shape, frame_std.shape, )
        std_dict[key] = {'traj_std': traj_std, 'frame_std': frame_std}
        print('nan std', key, torch.isnan(traj_std).any(), torch.isnan(frame_std).any(), torch.where(torch.isnan(traj_std)), torch.where(torch.isnan(frame_std)))

    save_file = 'data/cache/noise_stats_hand.pkl'
    pickle.dump(std_dict, open(save_file, 'wb'))
    T = 120
    time = 0

    rr.init("est_noise")
    rr.save('outputs/est_noise.rrd')
    for seq_idx in range(5):
        right_hand_theta = gt_data['right_hand_theta'][seq_idx][:T].to(device)
        right_hand_shape = gt_data['right_hand_shape'][seq_idx][:T].to(device)
        verts_clean, faces_clean, joints = sided_mano_model.hand_para2verts_faces_joints(right_hand_theta, right_hand_shape, side="right")
        
        gen_theta = generate_noisy_trajectory(right_hand_theta[None], std_dict['right_hand_theta']['traj_std'], std_dict['right_hand_theta']['frame_std'])[0]
        gen_shape = generate_noisy_trajectory(right_hand_shape[None], std_dict['right_hand_shape']['traj_std'], std_dict['right_hand_shape']['frame_std'])[0]
        verts_gen, faces_gen, joints_gen = sided_mano_model.hand_para2verts_faces_joints(gen_theta, gen_shape, side="right")

        right_theta_est = est_data['right_hand_theta'][seq_idx][:T].to(device)
        right_shape_est = est_data['right_hand_shape'][seq_idx][:T].to(device)
        verts_est, faces_est, joints_est = sided_mano_model.hand_para2verts_faces_joints(right_theta_est, right_shape_est, side="right")
        print(torch.isnan(verts_est).any(), torch.isnan(verts_gen).any(), torch.isnan(verts_clean).any())
        print(torch.where(torch.isnan(verts_est)))
        print(torch.where(torch.isnan(verts_gen)))
        # print(torch.where(torch.isnan(verts_clean)))

        V = verts_clean.shape[-2]
        for t in range(T):
            rr.set_time_sequence("frame", time)
            time += 1
            rr.log("world/hand_clean", rr.Mesh3D(
                vertex_positions=verts_clean[t].cpu().numpy(),
                triangle_indices=faces_clean[t].cpu().numpy(),
                vertex_colors=[[0, 200, 0] * V],

            ))
            rr.log("world/hand_gen", rr.Mesh3D(
                vertex_positions=verts_gen[t].cpu().numpy(),
                triangle_indices=faces_gen[t].cpu().numpy(),
                vertex_colors=[[0, 0, 200] * V],
            ))

            rr.log("world/hand_est", rr.Mesh3D(
                vertex_positions=verts_est[t].cpu().numpy(),
                triangle_indices=faces_est[t].cpu().numpy(),
                vertex_colors=[[200, 0, 0] * V],
            ))
        

    return 





if __name__ == "__main__":
    est_noise()