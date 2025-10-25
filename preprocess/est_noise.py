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
from scipy.spatial.transform import Rotation as R
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


def print_validity_statistics(valid_data):
    """Print comprehensive validity statistics including masking ratios and segment lengths.
    
    Args:
        valid_data: dict containing validity data arrays
    """
    print("\n" + "="*60)
    print("FRAME VALIDITY STATISTICS")
    print("="*60)
    
    # Calculate masking ratios
    left_valid_ratio = np.mean(valid_data["left_valid"]) if valid_data["left_valid"] else 0
    right_valid_ratio = np.mean(valid_data["right_valid"]) if valid_data["right_valid"] else 0
    
    print("Masking Ratios:")
    print(f"  Left hand valid frames:  {left_valid_ratio:.3f} ({left_valid_ratio*100:.1f}%)")
    print(f"  Right hand valid frames: {right_valid_ratio:.3f} ({right_valid_ratio*100:.1f}%)")
    
    # Left hand consecutive invalid segment statistics
    if valid_data.get("left_consec_valid"):
        left_consec_stats = {
            "mean": np.mean(valid_data["left_consec_valid"]),
            "std": np.std(valid_data["left_consec_valid"]),
            "max": np.max(valid_data["left_consec_valid"]),
            "min": np.min(valid_data["left_consec_valid"]),
            "count": len(valid_data["left_consec_valid"])
        }
        print("\nLeft Hand Consecutive Invalid Segment Statistics:")
        print(f"  Average length: {left_consec_stats['mean']:.2f} frames")
        print(f"  Min length:     {left_consec_stats['min']} frames")
        print(f"  Max length:     {left_consec_stats['max']} frames")
        print(f"  Std deviation:  {left_consec_stats['std']:.2f} frames")
        print(f"  Total segments: {left_consec_stats['count']}")
    else:
        print("\nLeft Hand Consecutive Invalid Segments: None found")
    
    # Right hand consecutive invalid segment statistics
    if valid_data.get("right_consec_valid"):
        right_consec_stats = {
            "mean": np.mean(valid_data["right_consec_valid"]),
            "std": np.std(valid_data["right_consec_valid"]),
            "max": np.max(valid_data["right_consec_valid"]),
            "min": np.min(valid_data["right_consec_valid"]),
            "count": len(valid_data["right_consec_valid"])
        }
        print("\nRight Hand Consecutive Invalid Segment Statistics:")
        print(f"  Average length: {right_consec_stats['mean']:.2f} frames")
        print(f"  Min length:     {right_consec_stats['min']} frames")
        print(f"  Max length:     {right_consec_stats['max']} frames")
        print(f"  Std deviation:  {right_consec_stats['std']:.2f} frames")
        print(f"  Total segments: {right_consec_stats['count']}")
    else:
        print("\nRight Hand Consecutive Invalid Segments: None found")
    
    # Left hand all invalid frame statistics
    if valid_data.get("left_all_invalid"):
        left_all_invalid_stats = {
            "mean": np.mean(valid_data["left_all_invalid"]),
            "std": np.std(valid_data["left_all_invalid"]),
            "max": np.max(valid_data["left_all_invalid"]),
            "min": np.min(valid_data["left_all_invalid"]),
            "total": np.sum(valid_data["left_all_invalid"])
        }
        print("\nLeft Hand All Invalid Frame Statistics:")
        print(f"  Average per sequence: {left_all_invalid_stats['mean']:.2f} frames")
        print(f"  Min per sequence:     {left_all_invalid_stats['min']} frames")
        print(f"  Max per sequence:     {left_all_invalid_stats['max']} frames")
        print(f"  Std deviation:        {left_all_invalid_stats['std']:.2f} frames")
        print(f"  Total invalid frames: {left_all_invalid_stats['total']}")
    else:
        print("\nLeft Hand All Invalid Frames: None found")
    
    # Right hand all invalid frame statistics
    if valid_data.get("right_all_invalid"):
        right_all_invalid_stats = {
            "mean": np.mean(valid_data["right_all_invalid"]),
            "std": np.std(valid_data["right_all_invalid"]),
            "max": np.max(valid_data["right_all_invalid"]),
            "min": np.min(valid_data["right_all_invalid"]),
            "total": np.sum(valid_data["right_all_invalid"])
        }
        print("\nRight Hand All Invalid Frame Statistics:")
        print(f"  Average per sequence: {right_all_invalid_stats['mean']:.2f} frames")
        print(f"  Min per sequence:     {right_all_invalid_stats['min']} frames")
        print(f"  Max per sequence:     {right_all_invalid_stats['max']} frames")
        print(f"  Std deviation:        {right_all_invalid_stats['std']:.2f} frames")
        print(f"  Total invalid frames: {right_all_invalid_stats['total']}")
    else:
        print("\nRight Hand All Invalid Frames: None found")
    
    print("="*60)
    # ============================================================
    # FRAME VALIDITY STATISTICS
    # ============================================================
    # Masking Ratios:
    #   Left hand valid frames:  0.930 (93.0%)
    #   Right hand valid frames: 0.941 (94.1%)

    # Left Hand Consecutive Invalid Segment Statistics:
    #   Average length: 22.47 frames
    #   Min length:     1 frames
    #   Max length:     150 frames
    #   Std deviation:  41.37 frames
    #   Total segments: 47

    # Right Hand Consecutive Invalid Segment Statistics:
    #   Average length: 17.60 frames
    #   Min length:     1 frames
    #   Max length:     150 frames
    #   Std deviation:  35.12 frames
    #   Total segments: 50

    # Left Hand All Invalid Frame Statistics:
    #   Average per sequence: 10.56 frames
    #   Min per sequence:     0 frames
    #   Max per sequence:     150 frames
    #   Std deviation:        35.62 frames
    #   Total invalid frames: 1056

    # Right Hand All Invalid Frame Statistics:
    #   Average per sequence: 8.80 frames
    #   Min per sequence:     0 frames
    #   Max per sequence:     150 frames
    #   Std deviation:        31.93 frames
    #   Total invalid frames: 880
    # ============================================================
    # Saving to outputs/est_noise_merged.rrd    


def analyze_frame_validity(left_valid, right_valid):
    """Analyze frame validity patterns for both hands separately.
    
    Args:
        left_valid: (T,) boolean array indicating left hand validity per frame
        right_valid: (T,) boolean array indicating right hand validity per frame
    
    Returns:
        dict: Contains 'left_consec_invalid_lengths', 'right_consec_invalid_lengths', 
              'left_total_invalid_count', 'right_total_invalid_count'
    """
    def find_consecutive_invalid_segments(valid_array):
        """Find consecutive invalid segments in a validity array."""
        invalid_segments = []
        in_invalid_segment = False
        current_length = 0
        
        for frame_valid in valid_array:
            if not frame_valid:  # Invalid frame
                if not in_invalid_segment:
                    in_invalid_segment = True
                    current_length = 1
                else:
                    current_length += 1
            else:  # Valid frame
                if in_invalid_segment:
                    invalid_segments.append(current_length)
                    in_invalid_segment = False
                    current_length = 0
        
        # Handle case where sequence ends with invalid frames
        if in_invalid_segment:
            invalid_segments.append(current_length)
        
        return invalid_segments
    
    # Analyze left hand
    left_consec_invalid = find_consecutive_invalid_segments(left_valid)
    left_total_invalid = np.sum(~left_valid)
    
    # Analyze right hand
    right_consec_invalid = find_consecutive_invalid_segments(right_valid)
    right_total_invalid = np.sum(~right_valid)
    
    return {
        'left_consec_invalid_lengths': left_consec_invalid,
        'right_consec_invalid_lengths': right_consec_invalid,
        'left_total_invalid_count': left_total_invalid,
        'right_total_invalid_count': right_total_invalid
    }



def compute_noise_param_dict(clean_vec, noisy_vec):
    """Compute noise parameter dictionary for hand trajectories.
    
    Args:
        clean_vec: (N, T, 31) clean hand parameter vector [global_orient(3) + transl(3) + hand_pose(15) + betas(10)]
        noisy_vec: (N, T, 31) noisy hand parameter vector
    
    Returns:
        noise_std_params_dict: dict containing traj_std and frame_std for each hand parameter type
            Both traj_std and frame_std have shape (D,) for each parameter
    """
    # Convert vectors to parameter dictionaries
    clean_param_dict = HandWrapper.para2dict(clean_vec)
    noisy_param_dict = HandWrapper.para2dict(noisy_vec)
    
    # Initialize noise std parameters dictionary
    noise_std_params_dict = {}
    
    # For translation, hand_pose (PCAs), and betas: compute std in parameter space
    # This matches how noise is applied in generate_noisy_hand_traj
    transl_diff = noisy_param_dict['transl'] - clean_param_dict['transl']  # (N, T, 3)
    hand_pose_diff = noisy_param_dict['hand_pose'] - clean_param_dict['hand_pose']  # (N, T, 15)
    betas_diff = noisy_param_dict['betas'] - clean_param_dict['betas']  # (N, T, 10)
    
    # Compute traj_std and frame_std: both should be shape (D,)
    # traj_std: std across batch for mean across time
    # frame_std: std across all (batch, time) for each dimension
    noise_std_params_dict['transl'] = {
        'traj_std': transl_diff.mean(axis=1).std(axis=0),  # (3,) - trajectory-level std
        'frame_std': transl_diff.reshape(-1, transl_diff.shape[-1]).std(axis=0)  # (3,) - per-frame std
    }
    noise_std_params_dict['hand_pose'] = {
        'traj_std': hand_pose_diff.mean(axis=1).std(axis=0),  # (15,) - trajectory-level std
        'frame_std': hand_pose_diff.reshape(-1, hand_pose_diff.shape[-1]).std(axis=0)  # (15,) - per-frame std
    }
    noise_std_params_dict['betas'] = {
        'traj_std': betas_diff.mean(axis=1).std(axis=0),  # (10,) - trajectory-level std
        'frame_std': betas_diff.reshape(-1, betas_diff.shape[-1]).std(axis=0)  # (10,) - per-frame std
    }
    
    # For global_orient: convert rotation vectors to euler angles and compute std in euler space
    # This matches the space where noise is applied in generate_noisy_hand_traj
    N, T = clean_param_dict['global_orient'].shape[:2]
    clean_global_orient_flat = clean_param_dict['global_orient'].reshape(-1, 3)
    noisy_global_orient_flat = noisy_param_dict['global_orient'].reshape(-1, 3)
    
    clean_global_orient_mat = R.from_rotvec(clean_global_orient_flat)
    clean_global_orient_angle = clean_global_orient_mat.as_euler('zxy', degrees=True).reshape(N, T, 3)
    
    noisy_global_orient_mat = R.from_rotvec(noisy_global_orient_flat)
    noisy_global_orient_angle = noisy_global_orient_mat.as_euler('zxy', degrees=True).reshape(N, T, 3)
    
    global_orient_diff = noisy_global_orient_angle - clean_global_orient_angle  # (N, T, 3)
    
    # Compute std in euler angle space (same as noise application space)
    noise_std_params_dict['global_orient'] = {
        'traj_std': global_orient_diff.mean(axis=1).std(axis=0),  # (3,) - trajectory-level std
        'frame_std': global_orient_diff.reshape(-1, global_orient_diff.shape[-1]).std(axis=0)  # (3,) - per-frame std
    }
    
    return noise_std_params_dict

def generate_noisy_hand_traj(clean_vec, hand_noise_dict, traj_std):
    """Generate noisy hand trajectories similar to generate_noisy_smplx.
    
    Args:
        clean_vec: (T, 31) clean hand parameter vector [global_orient(3) + transl(3) + hand_pose(15) + betas(10)]
        hand_noise_dict: dict to store generated noise for each parameter
        traj_std: dict containing noise std for each hand parameter type
    
    Returns:
        noisy_vec: (T, 31) noisy hand parameter vector
    """
    # Define noise std parameters for hand parameters
    noise_std_params_dict = {
        'global_orient': traj_std['global_orient'],
        'transl': traj_std['transl'], 
        'hand_pose': traj_std['hand_pose'],
        'betas': traj_std['betas']
    }
    
    # Convert vector to parameter dictionary
    param_dict = HandWrapper.para2dict(clean_vec)
    
    # Initialize noisy parameters dictionary
    cano_hand_params_dict_noisy = {}
    
    # Process each parameter type
    for param_name in ['transl', 'hand_pose', 'betas', 'global_orient']:
        if param_name == 'transl' or param_name == 'hand_pose' or param_name == 'betas':
            # For translation, hand pose, and betas, add Gaussian noise directly
            # hand_pose is treated as PCAs ... so just directly add
            noise = np.random.normal(
                loc=0.0, 
                scale=noise_std_params_dict[param_name], 
                size=param_dict[param_name].shape
            )
            cano_hand_params_dict_noisy[param_name] = param_dict[param_name] + noise
            
            # Store noise in dictionary
            if param_name not in hand_noise_dict.keys():
                hand_noise_dict[param_name] = []
            hand_noise_dict[param_name].append(noise)
            
        elif param_name == 'global_orient':
            # For global orientation, work in euler angles
            global_orient_mat = R.from_rotvec(param_dict['global_orient'])
            global_orient_angle = global_orient_mat.as_euler('zxy', degrees=True)
            
            # Add noise in euler angle space
            noise_global_rot = np.random.normal(
                loc=0.0, 
                scale=noise_std_params_dict[param_name], 
                size=global_orient_angle.shape
            )
            global_orient_angle_noisy = global_orient_angle + noise_global_rot
            
            # Convert back to rotation vector
            cano_hand_params_dict_noisy[param_name] = R.from_euler(
                'zxy', global_orient_angle_noisy, degrees=True
            ).as_rotvec()
            
            # Store noise in dictionary
            if param_name not in hand_noise_dict.keys():
                hand_noise_dict[param_name] = []
            hand_noise_dict[param_name].append(noise_global_rot)  # Store in euler angle
    
    # Convert back to vector format
    noisy_vec = HandWrapper.dict2para(cano_hand_params_dict_noisy, merge=True)
    
    return noisy_vec


def generate_noisy_smplx(cano_smplx_params_dict, smplx_noise_dict, traj_std):
    noise_std_params_dict = {'global_orient': noise_std_smplx_global_rot,
                                      'transl': noise_std_smplx_trans,
                                      'body_pose': noise_std_smplx_body_rot,
                                      'betas': noise_std_smplx_betas,}
    self.load_noise = False
    if self.input_noise and (not self.sep_noise):
        cano_smplx_params_dict_noisy = {}
        for param_name in ['transl', 'body_pose', 'betas', 'global_orient']:
            if param_name == 'transl' or param_name == 'betas':
                if self.load_noise:
                    noise_1 = self.loaded_smplx_noise_dict[param_name][i*self.spacing]
                else:
                    noise_1 = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=cano_smplx_params_dict[param_name].shape)
                cano_smplx_params_dict_noisy[param_name] = cano_smplx_params_dict[param_name] + noise_1
                if param_name not in smplx_noise_dict.keys():
                    smplx_noise_dict[param_name] = []
                smplx_noise_dict[param_name].append(noise_1)
            elif param_name == 'global_orient':
                global_orient_mat = R.from_rotvec(cano_smplx_params_dict['global_orient'])  # [145, 3, 3]
                global_orient_angle = global_orient_mat.as_euler('zxy', degrees=True)
                if self.load_noise:
                    noise_global_rot = self.loaded_smplx_noise_dict[param_name][i*self.spacing]
                else:
                    noise_global_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=global_orient_angle.shape)
                global_orient_angle_noisy = global_orient_angle + noise_global_rot
                cano_smplx_params_dict_noisy[param_name] = R.from_euler('zxy', global_orient_angle_noisy, degrees=True).as_rotvec()
                if param_name not in smplx_noise_dict.keys():
                    smplx_noise_dict[param_name] = []
                smplx_noise_dict[param_name].append(noise_global_rot)  #  [145, 3] in euler angle
            elif param_name == 'body_pose':
                body_pose_mat = R.from_rotvec(cano_smplx_params_dict['body_pose'].reshape(-1, 3))
                body_pose_angle = body_pose_mat.as_euler('zxy', degrees=True)  # [145*21, 3]
                if self.load_noise:
                    noise_body_pose_rot = self.loaded_smplx_noise_dict[param_name][i*self.spacing].reshape(-1, 3)
                else:
                    noise_body_pose_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=body_pose_angle.shape)
                body_pose_angle_noisy = body_pose_angle + noise_body_pose_rot
                cano_smplx_params_dict_noisy[param_name] = R.from_euler('zxy', body_pose_angle_noisy, degrees=True).as_rotvec().reshape(-1, 21, 3)
                if param_name not in smplx_noise_dict.keys():
                    smplx_noise_dict[param_name] = []
                smplx_noise_dict[param_name].append(noise_body_pose_rot.reshape(-1, 21, 3))  # [145, 21, 3]  in euler angle

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
        "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact_patched_hawor_gtcamFalse.pkl"
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
    valid_data = defaultdict(list)

    for seq in tqdm(np.random.permutation(list(est_dataset.keys()))[:num]):
        # select valid data
        if est_dataset[seq]["ready"]:
            left_valid = est_dataset[seq]["left_hand"]["valid"]
            right_valid = est_dataset[seq]["right_hand"]["valid"]
            
            valid_data["left_valid"].append(left_valid)
            valid_data["right_valid"].append(right_valid)
            
            # Analyze frame validity patterns
            validity_analysis = analyze_frame_validity(left_valid, right_valid)
            
            # Store left hand consecutive invalid segments
            if validity_analysis['left_consec_invalid_lengths']:
                valid_data["left_consec_valid"].extend(validity_analysis['left_consec_invalid_lengths'])
            
            # Store right hand consecutive invalid segments
            if validity_analysis['right_consec_invalid_lengths']:
                valid_data["right_consec_valid"].extend(validity_analysis['right_consec_invalid_lengths'])
            
            # Store total invalid counts for each hand
            valid_data["left_all_invalid"].append(validity_analysis['left_total_invalid_count'])
            valid_data["right_all_invalid"].append(validity_analysis['right_total_invalid_count'])
            
            for key in key_list:
                side, _, name = key.split("_")
                gt_d = gt_dataset[seq][f"{side}_hand"][name]
                est_d = est_dataset[seq][f"{side}_hand"][name]
                if np.isnan(est_d).any():
                    idx = np.where(np.isnan(est_d))
                    print(idx, est_d[idx[0]])
                    print('valid', est_dataset[seq][f"{side}_hand"]["valid"][idx[0]])
                    print("NaN????", name, seq, side)
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
    
    # Print comprehensive validity statistics
    print_validity_statistics(valid_data)
    

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
