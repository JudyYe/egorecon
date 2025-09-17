#!/usr/bin/env python3

import os
import os.path as osp
from copy import deepcopy

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from ...utils.motion_repr import cano_seq_mano
from ...utils.rotation_utils import (matrix_to_rotation_6d_numpy,
                                     quaternion_to_matrix_numpy,
                                     rotation_6d_to_matrix_numpy)
from ..lafan1.utils import rotate_at_frame_w_obj


def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    # with open(path, "rb") as f:
    #     return pickle.load(f)
    data = np.load(path, allow_pickle=True)
    data = dict(data)
    uid_list = data["objects"]
    data.pop("objects")
    processed_data = {}
    objects = {}
    for uid in uid_list:
        uid = str(uid)
        objects[uid] = {}
        for k in data.keys():
            if k.startswith(f"obj_{uid}"):
                new_key = k.replace(f"obj_{uid}_", "")
                objects[uid][new_key] = data[k]
        # get wTo_shelf
        cTo_shelf = objects[uid]["cTo_shelf"]
        wTc = data["wTc"]
        # cTw = np.linalg.inv(wTc)
        wTo_shelf = wTc @ cTo_shelf
        objects[uid]["wTo_shelf"] = wTo_shelf

    processed_data["left_hand"] = {}
    processed_data["right_hand"] = {}
    for k in data:
        if not k.startswith("obj_"):
            if k.startswith("left_hand"):
                processed_data["left_hand"][k.replace("left_hand_", "")] = data[k]
            elif k.startswith("right_hand"):
                processed_data["right_hand"][k.replace("right_hand_", "")] = data[k]
            else:
                processed_data[k] = data[k]

    processed_data["objects"] = objects
    seq_index = osp.basename(path).split(".")[0]

    return {seq_index: processed_data}


# in rohm:
# init(): create windows -> canonical --> add noise --> get rep


def to_tensor(array, dtype=torch.float32):
    """Convert array to tensor with specified dtype."""
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


class HandToObjectDataset(Dataset):
    """
    Dataset for hand-to-object trajectory denoising.

    Loads HOT3D processed data and creates windows for training.
    Inputs: Left hand trajectory + Right hand trajectory (both T x 9 or T x 12, default: 9D)
    Output: Object trajectory (T x 9 or T x 12, default: 9D)
    """

    def __init__(
        self,
        data_path,
        is_train=False,
        window_size=120,
        single_demo=None,  # For overfitting: specify demo_id
        single_object=None,  # For overfitting: specify object_id
        motion_threshold=0.005,  # Threshold for considering motion vs stationary
        sampling_strategy="balanced",  # 'balanced', 'motion_only', 'random'
        min_motion_frames=10,  # Minimum consecutive motion frames to consider
        augment=False,  # Whether in training mode (for data augmentation)
        split="train",  # 'train', 'val', or 'all'
        val_split_ratio=0.2,  # Fraction of windows to use for validation
        split_seed=42,  # Seed for reproducible splits
        noise_std_obj_rot=0.0,
        noise_std_obj_trans=0.0,
        noise_std_mano_global_rot=0.0,
        noise_std_mano_body_rot=0.0,
        noise_std_mano_trans=0.0,
        noise_std_mano_betas=0.0,
        traj_std_obj_rot=0.0,
        traj_std_obj_trans=0.0,
        use_constant_noise=False,
        noise_scheme="syn",  # 'syn', 'real'
        opt=None,
    ):
        self.is_train = is_train
        self.data_path = data_path
        self.window_size = window_size
        self.motion_threshold = motion_threshold
        self.sampling_strategy = sampling_strategy
        self.min_motion_frames = min_motion_frames
        self.augment = augment  # Add training mode flag
        self.split = split
        self.val_split_ratio = val_split_ratio
        self.split_seed = split_seed
        self.opt = opt

        self.mano_model_path =  opt.paths.mano_dir
        self.sided_mano_models = {
            "left": smplx.create(
                os.path.join(self.mano_model_path, "MANO_LEFT.pkl"),
                "mano",
                is_rhand=False,
                num_pca_comps=15,
            ),
            "right": smplx.create(
                os.path.join(self.mano_model_path, "MANO_RIGHT.pkl"),
                "mano",
                is_rhand=True,
                num_pca_comps=15,
            ),
        }

        # Load processed data
        print(f"Loading data from {data_path}...")
        self.processed_data = load_pickle(data_path)
        print(f"Found {len(self.processed_data)} demonstrations")

        self.pose_dim = 9

        # Filter for single demo/object if specified (for overfitting)
        if single_demo or single_object:
            self.processed_data = self._filter_data(
                single_demo, single_object.split("+")
            )
            print(
                f"Filtered to demo: {single_demo} object: {single_object}, total: {len(self.processed_data)} demonstrations for overfitting"
            )

        # Create windows
        self.windows = self._create_windows()
        print(f"Created {len(self.windows)} windows")

        # Apply train/validation split
        self.windows = self._apply_train_val_split()
        print(f"Created {len(self.windows)} {self.split} windows")

        # Compute normalization statistics
        self._compute_normalization_stats()

        # Setup full trajectory data for the current split
        self._setup_full_trajectory_data_from_windows()

        self.noise_std_params_dict = {
            "global_orient": noise_std_mano_global_rot,
            "transl": noise_std_mano_trans,
            "hand_pose": noise_std_mano_body_rot,
            "betas": noise_std_mano_betas,
            "object_rot": noise_std_obj_rot,
            "object_trans": noise_std_obj_trans,
            'traj_rot': traj_std_obj_rot,
            'traj_trans': traj_std_obj_trans,
        }
        self.use_constant_noise = use_constant_noise
        self.noise_scheme = noise_scheme

    def _filter_data(self, single_demo=None, single_object=None):
        """Filter data for overfitting on specific demo/object."""
        filtered_data = {}
        if self.split == "mini":
            # seq = 'P0001_624f2ba9'
            seq = list(self.processed_data.keys())[0]
            filtered_data[seq] = {}
            # '194930206998778',
            if single_object:
                # 225397651484143 : bowl
                obj_list = single_object
            else:
                obj_list = list(self.processed_data[seq]["objects"].keys())

            if self.opt.datasets.one_window:
                t0 = self.opt.datasets.t0
                t1 = t0 + 120
            else:
                t0 = 0
                t1 = len(self.processed_data[seq]["wTc"])
            for key, value in self.processed_data[seq].items():
                if key == "objects":
                    continue
                filtered_data[seq][key] = value
            filtered_data[seq]["objects"] = {}
            for obj_id in obj_list:
                filtered_data[seq]["objects"][obj_id] = self.processed_data[seq][
                    "objects"
                ][obj_id]

            for key, value in filtered_data[seq].items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                filtered_data[seq][key][k][kk] = vv[t0:t1]
                        else:
                            filtered_data[seq][key][k] = v[t0:t1]
                elif isinstance(value, np.ndarray):
                    filtered_data[seq][key] = value[t0:t1]
                else:
                    print(key, type(value))
                    filtered_data[seq][key] = value

            return filtered_data

        for demo_id, demo_data in self.processed_data.items():
            if single_demo and demo_id != single_demo:
                continue

            if single_object and "objects" in demo_data:
                # Keep only the specified object
                if single_object in demo_data["objects"]:
                    filtered_demo = {
                        "left_hand": demo_data.get("left_hand"),
                        "right_hand": demo_data.get("right_hand"),
                        "objects": {single_object: demo_data["objects"][single_object]},
                    }
                    filtered_data[demo_id] = filtered_demo
            else:
                filtered_data[demo_id] = demo_data

        return filtered_data

    def _create_windows(self):
        """Create sliding windows from trajectory data."""
        windows = []

        for demo_id, demo_data in self.processed_data.items():
            for obj_id, obj_data in demo_data["objects"].items():
                print(obj_id, "current window obj_id:")
                for win in windows:
                    print(win["object_id"])

                object_traj = obj_data["wTo"]  # (T, 4, 4)
                camera_traj = demo_data["wTc"]  # (T, 4, 4)

                # Find the overlapping time range for all trajectories
                left_hand = demo_data["left_hand"]["theta"]
                right_hand = demo_data["right_hand"]["theta"]
                min_len = min(len(left_hand), len(right_hand), len(object_traj))

                if min_len < self.window_size:
                    print(
                        f"Warning: Trajectory too short for demo {demo_id}, obj {obj_id}: {min_len}"
                    )
                    continue

                # Trim all trajectories to same length
                left_hand_trimmed = left_hand[:min_len]
                right_hand_trimmed = right_hand[:min_len]
                object_traj_trimmed = object_traj[:min_len]
                camera_traj_trimmed = camera_traj[:min_len]

                # Create sliding windows
                for start_idx in range(
                    0, min_len - self.window_size + 1, self.window_size // 2
                ):
                    end_idx = start_idx + self.window_size

                    left_wrist_aa = left_hand_trimmed[start_idx:end_idx][:, 0:3]
                    left_wrist_pos = left_hand_trimmed[start_idx:end_idx][:, 3:6]

                    left_wrist_r = R.from_rotvec(left_wrist_aa)
                    left_wrist_quat = left_wrist_r.as_quat()[
                        :, [3, 0, 1, 2]
                    ]  # Returns [x, y, z, w] -> wxyz

                    right_wrist_aa = right_hand_trimmed[start_idx:end_idx][:, 0:3]
                    right_wrist_pos = right_hand_trimmed[start_idx:end_idx][:, 3:6]

                    right_wrist_r = R.from_rotvec(right_wrist_aa)
                    right_wrist_quat = right_wrist_r.as_quat()[
                        :, [3, 0, 1, 2]
                    ]  # Returns [x, y, z, w] -> wxyz

                    object_pos = object_traj_trimmed[start_idx:end_idx][:, :3, 3]
                    object_rot_mat = object_traj_trimmed[start_idx:end_idx][:, :3, :3]
                    object_r = R.from_matrix(object_rot_mat)
                    object_quat = object_r.as_quat()[
                        :, [3, 0, 1, 2]
                    ]  # Returns [x, y, z, w] -> wxyz

                    camera_pos = camera_traj_trimmed[start_idx:end_idx][:, :3, 3]
                    camera_rot_mat = camera_traj_trimmed[start_idx:end_idx][:, :3, :3]
                    camera_r = R.from_matrix(camera_rot_mat)
                    camera_quat = camera_r.as_quat()[
                        :, [3, 0, 1, 2]
                    ]  # Returns [x, y, z, w] -> wxyz

                    # ######### 1. Canonicalize the data using the camera frame  ##########
                    # put them to 1st camera frame
                    (
                        cano_camera_pos,
                        cano_camera_quat,
                        cano_object_pos,
                        cano_object_quat,
                        canoTw_rot,
                    ) = rotate_at_frame_w_obj(
                        camera_pos[np.newaxis, :, np.newaxis, :],
                        camera_quat[np.newaxis, :, np.newaxis, :],
                        object_pos[np.newaxis],
                        object_quat[np.newaxis],
                    )

                    object_shelf = demo_data["objects"][obj_id]["wTo_shelf"][
                        start_idx:end_idx
                    ]
                    object_shelf_pos = object_shelf[:, :3, 3]
                    object_shelf_r = R.from_matrix(object_shelf[:, :3, :3])
                    object_shelf_quat = object_shelf_r.as_quat()[:, [3, 0, 1, 2]]

                    _, _, cano_object_shelf_pos, cano_object_shelf_quat, _ = (
                        rotate_at_frame_w_obj(
                            camera_pos[np.newaxis, :, np.newaxis, :],
                            camera_quat[np.newaxis, :, np.newaxis, :],
                            object_shelf_pos[np.newaxis],
                            object_shelf_quat[np.newaxis],
                        )
                    )

                    _, _, cano_left_wrist_pos, cano_left_wrist_quat, _ = (
                        rotate_at_frame_w_obj(
                            camera_pos[np.newaxis, :, np.newaxis, :],
                            camera_quat[np.newaxis, :, np.newaxis, :],
                            left_wrist_pos[np.newaxis],
                            left_wrist_quat[np.newaxis],
                        )
                    )

                    _, _, cano_right_wrist_pos, cano_right_wrist_quat, _ = (
                        rotate_at_frame_w_obj(
                            camera_pos[np.newaxis, :, np.newaxis, :],
                            camera_quat[np.newaxis, :, np.newaxis, :],
                            right_wrist_pos[np.newaxis],
                            right_wrist_quat[np.newaxis],
                        )
                    )

                    cano_left_wrist_pos = cano_left_wrist_pos[0]  # T X 3
                    cano_left_wrist_quat = cano_left_wrist_quat[0]  # T X 4

                    cano_object_pos = cano_object_pos[0]  # T X 3
                    cano_object_quat = cano_object_quat[0]  # T X 4

                    cano_right_wrist_pos = cano_right_wrist_pos[0]  # T X 3
                    cano_right_wrist_quat = cano_right_wrist_quat[0]  # T X 4

                    cano_object_shelf_pos = cano_object_shelf_pos[0]  # T X 3
                    cano_object_shelf_quat = cano_object_shelf_quat[0]  # T X 4

                    cano_camera_pos = cano_camera_pos[0, :, 0, :]  # T X 3
                    cano_camera_quat = cano_camera_quat[0, :, 0, :]  # T X 4

                    # Translate everything such that the left_wrist initial position is at the origin.
                    # left_wrist2origin_trans = cano_left_wrist_pos[0:1, :].copy()
                    # cano_left_wrist_pos -= left_wrist2origin_trans
                    # cano_right_wrist_pos -= left_wrist2origin_trans
                    # cano_object_pos -= left_wrist2origin_trans

                    # Translate everything such that the camear initial position is at the origin.
                    camera2origin_trans = cano_camera_pos[0:1, :].copy()
                    canoTw = np.eye(4)
                    canoTw[:3, :3] = canoTw_rot
                    canoTw[:3, 3] = -camera2origin_trans

                    cano_camera_pos -= camera2origin_trans
                    cano_left_wrist_pos -= camera2origin_trans
                    cano_right_wrist_pos -= camera2origin_trans
                    cano_object_pos -= camera2origin_trans

                    print(left_hand_trimmed[start_idx:end_idx].shape)
                    mano_params_dict = {
                        "global_orient": left_wrist_aa,
                        "transl": left_wrist_pos,
                        "betas": demo_data["left_hand"]["shape"][start_idx:end_idx],
                        "hand_pose": left_hand_trimmed[start_idx:end_idx][:, 6:],
                    }

                    cano_left_positions, cano_left_mano_params_dict = cano_seq_mano(
                        canoTw=canoTw,
                        positions=None,
                        mano_params_dict=mano_params_dict,
                        mano_model=self.sided_mano_models["left"],
                        device="cpu",
                    )

                    mano_params_dict = {
                        "global_orient": right_wrist_aa,
                        "transl": right_wrist_pos,
                        "betas": demo_data["right_hand"]["shape"][start_idx:end_idx],
                        "hand_pose": right_hand_trimmed[start_idx:end_idx][:, 6:],
                    }

                    cano_right_positions, cano_right_mano_params_dict = cano_seq_mano(
                        canoTw=canoTw,
                        positions=None,
                        mano_params_dict=mano_params_dict,
                        mano_model=self.sided_mano_models["right"],
                        device="cpu",
                    )

                    # Prepare 9D repreesntation for left/right hand, and object
                    cano_left_wrist_rot_mat = quaternion_to_matrix_numpy(
                        cano_left_wrist_quat
                    )
                    cano_left_wrist_rot6d = matrix_to_rotation_6d_numpy(
                        cano_left_wrist_rot_mat
                    )

                    cano_right_wrist_rot_mat = quaternion_to_matrix_numpy(
                        cano_right_wrist_quat
                    )
                    cano_right_wrist_rot6d = matrix_to_rotation_6d_numpy(
                        cano_right_wrist_rot_mat
                    )

                    cano_object_rot_mat = quaternion_to_matrix_numpy(cano_object_quat)
                    cano_object_rot6d = matrix_to_rotation_6d_numpy(cano_object_rot_mat)

                    cano_camera_rot_mat = quaternion_to_matrix_numpy(cano_camera_quat)
                    cano_camera_rot6d = matrix_to_rotation_6d_numpy(cano_camera_rot_mat)

                    cano_object_shelf_rot_mat = quaternion_to_matrix_numpy(
                        cano_object_shelf_quat
                    )
                    cano_object_shelf_rot6d = matrix_to_rotation_6d_numpy(
                        cano_object_shelf_rot_mat
                    )
                    cano_object_shelf_data = np.concatenate(
                        (cano_object_shelf_pos, cano_object_shelf_rot6d), axis=-1
                    )

                    cano_left_hand_data = np.concatenate(
                        (cano_left_wrist_pos, cano_left_wrist_rot6d), axis=-1
                    )
                    cano_right_hand_data = np.concatenate(
                        (cano_right_wrist_pos, cano_right_wrist_rot6d), axis=-1
                    )
                    cano_object_data = np.concatenate(
                        (cano_object_pos, cano_object_rot6d), axis=-1
                    )
                    cano_camera_data = np.concatenate(
                        (cano_camera_pos, cano_camera_rot6d), axis=-1
                    )
                    shelf_valid = demo_data["objects"][obj_id]["shelf_valid"][
                        start_idx:end_idx
                    ]

                    window_data = {
                        "demo_id": demo_id,
                        "object_id": obj_id,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        # 'left_hand': cano_left_hand_data,
                        # 'right_hand': cano_right_hand_data,
                        "shelf_valid": shelf_valid,
                        "object_shelf": cano_object_shelf_data,
                        "object": cano_object_data,
                        "camera": cano_camera_data,
                        "left_hand": cano_left_positions.reshape(-1, 21 * 3),
                        "right_hand": cano_right_positions.reshape(-1, 21 * 3),
                        "left_hand_params": cano_left_mano_params_dict,
                        "right_hand_params": cano_right_mano_params_dict,
                        "left_hand_joints": cano_left_positions,
                        "right_hand_joints": cano_right_positions,
                        # 'wTc':
                        # 'wTo':
                        # 'wTo_shelf':
                        # 'left_hand': cano_left_hand_data,
                        # 'right_hand': cano_right_hand_data,
                        # 'left_hand_theta': cano_left_hand_data,
                        # 'right_hand': cano_right_hand_data,
                        # 'object':cano_object_data,
                    }

                    # Check if this is a motion window
                    is_motion = self._is_motion_window(cano_object_data)
                    window_data["is_motion"] = is_motion

                    window_data["mean_velocity"] = self._compute_mean_velocity(
                        cano_object_data
                    )

                    windows.append(window_data)

        # Apply sampling strategy
        # windows = self._apply_sampling_strategy(windows)
        return windows

    def _compute_mean_velocity(self, object_traj):
        """Compute the mean velocity of the object trajectory."""
        positions = object_traj[:, :3]
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return np.mean(velocities)

    def _is_motion_window(self, object_traj):
        """Determine if this window contains significant object motion."""
        if len(object_traj) < 2:
            return False

        # Extract positions (first 3 dimensions)
        positions = object_traj[:, :3]
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        # Check if there are consecutive frames above threshold
        above_threshold = velocities > self.motion_threshold

        # Find longest consecutive sequence
        max_consecutive = 0
        current_consecutive = 0

        for is_moving in above_threshold:
            if is_moving:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive >= self.min_motion_frames

    def _apply_sampling_strategy(self, windows):
        """Apply sampling strategy to balance motion vs stationary windows."""
        motion_windows = [w for w in windows if w["is_motion"]]
        stationary_windows = [w for w in windows if not w["is_motion"]]

        print(
            f"Original windows: {len(motion_windows)} motion, {len(stationary_windows)} stationary"
        )

        if self.sampling_strategy == "motion_only":
            return motion_windows
        elif self.sampling_strategy == "balanced":
            # Balance by taking equal numbers, up to the smaller set size
            min_count = min(len(motion_windows), len(stationary_windows))
            if min_count == 0:
                return windows  # Return all if one category is empty

            # Randomly sample to balance
            np.random.seed(42)  # For reproducibility
            motion_indices = np.random.choice(
                len(motion_windows), min_count, replace=False
            )
            stationary_indices = np.random.choice(
                len(stationary_windows), min_count, replace=False
            )

            balanced_windows = [motion_windows[i] for i in motion_indices]
            balanced_windows.extend([stationary_windows[i] for i in stationary_indices])

            print(
                f"Balanced to: {len(balanced_windows)} windows ({min_count} motion, {min_count} stationary)"
            )
            return balanced_windows
        else:  # 'random'
            return windows

    def _apply_train_val_split(self):
        """Apply train/validation split to windows."""

        if self.split == "all" or self.split == "mini":
            return self.windows

        # Use seed for reproducible splits
        np.random.seed(self.split_seed)

        # Shuffle windows for random split
        indices = np.arange(len(self.windows))
        np.random.shuffle(indices)

        # Split indices
        n_val = int(len(self.windows) * self.val_split_ratio)
        if self.split == "val":
            selected_indices = indices[:n_val]
        else:  # 'train'
            selected_indices = indices[n_val:]

        # Return selected windows
        selected_windows = [self.windows[i] for i in selected_indices]

        print(
            f"Split info: {len(selected_windows)}/{len(self.windows)} windows for {self.split}"
        )
        return selected_windows

    def _compute_normalization_stats(self):
        """Compute normalization statistics for the dataset."""
        all_left_hand = []
        all_right_hand = []
        all_objects = []

        for window in self.windows:
            all_left_hand.append(window["left_hand"])
            all_right_hand.append(window["right_hand"])
            all_objects.append(window["object"])

        if not all_left_hand:
            print("Warning: No windows found for normalization")
            self.stats = {}
            return

        # Stack all data
        all_left_data = np.vstack(all_left_hand)  # [N*T, D]
        all_right_data = np.vstack(all_right_hand)  # [N*T, D]
        all_object_data = np.vstack(all_objects)  # [N*T, D]

        # Use standard normalization approach
        left_mean = np.mean(all_left_data, axis=0)
        left_std = np.std(all_left_data, axis=0)
        right_mean = np.mean(all_right_data, axis=0)
        right_std = np.std(all_right_data, axis=0)
        object_mean = np.mean(all_object_data, axis=0)
        object_std = np.std(all_object_data, axis=0)

        # Ensure minimum std to prevent division by very small values
        min_std = 1e-6
        left_std = np.maximum(left_std, min_std)
        right_std = np.maximum(right_std, min_std)
        object_std = np.maximum(object_std, min_std)

        # use minmax for mean and std such that it's land in [-1, 1]
        object_min, object_max = (
            np.min(all_object_data, axis=0),
            np.max(all_object_data, axis=0),
        )
        object_std = (object_max - object_min) / 2.0
        object_mean = (object_max + object_min) / 2.0

        if all_left_data.shape[-1] != all_object_data.shape[-1]:
            all_data = np.concatenate((all_left_data, all_right_data), axis=0)
            all_min, all_max = np.min(all_data, axis=0), np.max(all_data, axis=0)
            all_std = (all_max - all_min) / 2.0
            all_mean = (all_max + all_min) / 2.0

            self.stats = {
                "left_hand_mean": all_mean,
                "left_hand_std": all_std,
                "right_hand_mean": all_mean,
                "right_hand_std": all_std,
                "object_mean": object_mean,
                "object_std": object_std,
            }

        else:
            all_data = np.concatenate(
                (all_left_data, all_right_data, all_object_data), axis=0
            )
            all_min, all_max = np.min(all_data, axis=0), np.max(all_data, axis=0)
            all_std = (all_max - all_min) / 2.0
            all_mean = (all_max + all_min) / 2.0

            self.stats = {
                "left_hand_mean": all_mean,
                "left_hand_std": all_std,
                "right_hand_mean": all_mean,
                "right_hand_std": all_std,
                "object_mean": all_mean,
                "object_std": all_std,
            }

        print("Computed normalization statistics")
        print(
            f"Object trajectory - Mean range: [{object_mean.min():.3f}, {object_mean.max():.3f}]"
        )
        print(
            f"Object trajectory - Std range: [{object_std.min():.3f}, {object_std.max():.3f}]"
        )

    def _setup_full_trajectory_data_from_windows(self):
        """Setup full trajectory data by concatenating all windows in the current split."""
        if not self.windows:
            print("No windows available for full trajectory setup")
            return

        # Collect all trajectory data from windows
        all_left_hand = []
        all_right_hand = []
        all_object_motion = []

        for window in self.windows:
            left_hand_data = to_tensor(window["left_hand"])  # [T, D]
            right_hand_data = to_tensor(window["right_hand"])  # [T, D]
            object_data = to_tensor(window["object"])  # [T, D]

            all_left_hand.append(left_hand_data)
            all_right_hand.append(right_hand_data)
            all_object_motion.append(object_data)

        # Concatenate all windows to form full trajectories
        self.left_hand_full = torch.cat(all_left_hand, dim=0)  # [Total_T, D]
        self.right_hand_full = torch.cat(all_right_hand, dim=0)  # [Total_T, D]
        self.object_motion_full = torch.cat(all_object_motion, dim=0)  # [Total_T, D]
        self.full_length = self.left_hand_full.shape[0]

        print(
            f"Setup full trajectory data: {self.full_length} total frames from {len(self.windows)} windows ({self.split} split)"
        )

        # Store demo info for compatibility (use info from first window)
        if self.windows:
            first_window = self.windows[0]
            self.demo_id = first_window.get("demo_id", "unknown")

    def has_full_trajectory_data(self):
        """Check if full trajectory data is available."""
        return (
            hasattr(self, "full_length")
            and hasattr(self, "left_hand_full")
            and hasattr(self, "right_hand_full")
            and hasattr(self, "object_motion_full")
        )

    def normalize_data(self, data, data_type):
        """Normalize data using computed statistics."""
        if not self.stats:
            return data

        mean = self.stats[f"{data_type}_mean"]
        std = self.stats[f"{data_type}_std"]

        return (data - mean) / std

    def denormalize_data(self, data, data_type):
        """Denormalize data using computed statistics."""
        if not self.stats:
            return data

        mean = self.stats[f"{data_type}_mean"]
        std = self.stats[f"{data_type}_std"]

        return data * std + mean

    def __len__(self):
        if self.is_train:
            return max(1000, len(self.windows))
        else:
            return len(self.windows)

    def __getitem__(self, idx):
        idx = idx % len(self.windows)
        window = self.windows[idx]

        ######### 2. add synthetic noise / use off-the-shelf noise to the data ##########
        # get noisy data
        window_noisy = self.add_noise_data(window)

        ######### 3. create motion repr ##########

        ######### 4. Convert to tensors ##########
        left_hand = to_tensor(window["left_hand"])  # [T, D]
        right_hand = to_tensor(window["right_hand"])  # [T, D]
        object_traj = to_tensor(window["object"])  # [T, D]

        # Normalize
        left_hand_norm = to_tensor(self.normalize_data(left_hand.numpy(), "left_hand"))
        right_hand_norm = to_tensor(
            self.normalize_data(right_hand.numpy(), "right_hand")
        )
        object_norm = to_tensor(self.normalize_data(object_traj.numpy(), "object"))

        traj_noisy_norm = to_tensor(
            self.normalize_data(window_noisy["object"], "object")
        )

        # Concatenate hand inputs for conditioning
        condition = torch.cat(
            [left_hand_norm, right_hand_norm, traj_noisy_norm], dim=-1
        )  # [T, 3*D]

        return {
            "condition": condition,  # [T, 2*D] - left and right hand trajectories
            "target": object_norm,  # [T, D] - object trajectory to denoise
            "traj_noisy_raw": to_tensor(window_noisy["object"]),
            "hand_raw": torch.cat(
                [left_hand, right_hand], dim=-1
            ),  # [T, 2*D] - left and right hand trajectories
            "target_raw": object_traj,  # [T, D] - unnormalized for evaluation
            "demo_id": window["demo_id"],
            "object_id": str(window["object_id"]),
            "is_motion": window["is_motion"],
            "mean_velocity": window["mean_velocity"],
        }

    def add_noise_data(self, can_window_dict):
        # just for 6d pose for now
        if self.noise_scheme == "real":
            can_window_dict_noisy = deepcopy(can_window_dict)
            can_window_dict_noisy["object"] = can_window_dict["object_shelf"]  # [T, D]
            # add mask
            mask = can_window_dict_noisy["shelf_valid"]  # [T, ]
            can_window_dict_noisy["object"] = (
                can_window_dict_noisy["object"] * mask[:, None]
            )
            return can_window_dict_noisy

        ######################################## add noise to  params

        can_window_dict_noisy = {}
        for param_name in [
            "object",
        ]:
            param = can_window_dict[param_name]  # [T, D]
            transl, rot6d = param[:, :3], param[:, 3:]

            transl_noisy = transl + np.random.normal(
                loc=0.0,
                scale=self.noise_std_params_dict["object_trans"],
                size=transl.shape,
            )

            rot_mat = rotation_6d_to_matrix_numpy(rot6d)
            # euler angle
            rot_euler = R.from_matrix(rot_mat).as_euler("zxy", degrees=True)
            rot_euler_noisy = rot_euler + np.random.normal(
                loc=0.0,
                scale=self.noise_std_params_dict["object_rot"],
                size=rot_euler.shape,
            )

            # add a constatn random translation noise to object traj
            if self.use_constant_noise:
                transl_noise = np.random.normal(
                    loc=0.0,
                    scale=self.noise_std_params_dict["traj_trans"],
                    size=(1, transl_noisy.shape[-1]),
                )
                transl_noisy = transl_noisy + transl_noise

                rot_noise = np.random.normal(
                    loc=0.0,
                    scale=self.noise_std_params_dict["traj_rot"],
                    size=(1, rot_euler_noisy.shape[-1]),
                )
                rot_euler_noisy = rot_euler_noisy + rot_noise

            rot_mat_noisy = R.from_euler(
                "zxy", rot_euler_noisy, degrees=True
            ).as_matrix()
            rot6d_noisy = matrix_to_rotation_6d_numpy(rot_mat_noisy)

            can_window_dict_noisy[param_name] = np.concatenate(
                [transl_noisy, rot6d_noisy], axis=-1
            )

            # if param_name == 'transl' or param_name == 'betas':
            #         noise_1 = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=can_window_dict[param_name].shape)
            #     cano_smplx_params_dict_noisy[param_name] = can_window_dict[param_name] + noise_1
            #     if param_name not in smplx_noise_dict.keys():
            #         smplx_noise_dict[param_name] = []
            #     smplx_noise_dict[param_name].append(noise_1)
            # elif param_name == 'global_orient':
            #     global_orient_mat = R.from_rotvec(can_window_dict['global_orient'])  # [145, 3, 3]
            #     global_orient_angle = global_orient_mat.as_euler('zxy', degrees=True)
            #     if self.load_noise:
            #         noise_global_rot = self.loaded_smplx_noise_dict[param_name][i*self.spacing]
            #     else:
            #         noise_global_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=global_orient_angle.shape)
            #     global_orient_angle_noisy = global_orient_angle + noise_global_rot
            #     cano_smplx_params_dict_noisy[param_name] = R.from_euler('zxy', global_orient_angle_noisy, degrees=True).as_rotvec()
            #     if param_name not in smplx_noise_dict.keys():
            #         smplx_noise_dict[param_name] = []
            #     smplx_noise_dict[param_name].append(noise_global_rot)  #  [145, 3] in euler angle
            # elif param_name == 'body_pose':
            #     body_pose_mat = R.from_rotvec(can_window_dict['body_pose'].reshape(-1, 3))
            #     body_pose_angle = body_pose_mat.as_euler('zxy', degrees=True)  # [145*21, 3]
            #     if self.load_noise:
            #         noise_body_pose_rot = self.loaded_smplx_noise_dict[param_name][i*self.spacing].reshape(-1, 3)
            #     else:
            #         noise_body_pose_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=body_pose_angle.shape)
            #     body_pose_angle_noisy = body_pose_angle + noise_body_pose_rot
            #     cano_smplx_params_dict_noisy[param_name] = R.from_euler('zxy', body_pose_angle_noisy, degrees=True).as_rotvec().reshape(-1, 21, 3)
            #     if param_name not in smplx_noise_dict.keys():
            #         smplx_noise_dict[param_name] = []
            #     smplx_noise_dict[param_name].append(noise_body_pose_rot.reshape(-1, 21, 3))  # [145, 21, 3]  in euler angle

        # ### using FK to obtain noisy joint positions from noisy smplx params
        # smplx_params_dict_noisy_torch = {}
        # for key in cano_smplx_params_dict_noisy.keys():
        #     smplx_params_dict_noisy_torch[key] = torch.FloatTensor(cano_smplx_params_dict_noisy[key]).to(self.device)
        # bs = smplx_params_dict_noisy_torch['transl'].shape[0]
        # # we do not consider face/hand details in RoHM
        # smplx_params_dict_noisy_torch['jaw_pose'] = torch.zeros(bs, 3).to(self.device)
        # smplx_params_dict_noisy_torch['leye_pose'] = torch.zeros(bs, 3).to(self.device)
        # smplx_params_dict_noisy_torch['reye_pose'] = torch.zeros(bs, 3).to(self.device)
        # smplx_params_dict_noisy_torch['left_hand_pose'] = torch.zeros(bs, 45).to(self.device)
        # smplx_params_dict_noisy_torch['right_hand_pose'] = torch.zeros(bs, 45).to(self.device)
        # smplx_params_dict_noisy_torch['expression'] = torch.zeros(bs, 10).to(self.device)
        # cano_positions_noisy = self.smplx_neutral(**smplx_params_dict_noisy_torch).joints[:, 0:22].detach().cpu().numpy()  # [clip_len, 22, 3]

        return can_window_dict_noisy


# Convenience function for creating datasets
def create_hand_to_object_dataset(
    data_path,
    window_size=120,
    single_demo=None,
    single_object=None,
    motion_threshold=0.01,
    sampling_strategy="balanced",
    split="train",
    val_split_ratio=0.2,
    augment=False,
):
    """
    Convenience function to create HandToObjectDataset.

    Args:
        data_path: Path to processed data pickle file
        window_size: Size of trajectory windows
        use_velocity: Whether to use 12D (with velocity) or 9D format (default: 9D)
        single_demo: For overfitting, specify single demo ID
        single_object: For overfitting, specify single object ID
        motion_threshold: Threshold for motion detection
        sampling_strategy: 'balanced', 'motion_only', or 'random'
        split: 'train', 'val', or 'all'
        val_split_ratio: Fraction of data for validation
        augment: Whether to apply data augmentation
    """
    return HandToObjectDataset(
        data_path=data_path,
        window_size=window_size,
        single_demo=single_demo,
        single_object=single_object,
        motion_threshold=motion_threshold,
        sampling_strategy=sampling_strategy,
        split=split,
        val_split_ratio=val_split_ratio,
        augment=augment,
    )


def vis_traj():
    import plotly.graph_objects as go

    data = load_pickle("/move/u/yufeiy2/data/HOT3D/pred_pose/mini_P0001_624f2ba9.npz")
    for demo_id, demo_data in data.items():
        fig = go.Figure()
        for uid, obj_data in demo_data["objects"].items():
            wTo = obj_data["wTo"]  # (T, 4, 4)
            wTo_shelf = obj_data["wTo_shelf"]  # (T, 4, 4)
            wTo_shelf_valid = obj_data["shelf_valid"]  # (T, )

            wTo_pos = wTo[:, :3, 3]
            wTo_shelf_pos = wTo_shelf[:, :3, 3]

            # plot as connecting 3D lines
            fig.add_trace(
                go.Scatter3d(
                    x=wTo_pos[:, 0],
                    y=wTo_pos[:, 1],
                    z=wTo_pos[:, 2],
                    mode="lines",
                    name=f"wTo_{uid}",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=wTo_shelf_pos[:, 0],
                    y=wTo_shelf_pos[:, 1],
                    z=wTo_shelf_pos[:, 2],
                    mode="lines",
                    name=f"shelf_{uid}",
                )
            )
        os.makedirs("outputs", exist_ok=True)
        fig.write_html(f"outputs/traj_{demo_id}.html")

        break


if __name__ == "__main__":
    # # Test the dataset (default 9D format)
    # dataset = create_hand_to_object_dataset(
    #     data_path="data/processed_data.pkl",
    #     window_size=120,
    #     use_velocity=False,  # Default: 9D format (pos + rot)
    #     single_demo="P0001_10a27bf7",  # For overfitting test
    #     single_object="37787722328019",
    # )

    # print(f"Dataset size: {len(dataset)}")

    # if len(dataset) > 0:
    #     sample = dataset[0]
    #     print(f"Sample shapes:")
    #     print(f"  Condition: {sample['condition'].shape}")  # [T, 2*D]
    #     print(f"  Target: {sample['target'].shape}")  # [T, D]
    #     print(f"  Demo ID: {sample['demo_id']}")
    #     print(f"  Object ID: {sample['object_id']}")
    #     print(f"  Is motion: {sample['is_motion']}")

    vis_traj()
