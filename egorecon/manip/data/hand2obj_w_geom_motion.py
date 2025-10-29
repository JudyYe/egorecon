#!/usr/bin/env python3
from glob import glob
import logging
import hydra
from tqdm import tqdm
import json
import os
import os.path as osp
import pickle
from copy import deepcopy

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from jutils import geom_utils, mesh_utils
from ...utils.motion_repr import cano_seq_mano, HandWrapper
from ...utils.rotation_utils import (
    rotation_6d_to_matrix_numpy,
    mat_to_9d_numpy,
)
from ..lafan1.utils import rotate_at_frame_w_obj
from ...visualization.pt3d_visualizer import Pt3dVisualizer

from pytorch3d.ops import sample_points_from_meshes
from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from preprocess.est_noise import generate_noisy_trajectory
from .utils import get_norm_stats, load_pickle, to_tensor


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
        motion_threshold=0.05,  # Threshold for considering motion vs stationary
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
        aug_world=False,
        aug_cano=True,
        use_constant_noise=False,
        one_window=False,
        t0=300,
        # split_file=None,
        noise_scheme="syn",  # 'syn', 'real'
        opt=None,
        data_cfg=None,
        **kwargs,
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
        self.one_window = one_window
        self.t0 = t0
        self.use_constant_noise = use_constant_noise
        self.noise_scheme = noise_scheme
        self.split_file = data_cfg.split_file
        self.opt = opt
        self.data_cfg = data_cfg
        self.hand_wrapper = HandWrapper(opt.paths.mano_dir)

        self.bps_path = osp.join(opt.paths.data_dir, "bps/bps.pt")
        # bps_dir = osp.join(opt.paths.data_dir, 'bps')

        # dest_obj_bps_npy_folder = os.path.join(bps_dir, "object_bps_npy_files_joints24")
        # dest_obj_bps_npy_folder_for_test = os.path.join(bps_dir, "object_bps_npy_files_for_eval_joints24")

        # if self.is_train:
        #     self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder
        # else:
        #     self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test

        self.prep_bps_data()

        self.mano_model_path = opt.paths.mano_dir
        self.hand_wrapper = HandWrapper(opt.paths.mano_dir)
        self.sided_mano_models = self.hand_wrapper._sided_mano_models
        # self.sided_mano_models = {
        #     "left": smplx.create(
        #         os.path.join(self.mano_model_path, "MANO_LEFT.pkl"),
        #         "mano",
        #         is_rhand=False,
        #         num_pca_comps=15,
        #     ),
        #     "right": smplx.create(
        #         os.path.join(self.mano_model_path, "MANO_RIGHT.pkl"),
        #         "mano",
        #         is_rhand=True,
        #         num_pca_comps=15,
        #     ),
        # }

        # Load processed data
        print(f"Loading data from {data_path}...")
        self.processed_data = load_pickle(data_path)
        print(f"Found {len(self.processed_data)} demonstrations")
        self.pose_dim = 9

        self.processed_data = self._filter_data(single_demo, single_object)
        print(
            f"Filtered to demo: {single_demo} object: {single_object}, total: {len(self.processed_data)} demonstrations for overfitting"
        )

        shelf_data_list = []
        for shelf_name in self.data_cfg.shelf:
            shelf_data = load_pickle(shelf_name)
            shelf_data = {
                seq: v for seq, v in shelf_data.items() if seq in self.processed_data
            }
            for seq, data in shelf_data.items():
                assert len(data["wTc"]) == len(self.processed_data[seq]["wTc"])
            shelf_data_list.append(shelf_data)
        self.shelf_data = shelf_data_list

        # Create windows
        print(f"Creating windows for {self.split}...")
        self.windows = self._create_windows()
        print("Created ", len(self.windows), " windows")
        print("Filtering dynamic windows...")
        if self.opt.dyn_only:
            # only keep the dynamic windows
            self._filter_windows(threshold=self.motion_threshold)
            print(f"Filtered to {len(self.windows)} dynamic windows")

        # Apply train/validation split
        print(f"Created {len(self.windows)} {self.split} windows")

        # Compute normalization statistics
        # self._compute_normalization_stats()
        self.stats = {}

        with open(
            osp.join(self.opt.paths.data_dir, "cache", f"noise_stats_hand.pkl"), "rb"
        ) as f:
            self.noise_std_params_dict = pickle.load(f)
        # {
        #     "global_orient": noise_std_mano_global_rot,
        #     "transl": noise_std_mano_trans,
        #     "hand_pose": noise_std_mano_body_rot,
        #     "betas": noise_std_mano_betas,
        #     "object_rot": noise_std_obj_rot,
        #     "object_trans": noise_std_obj_trans,
        #     "traj_rot": traj_std_obj_rot,
        #     "traj_trans": traj_std_obj_trans,
        # }

    def compute_object_geo_bps(self, obj_verts, obj_trans):
        # obj_verts: T X Nv X 3, obj_trans: T X 3
        bps_object_geo = self.bps_torch.encode(
            x=obj_verts,
            feature_type=["deltas"],
            custom_basis=self.obj_bps.repeat(obj_trans.shape[0], 1, 1)
            + obj_trans[:, None, :],
        )["deltas"]  # T X N X 3

        return bps_object_geo

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(
                1, -1, 3
            )

            bps = {
                "obj": bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            os.makedirs(osp.dirname(self.bps_path), exist_ok=True)
            torch.save(bps, self.bps_path)

        self.bps = torch.load(self.bps_path)
        self.bps_torch = bps_torch()

        self.obj_bps = self.bps["obj"]

        # Load object geometry
        self.object_library = {}
        self.object_library_mesh = Pt3dVisualizer.setup_template(
            self.opt.paths.object_mesh_dir
        )
        for uid, mesh in self.object_library_mesh.items():
            self.object_library[uid] = sample_points_from_meshes(mesh, 50000)

    def _filter_data(self, single_demo=None, single_object=None):
        """Filter data for overfitting on specific demo/object."""
        filtered_data = {}
        # if self.split == "mini":
        #     seq = list(self.processed_data.keys())[0]
        #     seq_list = [seq]
        # else:
        seq_list = json.load(open(self.split_file))[self.split]

        for seq in seq_list:
            if seq not in self.processed_data:
                continue
            filtered_data[seq] = {}
            if single_object:
                # 225397651484143 : bowl
                obj_list = single_object.split("+")
            else:
                obj_list = list(self.processed_data[seq]["objects"].keys())

            logging.warning("Well let's remove this in the end!!")
            # if self.one_window:
            #     t0 = self.t0
            #     t1 = t0 + 120
            # else:
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
                    if key == "intrinsic":
                        filtered_data[seq][key] = value
                    else:
                        filtered_data[seq][key] = value[t0:t1]
                else:
                    print(key, type(value))
                    filtered_data[seq][key] = value

        return filtered_data

    def _filter_windows(self, threshold=0.05):
        """Filter windows to only include dynamic windows."""
        origin_size = len(self.windows)
        filtered_windows = [
            window for window in self.windows if window["is_motion"] > threshold
        ]
        self.windows = filtered_windows
        print(f"{origin_size} -> {len(self.windows)} dynamic windows")

    def _create_windows(self):
        use_cache = self.opt.datasets.use_cache
        save_cache = self.opt.datasets.save_cache
        
        cache_name = osp.join(
            self.opt.paths.data_dir,
            "cache",
            f"{self.data_cfg.name}_{self.split}_{self.opt.get('coord', 'camera')}.npz",
        )

        if osp.exists(cache_name) and use_cache:
            print("loading window cache from ", cache_name)
            self.windows = pickle.load(open(cache_name, "rb"))
            return self.windows

        print("creating window cache and save to ", cache_name)

        windows = []
        for demo_id, demo_data in tqdm(
            self.processed_data.items(), desc="Creating windows"
        ):
            for obj_id, obj_data in demo_data["objects"].items():
                obj_traj = obj_data["wTo"]
                T = len(obj_traj)

                for start_idx in range(
                    0, T - self.window_size + 1, self.window_size // 4
                ):
                    end_idx = start_idx + self.window_size
                    if "wTo_valid" in obj_data:
                        obj_valid = obj_data["wTo_valid"][start_idx:end_idx]
                    else:
                        obj_valid = np.ones(self.window_size).astype(bool)
                        logging.warning(
                            f"No object valid mask found for {demo_id}, {obj_id}"
                        )

                    if not np.all(obj_valid):
                        continue

                    left_wrist = demo_data["left_hand"]["theta"][start_idx:end_idx]
                    left_wrist_aa, left_wrist_pos, left_wrist_hA = np.split(
                        left_wrist, [3, 6], axis=-1
                    )
                    right_wrist = demo_data["right_hand"]["theta"][start_idx:end_idx]
                    right_wrist_aa, right_wrist_pos, right_wrist_hA = np.split(
                        right_wrist, [3, 6], axis=-1
                    )

                    object_pos = obj_traj[start_idx:end_idx][:, :3, 3]
                    object_rot_mat = obj_traj[start_idx:end_idx][:, :3, :3]
                    object_quat = R.from_matrix(object_rot_mat).as_quat()[
                        :, [3, 0, 1, 2]
                    ]  # wxyz

                    coord = self.opt.get("coord", "camera")
                    if coord == "camera":
                        camera_pos = demo_data["wTc"][start_idx:end_idx][:, :3, 3]
                        camera_rot = demo_data["wTc"][start_idx:end_idx][:, :3, :3]
                        camera_quat = R.from_matrix(camera_rot).as_quat()[
                            :, [3, 0, 1, 2]
                        ]  # wxyz
                        # Canonicalize the data using the camera frame
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
                        cano_camera_pos = cano_camera_pos.reshape(-1, 3)
                        canoTw = np.eye(4)
                        canoTw[:3, :3] = canoTw_rot
                        canoTw[:3, 3] = -cano_camera_pos[0:1, :].copy()
                    elif coord == "right_wrist":
                        # 1st frame of right wrist
                        mano_params_dict = {
                            "global_orient": right_wrist_aa,
                            "transl": right_wrist_pos,
                            "betas": demo_data["right_hand"]["shape"][
                                start_idx:end_idx
                            ],
                            "hand_pose": right_wrist_hA,
                        }
                        cano_right_positions, cano_right_mano_params_dict, canoTw = (
                            cano_seq_mano(
                                None,
                                None,
                                mano_params_dict,
                                mano_model=self.sided_mano_models["right"],
                                device="cpu",
                                return_transf_mat=True,
                            )
                        )
                    else:
                        raise ValueError(f"Invalid coordinate system: {coord}")

                    # canonicalize: lr-hand, object, camera
                    cano_left_positions, cano_left_params = self.cano_hand_param(
                        canoTw,
                        demo_data["left_hand"]["theta"][start_idx:end_idx],
                        demo_data["left_hand"]["shape"][start_idx:end_idx],
                        "left",
                    )
                    cano_right_positions, cano_right_params = self.cano_hand_param(
                        canoTw,
                        demo_data["right_hand"]["theta"][start_idx:end_idx],
                        demo_data["right_hand"]["shape"][start_idx:end_idx],
                        "right",
                    )

                    # mano_params_dict = {
                    #     "global_orient": left_wrist_aa,
                    #     "transl": left_wrist_pos,
                    #     "betas": demo_data["left_hand"]["shape"][start_idx:end_idx],
                    #     "hand_pose": left_wrist_hA,
                    # }

                    # cano_left_positions, cano_left_mano_params_dict = cano_seq_mano(
                    #     canoTw=canoTw,
                    #     positions=None,
                    #     mano_params_dict=mano_params_dict,
                    #     mano_model=self.sided_mano_models["left"],
                    #     device="cpu",
                    # )  # joints position

                    # mano_params_dict = {
                    #     "global_orient": right_wrist_aa,
                    #     "transl": right_wrist_pos,
                    #     "betas": demo_data["right_hand"]["shape"][start_idx:end_idx],
                    #     "hand_pose": right_wrist_hA,
                    # }

                    # cano_right_positions, cano_right_mano_params_dict = cano_seq_mano(
                    #     canoTw=canoTw,
                    #     positions=None,
                    #     mano_params_dict=mano_params_dict,
                    #     mano_model=self.sided_mano_models["right"],
                    #     device="cpu",
                    # )  # joints position

                    # canonicalize object
                    wTo = obj_traj[start_idx:end_idx]  # (T, 4, 4)
                    wTo = canoTw[None] @ wTo

                    # canonicalize camera
                    wTc = demo_data["wTc"][start_idx:end_idx]  # (T, 4, 4)
                    wTc = canoTw[None] @ wTc

                    # create 9D representation for object
                    cano_object_data = mat_to_9d_numpy(wTo)

                    # create 9D representation for camera
                    cano_camera_data = mat_to_9d_numpy(wTc)

                    if "contact_lr" in obj_data:
                        contact = obj_data["contact_lr"][start_idx:end_idx]
                    else:
                        contact = np.zeros([self.window_size, 2])
                        logging.warning(
                            f"are you sure you don't have contact data?, {obj_data.keys()}, {self.data_path} {demo_id} {obj_id}"
                        )

                    shelf_left_positions_list = []
                    shelf_right_positions_list = []
                    shelf_left_params_list = []
                    shelf_right_params_list = []
                    shelf_ready_list = []
                    shelf_left_valid_list = []
                    shelf_right_valid_list = []
                    for shelf_data_all in self.shelf_data:
                        shelf_data = shelf_data_all[demo_id]
                        shelf_left_positions, shelf_left_params = self.cano_hand_param(
                            canoTw,
                            shelf_data["left_hand"]["theta"][start_idx:end_idx],
                            shelf_data["left_hand"]["shape"][start_idx:end_idx],
                            "left",
                        )
                        shelf_right_positions, shelf_right_params = (
                            self.cano_hand_param(
                                canoTw,
                                shelf_data["right_hand"]["theta"][start_idx:end_idx],
                                shelf_data["right_hand"]["shape"][start_idx:end_idx],
                                "right",
                            )
                        )
                        if np.isnan(shelf_left_positions).any():
                            import ipdb; ipdb.set_trace()
                        shelf_left_positions_list.append(shelf_left_positions)
                        shelf_right_positions_list.append(shelf_right_positions)
                        shelf_left_params_list.append(shelf_left_params)
                        shelf_right_params_list.append(shelf_right_params)
                        shelf_ready_list.append(shelf_data["ready"])
                        shelf_left_valid_list.append(
                            shelf_data["left_hand"]["valid"][start_idx:end_idx]
                        )
                        shelf_right_valid_list.append(
                            shelf_data["right_hand"]["valid"][start_idx:end_idx]
                        )
                    
                    window_data = {
                        "demo_id": demo_id,
                        "object_id": obj_id,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "object": cano_object_data,
                        "object_valid": obj_valid,
                        "wTc": cano_camera_data,
                        "intr": demo_data["intrinsic"],
                        "left_hand": cano_left_positions.reshape(-1, 21 * 3),
                        "right_hand": cano_right_positions.reshape(-1, 21 * 3),
                        "left_hand_params": cano_left_params,
                        "right_hand_params": cano_right_params,
                        "contact": contact,
                        "shelf_left_hand": np.stack(
                            shelf_left_positions_list, axis=0
                        ),  # [N, T, 3]
                        "shelf_right_hand": np.stack(
                            shelf_right_positions_list, axis=0
                        ),  # [N, T, 3]
                        "shelf_left_hand_params": np.stack(
                            shelf_left_params_list, axis=0
                        ),  # [N, T, 15]
                        "shelf_right_hand_params": np.stack(
                            shelf_right_params_list, axis=0
                        ),  # [N, T, 15]
                        "shelf_ready": np.stack(shelf_ready_list, axis=0),  # [N, ]
                        "shelf_left_valid": np.stack(
                            shelf_left_valid_list, axis=0
                        ),  # [N, T, ]
                        "shelf_right_valid": np.stack(
                            shelf_right_valid_list, axis=0
                        ),  # [N, T, ]
                        # "canoTw": canoTw,
                    }

                    # load

                    is_motion = self._is_motion_window(cano_object_data)
                    window_data["is_motion"] = is_motion

                    if self.window_check(window_data):
                        windows.append(window_data)

        os.makedirs(osp.dirname(cache_name), exist_ok=True)
        if save_cache:
            with open(cache_name, "wb") as f:
                pickle.dump(windows, f)
        return windows

    def _param2dict(self, params, shape=None):
        if shape is None:
            shape = params[..., -10:]
            params = params[..., :-10]
        aa, pos, hA = np.split(params, [3, 6], axis=-1)
        return {
            "global_orient": aa,
            "transl": pos,
            "hand_pose": hA,
            "betas": shape,
        }

    def _dict2param(self, mano_params_dict):
        aa = mano_params_dict["global_orient"]
        pos = mano_params_dict["transl"]
        hA = mano_params_dict["hand_pose"]
        shape = mano_params_dict["betas"]
        return np.concatenate([aa, pos, hA, shape], axis=-1)

    def cano_hand_param(self, canoTw, theta, shape, side):
        mano_params_dict = self._param2dict(theta, shape)
        cano_positions, cano_mano_params_dict = cano_seq_mano(
            canoTw=canoTw,
            positions=None,
            mano_params_dict=mano_params_dict,
            mano_model=self.sided_mano_models[side],
            device="cpu",
        )  # joints position
        params = self._dict2param(cano_mano_params_dict)
        return cano_positions, params

    def window_check(self, window_data):
        """Check if window has valid object data."""
        if not np.any(window_data["object_valid"]):
            return False
        return True

    def _is_motion_window(self, object_traj, motion_threshold=0.05):
        """Determine if this window contains significant object motion.
        :param object_traj: [T, 3+6] - position (3D) + rotation (6D)
        :return: bool
        """
        # Extract position and rotation from trajectory
        positions = object_traj[:, :3]  # [T, 3]
        rotations = object_traj[:, 3:]  # [T, 6]

        # Convert 6D rotation to rotation matrices
        rot_matrices = rotation_6d_to_matrix_numpy(rotations)  # [T, 3, 3]

        # Define object keypoints: origin (0,0,0) and x,y,z axes with 0.01m length
        keypoints_local = np.array(
            [
                [0.0, 0.0, 0.0],  # origin
                [0.01, 0.0, 0.0],  # x-axis
                [0.0, 0.01, 0.0],  # y-axis
                [0.0, 0.0, 0.01],  # z-axis
            ]
        )  # [4, 3]

        # Transform keypoints to world coordinates for all frames at once
        # keypoints_local: [4, 3], rot_matrices: [T, 3, 3], positions: [T, 3]
        # Use batch matrix multiplication: [T, 3, 3] @ [3, 4] -> [T, 3, 4]
        keypoints_world = np.einsum(
            "tij,jk->tik", rot_matrices, keypoints_local.T
        )  # [T, 3, 4]
        # Transpose to [T, 4, 3] and add translation
        keypoints_world = (
            keypoints_world.transpose(0, 2, 1) + positions[:, np.newaxis, :]
        )  # [T, 4, 3]

        # Calculate accumulated motion for all keypoints at once
        # Calculate displacement between consecutive frames for all keypoints
        # keypoints_world: [T, 4, 3] -> diff: [T-1, 4, 3]
        displacements = np.linalg.norm(
            np.diff(keypoints_world, axis=0), axis=2
        )  # [T-1, 4]
        displacements = displacements.mean(axis=-1)

        # Sum displacements across time for each keypoint, then sum across all keypoints
        total_motion = np.sum(displacements)  # Scalar

        # Check if total motion exceeds threshold
        return total_motion

    def set_metadata(self):
        """Compute normalization statistics for the dataset."""
        meta_file = self.data_cfg.meta_file
        with open(meta_file, "rb") as f:
            metadata = pickle.load(f)

        mean, std = get_norm_stats(metadata, self.opt, "target")
        self.mean_target = torch.FloatTensor(mean)
        self.std_target = torch.FloatTensor(std)

        mean, std = get_norm_stats(metadata, self.opt, "condition")
        self.mean_condition = torch.FloatTensor(mean)
        self.std_condition = torch.FloatTensor(std)

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

    def normalize_target(self, target):
        mean = self.mean_target
        std = self.std_target
        return (target - mean) / std

    def normalize_condition(self, condition):
        mean = self.mean_condition
        std = self.std_condition
        return (condition - mean) / std

    # def normalize_data(self, data, data_type):
    #     """Normalize data using computed statistics."""
    #     if not self.stats:
    #         print("no stats!!! you'd better doing normailization right now!")
    #         return data

    #     mean = self.stats[f"{data_type}_mean"]
    #     std = self.stats[f"{data_type}_std"]

    #     return (data - mean) / std

    def __len__(self):
        if self.is_train:
            return max(100000, len(self.windows))
        else:
            return len(self.windows)

    def __getitem__(self, idx):
        idx = idx % len(self.windows)
        window = deepcopy(self.windows[idx])

        # # Add synthetic noise to the data
        window_noisy = None
        if self.opt.hand == "cond_out":
            window_noisy = self.add_noise_data(window)

        # Add geometry and random canonical augmentation, for object, nothign to do with noisy window
        # if self.opt.datasets.augument.get("aug_cano", True):
        window, newoTo = self.augment_cano_object(window)

        if self.opt.datasets.augument.get("aug_world", False):
            print(
                "TODO: augment world cooridate too!"
            )  # random rotation around gravity direction
            assert False

        # get target, condition , Normalize
        object_traj = to_tensor(window["object"])
        contact = to_tensor(window["contact"])

        def _legacy_to_hand_param(window_data):
            if isinstance(window_data, dict):
                return self._dict2param(window_data)
            else:
                return to_tensor(window_data)

        def get_hand_rep_tensor(window, hand_rep):
            if hand_rep == "joint":
                hand_rep = torch.cat(
                    [
                        to_tensor(window["left_hand"]).reshape(
                            self.window_size, 21 * 3
                        ),
                        to_tensor(window["right_hand"]).reshape(
                            self.window_size, 21 * 3
                        ),
                    ],
                    dim=-1,
                )
            elif hand_rep == "theta":
                left_hand_params = to_tensor(
                    _legacy_to_hand_param(window["left_hand_params"])
                )
                right_hand_params = to_tensor(
                    _legacy_to_hand_param(window["right_hand_params"])
                )
                hand_rep = torch.cat([left_hand_params, right_hand_params], dim=-1)

            elif hand_rep == "joint_theta":
                left_hand_joints = to_tensor(window["left_hand"]).reshape(
                    self.window_size, 21 * 3
                )
                right_hand_joints = to_tensor(window["right_hand"]).reshape(
                    self.window_size, 21 * 3
                )
                left_hand_theta = to_tensor(
                    _legacy_to_hand_param(window["left_hand_params"])
                )
                right_hand_theta = to_tensor(
                    _legacy_to_hand_param(window["right_hand_params"])
                )
                hand_rep = torch.cat(
                    [
                        left_hand_joints,
                        left_hand_theta,
                        right_hand_joints,
                        right_hand_theta,
                    ],
                    dim=-1,
                )
            else:
                raise ValueError(f"Invalid hand representation: {hand_rep}")
            return hand_rep

        hand_rep_method = self.opt.get("hand_rep", "joint")
        hand_rep = get_hand_rep_tensor(window, hand_rep_method)
        if window_noisy is not None:
            noisy_hand_rep = get_hand_rep_tensor(window_noisy, hand_rep_method)

        target = object_traj
        condition = torch.zeros([self.window_size, 0])

        # create target
        hand_io = self.opt.get("hand", "cond")
        if hand_io in ["out", "cond_out"]:
            target = torch.cat([target, hand_rep], dim=-1)
        if "output" in self.opt and self.opt.output.contact:
            target = torch.cat([target, contact], dim=-1)

        target_unnorm = target.clone()
        target = self.normalize_target(target)

        # create condition
        if hand_io == "cond":
            condition = torch.cat([condition, hand_rep], dim=-1)
        elif hand_io == "cond_out":
            condition = torch.cat([condition, noisy_hand_rep], dim=-1)
        condition = self.normalize_condition(condition)
        # mask out condition if there is noisy input?
        if self.opt.hand == "cond_out":
            D = hand_rep.shape[-1] // 2
            cond_mask = torch.cat(
                [
                    to_tensor(window_noisy["left_hand_valid"])[..., None].repeat(1, D),
                    to_tensor(window_noisy["right_hand_valid"])[..., None].repeat(1, D),
                ],
                dim=-1,
            ).float()  # (T, 2*D)
            condition = torch.where(cond_mask > 0, condition, 0)
            # condition = condition * cond_mask
        else:
            cond_mask = torch.ones_like(condition)
                
        left_hand_param = to_tensor(_legacy_to_hand_param(window["left_hand_params"]))
        right_hand_param = to_tensor(_legacy_to_hand_param(window["right_hand_params"]))
        data = {
            "contact": contact,
            "condition": condition,  # [T, 2*D] - left and right hand trajectories
            "cond_mask": cond_mask,
            "target": target,  # [T, D] - object trajectory to denoise
            "hand_raw": torch.cat(
                [to_tensor(window["left_hand"]), to_tensor(window["right_hand"])],
                dim=-1,
            ),
            "left_hand_params": left_hand_param,
            "right_hand_params": right_hand_param,
            "target_raw": target_unnorm,
            "wTo": object_traj,  # [T, D] - unnormalized for evaluation
            "demo_id": window["demo_id"],
            "object_id": str(window["object_id"]),
            # "is_motion": window["is_motion"],
            "object_valid": window["object_valid"],
            "intr": to_tensor(window["intr"]),
            "wTc": to_tensor(window["wTc"]),
            "newTo": window["newTo"],
            "newPoints": window["newPoints"][0],
            "newMesh": window["newMesh"],
            "start_idx": window["start_idx"],
            "end_idx": window["end_idx"],
            "ind": idx,
        }
        return data

    def transform_wTo_traj(self, wTo, newTo):
        wTo = geom_utils.se3_to_matrix_v2(torch.FloatTensor(wTo))  # (T, 4, 4)

        oTnew = geom_utils.inverse_rt_v2(newTo)
        wTnew = wTo @ oTnew  # (T, 4, 4)
        wTnew = geom_utils.matrix_to_se3_v2(wTnew).cpu().numpy()
        return wTnew

    def augment_cano_object(self, window, newTo=None):
        """Augment object with random canonical orientation.

        Load object geometry, apply random orientation as canonical pose,
        and transform the object trajectory accordingly.

        Args:
            window: Window data containing object information
            newTo: Optional transformation matrix. If None, generates random rotation.

        Returns:
            tuple: (updated_window, newTo)
        """
        aug_cano = self.opt.datasets.augument.get("aug_cano", True)
        
        if newTo is None:
            if aug_cano and self.is_train and np.random.rand() < self.opt.datasets.augument.get(
                "use_aug_cano", 0.5
            ):
                newTo_rot = geom_utils.random_rotations(
                    1,
                )  # (1, 3, 3)
                # maximum radius is 0.25. let's jitter translation with 0.05 (5cm)
                newTo_tsl = (torch.randn(1, 3) * 0.05).clamp(-0.05, 0.05)
                newTo = geom_utils.rt_to_homo(newTo_rot, newTo_tsl)  # (1, 4, 4)
            else:
                newTo = torch.eye(4)[None]


        window["object"] = self.transform_wTo_traj(window["object"], newTo)

        # Load object geometry
        oPoints = self.object_library[window["object_id"]]
        newPoints = mesh_utils.apply_transform(oPoints, newTo)  # (1, P, 3)

        # Randomly sample points for visualization
        nP = min(5000, newPoints.shape[1])
        index = torch.randint(0, newPoints.shape[1], (nP,))
        newPoints = newPoints[:, index, :]

        oMesh = self.object_library_mesh[window["object_id"]]
        newMesh = mesh_utils.apply_transform(oMesh, newTo)

        window["newTo"] = newTo
        window["newPoints"] = newPoints
        window["newMesh"] = newMesh
        return window, newTo

    def add_noise_data(self, can_window_dict):
        """Add noise to the hand for training."""
        # can_window_dict_noisy = deepcopy(can_window_dict)

        can_window_dict_noisy = {}
        use_real_noise = False  # default to use synthetic noise
        shelf_idx = None
        if (
            not self.is_train or np.random.rand() < 0.5
        ):  # if test, or 50% of training data, use real noise
            shelf_idx = np.random.randint(0, len(can_window_dict["shelf_left_hand"]))
            use_real_noise = can_window_dict["shelf_ready"][
                shelf_idx
            ]  # if not ready, not use real noise

        if use_real_noise:
            can_window_dict_noisy["left_hand_params"] = can_window_dict[
                "shelf_left_hand_params"
            ][shelf_idx]
            can_window_dict_noisy["right_hand_params"] = can_window_dict[
                "shelf_right_hand_params"
            ][shelf_idx]

            can_window_dict_noisy["left_hand"] = can_window_dict["shelf_left_hand"][
                shelf_idx
            ]
            can_window_dict_noisy["right_hand"] = can_window_dict["shelf_right_hand"][
                shelf_idx
            ]

            can_window_dict_noisy["left_hand_valid"] = can_window_dict[
                "shelf_left_valid"
            ][shelf_idx]
            can_window_dict_noisy["right_hand_valid"] = can_window_dict[
                "shelf_right_valid"
            ][shelf_idx]

        else:
            clean_left_params = can_window_dict["left_hand_params"]
            left_theta = generate_noisy_trajectory(
                clean_left_params[None, ..., :-10],
                **self.noise_std_params_dict["left_hand_theta"],
            )[0]
            left_shape = generate_noisy_trajectory(
                clean_left_params[None, ..., -10:],
                **self.noise_std_params_dict["left_hand_shape"],
            )[0]
            left_params = torch.cat([left_theta, left_shape], dim=-1)  # [T, D]

            clean_right_params = can_window_dict["right_hand_params"]
            right_theta = generate_noisy_trajectory(
                clean_right_params[None, ..., :-10],
                **self.noise_std_params_dict["right_hand_theta"],
            )[0]
            right_shape = generate_noisy_trajectory(
                clean_right_params[None, ..., -10:],
                **self.noise_std_params_dict["right_hand_shape"],
            )[0]
            right_params = torch.cat([right_theta, right_shape], dim=-1)

            _, _, left_joints = self.hand_wrapper.hand_para2verts_faces_joints(
                left_params, side="left"
            )
            _, _, right_joints = self.hand_wrapper.hand_para2verts_faces_joints(
                right_params, side="right"
            )
            T = left_joints.shape[0]
            left_joints = left_joints.reshape(T, -1)
            right_joints = right_joints.reshape(T, -1)

            can_window_dict_noisy["left_hand_params"] = left_params
            can_window_dict_noisy["right_hand_params"] = right_params

            can_window_dict_noisy["left_hand"] = left_joints
            can_window_dict_noisy["right_hand"] = right_joints

            can_window_dict_noisy["left_hand_valid"] = torch.ones(T)
            can_window_dict_noisy["right_hand_valid"] = torch.ones(T)
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


@hydra.main(config_path="../../../config", config_name="train", version_base=None)
def create_mini_dataset(opt):
    with open(opt.testdata.data_path, "rb") as f:
        data = pickle.load(f)
    # data = load_pickle(opt.testdata.data_path)
    split_file = opt.testdata.split_file
    split = opt.testdata.testsplit

    split_data = json.load(open(split_file))
    seq_list = split_data[split]
    new_data = {}
    seq = seq_list[0]
    for seq in seq_list[:3]:
        new_data[seq] = data[seq]

    mini_file = opt.testdata.data_path.replace(".pkl", "_mini.pkl")
    print(
        f"Saving mini dataset to {mini_file} {len(data)} -> {len(seq_list)} -> {len(new_data)}"
    )

    split_data["minitest"] = list(new_data.keys())
    print(split_file)

    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=4)

    with open(mini_file, "wb") as f:
        pickle.dump(new_data, f)


@hydra.main(config_path="../../../config", config_name="train", version_base=None)
@torch.no_grad()
def vis_clip(opt):
    # okay if we load seq
    import plotly.graph_objects as go

    # from visualization.rerun_visualizer import RerunVisualizer
    from egorecon.visualization.pt3d_visualizer import Pt3dVisualizer
    import smplx
    from jutils import geom_utils, image_utils, mesh_utils, model_utils
    from pytorch3d.structures import Meshes
    from egorecon.utils.motion_repr import HandWrapper

    pt3d_viz = Pt3dVisualizer(
        exp_name="vis_traj",
        save_dir=osp.join("outputs/debug_vis", opt.expname),
        mano_models_dir=opt.paths.mano_dir,
        object_mesh_dir=opt.paths.object_mesh_dir,
    )

    mano_model_folder = "assets/mano"
    device = "cuda:0"

    cnt = 0
    sided_mano_model = HandWrapper(mano_model_folder).to(device)

    train_dataset = HandToObjectDataset(
        is_train=False,
        data_path=opt.traindata.data_path,
        window_size=opt.model.window,
        # single_demo="P0001_624f2ba9",
        # single_object="225397651484143",
        sampling_strategy="random",
        split=opt.traindata.trainsplit,
        split_seed=42,  # Ensure reproducible splits
        noise_scheme="syn",
        split_file=opt.traindata.split_file,
        **opt.datasets.augument,
        opt=opt,
        data_cfg=opt.traindata,
    )
    from jutils import hand_utils

    wrapper = hand_utils.ManopthWrapper()

    train_dataset.set_metadata()
    video_list = []
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=None, shuffle=False, num_workers=1
    )
    for b, batch in enumerate(dataloader):
        batch = model_utils.to_cuda(batch, device)

        wTo = batch["target_raw"]  # (1, T, D)
        hand_raw = batch["hand_raw"]  # (1, T, 2*D)

        print(
            f"condition shape (T, {21 * 3 * 2 + (3 + 3 + 15 + 10) * 2}) =? {batch['condition'].shape}"
        )
        print(
            f"cond_mask shape (T, {21 * 3 * 2 + (3 + 3 + 15 + 10) * 2}) =? {batch['cond_mask'].shape}"
        )
        print(
            f"target shape (T, {9 + (21 * 3 * 2 + (3 + 3 + 15 + 10) * 2)}) =? {batch['target'].shape}"
        )

        # left_hand_param, left_hand_shape = sided_mano_model.dict2para(
        #     batch["left_hand_params"], side="left"
        # )
        # right_hand_param, right_hand_shape = sided_mano_model.dict2para(
        #     batch["right_hand_params"], side="right"
        # )

        left_hand_verts, left_hand_faces, left_hand_joints = (
            sided_mano_model.hand_para2verts_faces_joints(
                batch["left_hand_params"].float(), side="left"
            )
        )
        right_hand_verts, right_hand_faces, right_hand_joints = (
            sided_mano_model.hand_para2verts_faces_joints(
                batch["right_hand_params"].float(), side="right"
            )
        )
        # left_hand, right_hand = torch.split(hand_raw, 21 * 3, dim=-1)
        # left_hand_verts, left_hand_faces = sided_mano_model.joint2verts_faces(left_hand)
        # right_hand_verts, right_hand_faces = sided_mano_model.joint2verts_faces(right_hand)
        left_hand_meshes = Meshes(verts=left_hand_verts, faces=left_hand_faces).to(
            device
        )
        left_hand_meshes.textures = mesh_utils.pad_texture(left_hand_meshes, "white")
        right_hand_meshes = Meshes(verts=right_hand_verts, faces=right_hand_faces).to(
            device
        )
        right_hand_meshes.textures = mesh_utils.pad_texture(right_hand_meshes, "blue")

        image_list = pt3d_viz.log_hoi_step(
            left_hand_meshes,
            right_hand_meshes,
            wTo,
            batch["newMesh"],
            pref="debug_vis_clip",
            contact=batch["contact"],
            save_to_file=False,
        )
        video_list.append(image_list)

        if b >= 10:
            break
    video_list = torch.cat(video_list, axis=0)
    image_utils.save_gif(
        video_list.unsqueeze(1),
        osp.join("outputs/debug_vis", opt.expname, f"video_{cnt}"),
        # f"outputs/debug_vis_{opt.coord}/video_{cnt}",
        fps=30,
        ext=".mp4",
    )
    print("saved video", f"outputs/debug_vis_clip/video_{cnt}.mp4")


def create_norm_starts():
    src_file = "data/cache/metadata_hot3d.pkl"

    dst_file = "data/cache/metadata_theta.pkl"

    with open(src_file, "rb") as f:
        metadata = pickle.load(f)

    obj_mean = metadata["object_mean"]
    obj_std = metadata["object_std"]

    left_hand_theta_mean = np.zeros([1, 3 + 3 + 15 + 10])
    left_hand_theta_std = np.ones([1, 3 + 3 + 15 + 10])
    left_hand_theta_mean[..., 3:6] = obj_mean[..., :3]
    left_hand_theta_std[..., 3:6] = obj_std[..., :3]

    right_hand_theta_mean = np.zeros([1, 3 + 3 + 15 + 10])
    right_hand_theta_std = np.ones([1, 3 + 3 + 15 + 10])

    right_hand_theta_mean[..., 3:6] = obj_mean[..., :3]
    right_hand_theta_std[..., 3:6] = obj_std[..., :3]

    metadata["left_hand_theta_mean"] = left_hand_theta_mean
    metadata["left_hand_theta_std"] = left_hand_theta_std
    metadata["right_hand_theta_mean"] = right_hand_theta_mean
    metadata["right_hand_theta_std"] = right_hand_theta_std

    with open(dst_file, "wb") as f:
        pickle.dump(metadata, f)
    print("saved metadata to", dst_file)


def check_nan(data_file="/move/u/yufeiy2/egorecon/data/cache/hotclip_train_camera.npz"):
    # data_list = np.load(data_file, allow_pickle=True)
    data_list = pickle.load(open(data_file, "rb"))
    for window in data_list:
        for k, v in window.items():
            if isinstance(v, np.ndarray):
                if np.isnan(v).any():
                    print(k, v[np.isnan(v)])
                    import ipdb; ipdb.set_trace()


# Global configuration variables
th = 0.05
use_cache = True  # True
save_cache = True  # True
# aug_world = False

if __name__ == "__main__":
    from jutils import plot_utils

    # create_mini_dataset()
    # vis_traj()
    # vis_clip()
    check_nan()

    # create_norm_starts()

    # compute_norm_stats()
