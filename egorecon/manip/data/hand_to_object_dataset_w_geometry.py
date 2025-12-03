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
from ...utils.motion_repr import cano_seq_mano
from ...utils.rotation_utils import (matrix_to_rotation_6d_numpy,
                                     quaternion_to_matrix_numpy,
                                     rotation_6d_to_matrix_numpy,
                                     mat_to_9d_numpy)
from ..lafan1.utils import rotate_at_frame_w_obj
from ...visualization.pt3d_visualizer import Pt3dVisualizer

from pytorch3d.ops import sample_points_from_meshes
from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform



def rotate(points, R):
    shape = list(points.shape)
    points = to_tensor(points)
    R = to_tensor(R)
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points.unsqueeze(dim=1)
    if R.shape[0] > shape[0]:
        shape[0] = R.shape[0]
    r_points = torch.matmul(points, R.transpose(1,2))
    return r_points.reshape(shape)

def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    # with open(path, "rb") as f:
    #     return pickle.load(f)
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            all_seq = pickle.load(f)

        all_processed_data = {}
        for seq, data in all_seq.items():
            processed_data = decode_npz(data)
            all_processed_data[seq] = processed_data

    elif path.endswith(".npz"):
        # Legacy support for mini dataset
        data = np.load(path, allow_pickle=True)
        seq_index = osp.basename(path).split(".")[0]

        processed_data = decode_npz(data)

        all_processed_data = {seq_index: processed_data}

    return all_processed_data


def decode_npz(data):
    # data = np.load(path, allow_pickle=True)
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
        if "cTo_shelf" in objects[uid]:
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
    return processed_data


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
        motion_threshold=0.05,  # Threshold for considering motion vs stationary
        sampling_strategy="balanced",  # 'balanced', 'motion_only', 'random'
        min_motion_frames=10,  # Minimum consecutive motion frames to consider
        augment=False,  # Whether in training mode (for data augmentation)
        split="train",  # 'train', 'val', or 'all'
        val_split_ratio=0.2,  # Fraction of windows to use for validation
        noise_std_obj_rot=0.0,
        noise_std_obj_trans=0.0,
        noise_std_mano_global_rot=0.0,
        noise_std_mano_body_rot=0.0,
        noise_std_mano_trans=0.0,
        noise_std_mano_betas=0.0,
        traj_std_obj_rot=0.0,
        traj_std_obj_trans=0.0,
        use_constant_noise=False,
        one_window=False,
        t0=300,
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
        self.one_window = one_window
        self.t0 = t0
        self.use_constant_noise = use_constant_noise
        self.noise_scheme = noise_scheme
        self.split_file = data_cfg.split_file
        self.opt = opt
        self.data_cfg = data_cfg

        self.bps_path = osp.join(opt.paths.data_dir, 'bps/bps.pt')
        # bps_dir = osp.join(opt.paths.data_dir, 'bps')

        # dest_obj_bps_npy_folder = os.path.join(bps_dir, "object_bps_npy_files_joints24")
        # dest_obj_bps_npy_folder_for_test = os.path.join(bps_dir, "object_bps_npy_files_for_eval_joints24")

        # if self.is_train:
        #     self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder 
        # else:
        #     self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test 
      
        self.prep_bps_data()

        self.mano_model_path = opt.paths.mano_dir
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

        self.processed_data = self._filter_data(
            single_demo, single_object
        )
        print(
            f"Filtered to demo: {single_demo} object: {single_object}, total: {len(self.processed_data)} demonstrations for overfitting"
        )

        # Create windows
        print(f"Creating windows for {self.split}...")
        # self.windows = self._create_windows_legecy()
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

        self.noise_std_params_dict = {
            "global_orient": noise_std_mano_global_rot,
            "transl": noise_std_mano_trans,
            "hand_pose": noise_std_mano_body_rot,
            "betas": noise_std_mano_betas,
            "object_rot": noise_std_obj_rot,
            "object_trans": noise_std_obj_trans,
            "traj_rot": traj_std_obj_rot,
            "traj_trans": traj_std_obj_trans,
        }

    def compute_object_geo_bps(self, obj_verts, obj_trans):
        # obj_verts: T X Nv X 3, obj_trans: T X 3
        bps_object_geo = self.bps_torch.encode(x=obj_verts, \
                    feature_type=['deltas'], \
                    custom_basis=self.obj_bps.repeat(obj_trans.shape[0], \
                    1, 1)+obj_trans[:, None, :])['deltas'] # T X N X 3 

        return bps_object_geo

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0 
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(1, -1, 3)
            
            bps = {
                'obj': bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            os.makedirs(osp.dirname(self.bps_path), exist_ok=True)
            torch.save(bps, self.bps_path)
        
        self.bps = torch.load(self.bps_path)
        self.bps_torch = bps_torch()

        self.obj_bps = self.bps['obj']

        # Load object geometry
        self.object_library = {}
        self.object_library_mesh = Pt3dVisualizer.setup_template(self.opt.paths.object_mesh_dir)
        for uid, mesh in self.object_library_mesh.items():
            self.object_library[uid] = sample_points_from_meshes(mesh, 50000)
        

    def _filter_data(self, single_demo=None, single_object=None):
        """Filter data for overfitting on specific demo/object."""
        filtered_data = {}
        if self.split == "mini":
            seq = list(self.processed_data.keys())[0]
            seq_list = [seq]
        else:
            seq_list = json.load(open(self.split_file))[self.split]
        
        for seq in seq_list:
            filtered_data[seq] = {}
            if single_object:
                # 225397651484143 : bowl
                obj_list = single_object.split("+")
            else:
                obj_list = list(self.processed_data[seq]["objects"].keys())

            if self.one_window:
                t0 = self.t0
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
        filtered_windows = [window for window in self.windows if window["is_motion"] > threshold]
        self.windows = filtered_windows
        print(f"{origin_size} -> {len(self.windows)} dynamic windows")

    def _create_windows(self):
        cache_name = osp.join(self.opt.paths.data_dir, 'cache', f"{self.data_cfg.name}_{self.split}_{self.noise_scheme}_{self.opt.coord}.npz")

        if osp.exists(cache_name):
            print('loading window cache from ', cache_name)
            self.windows = pickle.load(open(cache_name, 'rb'))
            return self.windows

        print('creating window cache and save to ', cache_name)
        windows = []
        for demo_id, demo_data in tqdm(self.processed_data.items(), desc="Creating windows"):
            for obj_id, obj_data in demo_data["objects"].items():
                obj_traj = obj_data['wTo']
                T = len(obj_traj)

                for start_idx in range(
                    0, T - self.window_size + 1, self.window_size // 4
                ):
                    end_idx = start_idx + self.window_size
                    if "wTo_valid" in obj_data:
                        obj_valid = obj_data["wTo_valid"][start_idx:end_idx]
                    else:
                        obj_valid = np.ones(self.window_size).astype(bool)
                        logging.warning(f"No object valid mask found for {demo_id}, {obj_id}")

                    if not np.all(obj_valid):
                        continue

                    left_wrist = demo_data["left_hand"]["theta"][start_idx:end_idx]
                    left_wrist_aa, left_wrist_pos, left_wrist_hA = np.split(left_wrist, [3, 6], axis=-1)
                    right_wrist = demo_data["right_hand"]["theta"][start_idx:end_idx]
                    right_wrist_aa, right_wrist_pos, right_wrist_hA = np.split(right_wrist, [3, 6], axis=-1)

                    object_pos = obj_traj[start_idx:end_idx][:, :3, 3]
                    object_rot_mat = obj_traj[start_idx:end_idx][:, :3, :3]
                    object_quat = R.from_matrix(object_rot_mat).as_quat()[:, [3, 0, 1, 2]]  # wxyz

                    if self.opt.coord == 'camera':
                        camera_pos = demo_data["wTc"][start_idx:end_idx][:, :3, 3]
                        camera_rot = demo_data["wTc"][start_idx:end_idx][:, :3, :3]
                        camera_quat = R.from_matrix(camera_rot).as_quat()[:, [3, 0, 1, 2]]  # wxyz
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
                    elif self.opt.coord == 'right_wrist':
                        # 1st frame of right wrist 
                        mano_params_dict = {
                            "global_orient": right_wrist_aa,
                            "transl": right_wrist_pos,
                            "betas": demo_data["right_hand"]["shape"][start_idx:end_idx],
                            "hand_pose": right_wrist_hA,
                        }
                        cano_right_positions, cano_right_mano_params_dict, canoTw = cano_seq_mano(
                            None, None, mano_params_dict, 
                            mano_model=self.sided_mano_models["right"],
                            device="cpu",
                            return_transf_mat=True,
                        )
                    else:
                        raise ValueError(f"Invalid coordinate system: {self.opt.coord}")

                    # canonicalize: lr-hand, object, camera

                    mano_params_dict = {
                        "global_orient": left_wrist_aa,
                        "transl": left_wrist_pos,
                        "betas": demo_data["left_hand"]["shape"][start_idx:end_idx],
                        "hand_pose": left_wrist_hA,
                    }

                    cano_left_positions, cano_left_mano_params_dict = cano_seq_mano(
                        canoTw=canoTw,
                        positions=None,
                        mano_params_dict=mano_params_dict,
                        mano_model=self.sided_mano_models["left"],
                        device="cpu",
                    )  # joints position

                    mano_params_dict = {
                        "global_orient": right_wrist_aa,
                        "transl": right_wrist_pos,
                        "betas": demo_data["right_hand"]["shape"][start_idx:end_idx],
                        "hand_pose": right_wrist_hA,
                    }

                    cano_right_positions, cano_right_mano_params_dict = cano_seq_mano(
                        canoTw=canoTw,
                        positions=None,
                        mano_params_dict=mano_params_dict,
                        mano_model=self.sided_mano_models["right"],
                        device="cpu",
                    )  # joints position


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
                        "left_hand_params": cano_left_mano_params_dict,
                        "right_hand_params": cano_right_mano_params_dict,
                        "left_hand_joints": cano_left_positions,
                        "right_hand_joints": cano_right_positions,
                        "contact": obj_data["contact_lr"][start_idx:end_idx],
                    }

                    is_motion = self._is_motion_window(cano_object_data)
                    window_data["is_motion"] = is_motion

                    if self.window_check(window_data):
                        windows.append(window_data)

        return windows

    def _create_windows_legecy(self):
        """Create sliding windows from trajectory data."""
        windows = []
        cache_name = osp.join(self.opt.paths.data_dir, 'cache', f"{self.data_cfg.name}_{self.split}_{self.noise_scheme}_{self.opt.coord}.pkl")
        if osp.exists(cache_name):
            print('loading window cache from ', cache_name)
            self.windows = pickle.load(open(cache_name, 'rb'))
            return self.windows

        print('creating window cache and save to ', cache_name)
        for demo_id, demo_data in tqdm(self.processed_data.items(), desc="Creating windows"):
            for obj_id, obj_data in demo_data["objects"].items():

                object_traj = obj_data["wTo"]  # (T, 4, 4)
                camera_traj = demo_data["wTc"]  # (T, 4, 4)

                # Find overlapping time range for all trajectories
                left_hand = demo_data["left_hand"]["theta"]
                right_hand = demo_data["right_hand"]["theta"]
                min_len = min(len(left_hand), len(right_hand), len(object_traj))

                if min_len < self.window_size:
                    print(
                        f"Warning: Trajectory too short for demo {demo_id}, obj {obj_id}: {min_len}"
                    )
                    continue

                # Trim trajectories to same length
                left_hand_trimmed = left_hand[:min_len]
                right_hand_trimmed = right_hand[:min_len]
                object_traj_trimmed = object_traj[:min_len]
                camera_traj_trimmed = camera_traj[:min_len]

                # Create sliding windows
                for start_idx in range(
                    0, min_len - self.window_size + 1, self.window_size // 2
                ):
                    end_idx = start_idx + self.window_size

                    if "wTo_valid" in demo_data["objects"][obj_id]:
                        object_valid = demo_data["objects"][obj_id]["wTo_valid"][
                            start_idx:end_idx
                        ]
                    else:
                        object_valid = np.ones(self.window_size).astype(bool)
                        print("Warning: No object valid mask found for", self.data_path)

                    if not np.all(object_valid):
                        # Skip windows with invalid object data
                        continue 
                    left_wrist_aa = left_hand_trimmed[start_idx:end_idx][:, 0:3]
                    left_wrist_pos = left_hand_trimmed[start_idx:end_idx][:, 3:6]

                    right_wrist_aa = right_hand_trimmed[start_idx:end_idx][:, 0:3]
                    right_wrist_pos = right_hand_trimmed[start_idx:end_idx][:, 3:6]

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

                    if self.noise_scheme == "real":

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

                    cano_object_pos = cano_object_pos[0]  # T X 3
                    cano_object_quat = cano_object_quat[0]  # T X 4

                    if self.noise_scheme == "real":
                        cano_object_shelf_pos = cano_object_shelf_pos[0]  # T X 3
                        cano_object_shelf_quat = cano_object_shelf_quat[0]  # T X 4

                    cano_camera_pos = cano_camera_pos[0, :, 0, :]  # T X 3
                    cano_camera_quat = cano_camera_quat[0, :, 0, :]  # T X 4

                    # Translate everything such that the camera initial position is at the origin.
                    camera2origin_trans = cano_camera_pos[0:1, :].copy()
                    canoTw = np.eye(4)
                    canoTw[:3, :3] = canoTw_rot
                    canoTw[:3, 3] = -camera2origin_trans

                    cano_camera_pos -= camera2origin_trans
                    cano_object_pos -= camera2origin_trans

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

                    wTo = object_traj_trimmed[start_idx:end_idx]  # (T, 4, 4)
                    wTo = canoTw[None] @ wTo  
                    cano_object_pos = wTo[:, :3, 3]
                    cano_object_rot_mat = wTo[:, :3, :3]
                    cano_object_quat = R.from_matrix(cano_object_rot_mat).as_quat()[:, [3, 0, 1, 2]]
                    cano_object_rot6d = matrix_to_rotation_6d_numpy(cano_object_rot_mat)

                    cano_camera_rot_mat = quaternion_to_matrix_numpy(cano_camera_quat)
                    cano_camera_rot6d = matrix_to_rotation_6d_numpy(cano_camera_rot_mat)

                    if self.noise_scheme == "real":
                        cano_object_shelf_rot_mat = quaternion_to_matrix_numpy(
                            cano_object_shelf_quat
                        )
                        cano_object_shelf_rot6d = matrix_to_rotation_6d_numpy(
                            cano_object_shelf_rot_mat
                        )
                        cano_object_shelf_data = np.concatenate(
                            (cano_object_shelf_pos, cano_object_shelf_rot6d), axis=-1
                        )

                    cano_object_data = np.concatenate(
                        (cano_object_pos, cano_object_rot6d), axis=-1
                    )
                    cano_camera_data = np.concatenate(
                        (cano_camera_pos, cano_camera_rot6d), axis=-1
                    )

                    window_data = {
                        "demo_id": demo_id,
                        "object_id": obj_id,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        # 'left_hand': cano_left_hand_data,
                        # 'right_hand': cano_right_hand_data,
                        # "shelf_valid": shelf_valid,
                        # "object_shelf": cano_object_shelf_data,
                        "object_valid": object_valid,
                        "object": cano_object_data,
                        "wTc": cano_camera_data,
                        "intr": demo_data["intrinsic"],
                        "left_hand": cano_left_positions.reshape(-1, 21 * 3),
                        "right_hand": cano_right_positions.reshape(-1, 21 * 3),
                        "left_hand_params": cano_left_mano_params_dict,
                        "right_hand_params": cano_right_mano_params_dict,
                        "left_hand_joints": cano_left_positions,
                        "right_hand_joints": cano_right_positions,
                    }

                    if self.noise_scheme == "real":
                        shelf_valid = demo_data["objects"][obj_id]["shelf_valid"][
                            start_idx:end_idx
                        ]
                        window_data["shelf_valid"] = shelf_valid
                        window_data["object_shelf"] = cano_object_shelf_data

                    # Check if this is a motion window
                    is_motion = self._is_motion_window(cano_object_data)
                    window_data["is_motion"] = is_motion
                    
                    # Calculate mean velocity for the object trajectory
                    if len(cano_object_data) > 1:
                        velocities = np.linalg.norm(np.diff(cano_object_data[:, :3], axis=0), axis=1)
                        mean_velocity = np.mean(velocities)
                    else:
                        mean_velocity = 0.0
                    window_data["mean_velocity"] = mean_velocity

                    if self.window_check(window_data):
                        windows.append(window_data)

        
        os.makedirs(osp.dirname(cache_name), exist_ok=True)
        pickle.dump(windows, open(cache_name, 'wb'))
        
        return windows

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
        keypoints_local = np.array([
            [0.0, 0.0, 0.0],  # origin
            [0.01, 0.0, 0.0], # x-axis
            [0.0, 0.01, 0.0], # y-axis
            [0.0, 0.0, 0.01]  # z-axis
        ])  # [4, 3]
        
        # Transform keypoints to world coordinates for all frames at once
        # keypoints_local: [4, 3], rot_matrices: [T, 3, 3], positions: [T, 3]
        # Use batch matrix multiplication: [T, 3, 3] @ [3, 4] -> [T, 3, 4]
        keypoints_world = np.einsum('tij,jk->tik', rot_matrices, keypoints_local.T)  # [T, 3, 4]
        # Transpose to [T, 4, 3] and add translation
        keypoints_world = keypoints_world.transpose(0, 2, 1) + positions[:, np.newaxis, :]  # [T, 4, 3]
        
        # Calculate accumulated motion for all keypoints at once
        # Calculate displacement between consecutive frames for all keypoints
        # keypoints_world: [T, 4, 3] -> diff: [T-1, 4, 3]
        displacements = np.linalg.norm(np.diff(keypoints_world, axis=0), axis=2)  # [T-1, 4]
        displacements = displacements.mean(axis=-1)
        
        # Sum displacements across time for each keypoint, then sum across all keypoints
        total_motion = np.sum(displacements)  # Scalar
        
        # Check if total motion exceeds threshold
        return total_motion

    def _batchify_is_motion_window(self, object_data, motion_threshold=0.05):
        """Batch version: Determine motion for multiple windows at once.
        :param object_data: [NUM_WINDOWS, T, 3+6] - position (3D) + rotation (6D) for multiple windows
        :param motion_threshold: float - threshold for motion detection
        :return: [NUM_WINDOWS] - boolean array indicating motion for each window
        """
        num_windows, T, D = object_data.shape
        
        # Extract position and rotation from trajectory
        positions = object_data[:, :, :3]  # [NUM_WINDOWS, T, 3]
        rotations = object_data[:, :, 3:]  # [NUM_WINDOWS, T, 6]
        
        # Convert 6D rotation to rotation matrices for all windows
        # Reshape to [NUM_WINDOWS * T, 6] for batch conversion
        rotations_flat = rotations.reshape(-1, 6)  # [NUM_WINDOWS * T, 6]
        rot_matrices_flat = rotation_6d_to_matrix_numpy(rotations_flat)  # [NUM_WINDOWS * T, 3, 3]
        rot_matrices = rot_matrices_flat.reshape(num_windows, T, 3, 3)  # [NUM_WINDOWS, T, 3, 3]
        
        # Define object keypoints: origin (0,0,0) and x,y,z axes with 0.01m length
        keypoints_local = np.array([
            [0.0, 0.0, 0.0],  # origin
            [0.01, 0.0, 0.0], # x-axis
            [0.0, 0.01, 0.0], # y-axis
            [0.0, 0.0, 0.01]  # z-axis
        ])  # [4, 3]
        
        # Transform keypoints to world coordinates for all windows and frames at once
        # keypoints_local: [4, 3], rot_matrices: [NUM_WINDOWS, T, 3, 3], positions: [NUM_WINDOWS, T, 3]
        # Use batch matrix multiplication: [NUM_WINDOWS, T, 3, 3] @ [3, 4] -> [NUM_WINDOWS, T, 3, 4]
        keypoints_world = np.einsum('ntij,jk->ntik', rot_matrices, keypoints_local.T)  # [NUM_WINDOWS, T, 3, 4]
        # Transpose to [NUM_WINDOWS, T, 4, 3] and add translation
        keypoints_world = keypoints_world.transpose(0, 1, 3, 2) + positions[:, :, np.newaxis, :]  # [NUM_WINDOWS, T, 4, 3]
        
        # Calculate accumulated motion for all windows and keypoints at once
        # Calculate displacement between consecutive frames for all keypoints
        # keypoints_world: [NUM_WINDOWS, T, 4, 3] -> diff: [NUM_WINDOWS, T-1, 4, 3]
        displacements = np.linalg.norm(np.diff(keypoints_world, axis=1), axis=3)  # [NUM_WINDOWS, T-1, 4]
        
        # Sum displacements across time and keypoints for each window
        total_motion = np.sum(displacements, axis=(1, 2))  # [NUM_WINDOWS]
        
        # Check if total motion exceeds threshold for each window
        is_motion = total_motion > motion_threshold  # [NUM_WINDOWS]
        
        return is_motion

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

    def set_metadata(self):
        """Compute normalization statistics for the dataset."""
        meta_file = self.data_cfg.meta_file
        if not osp.exists(meta_file):
            print(f'compute metadata....  and save to {meta_file}')
            # TODO: Implement compute_norm_stats function
            raise NotImplementedError("compute_norm_stats function not implemented")
        print(f'using loaded metadata from {meta_file}')
        self.stats = pickle.load(open(meta_file, 'rb'))    

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
            print("no stats!!! you'd better doing normailization right now!")
            return data

        mean = self.stats[f"{data_type}_mean"]
        std = self.stats[f"{data_type}_std"]

        return (data - mean) / std

    def denormalize_data(self, data, data_type):
        """Denormalize data using computed statistics."""
        if not self.stats:
            print("no stats!!! you'd better doing normailization right now!")
            return data

        mean = self.stats[f"{data_type}_mean"]
        std = self.stats[f"{data_type}_std"]

        return data * std + mean

    def __len__(self):
        if self.is_train:
            return max(100000, len(self.windows))
        else:
            return len(self.windows)

    def __getitem__(self, idx):
        idx = idx % len(self.windows)
        window = self.windows[idx]

        # # Add synthetic noise to the data
        # window_noisy = self.add_noise_data(window)
        
        # Add geometry and random canonical augmentation
        if aug_cano:
            window, newoTo = self.augment_cano_object(window)
            # window_noisy['object'] = self.transform_wTo_traj(window_noisy['object'], newoTo)

        if aug_world:
            print("TODO: augment world cooridate too!") # random rotation around gravity direction
            return 

        # Convert to tensors
        left_hand = to_tensor(window["left_hand"])  # [T, D]
        right_hand = to_tensor(window["right_hand"])  # [T, D]
        object_traj = to_tensor(window["object"])  # [T, D]

        print("TOOD: rewrite norm")

        # Normalize
        left_hand_norm = to_tensor(self.normalize_data(left_hand.numpy(), "left_hand"))
        right_hand_norm = to_tensor(
            self.normalize_data(right_hand.numpy(), "right_hand")
        )
        hand_norm = torch.cat([left_hand_norm, right_hand_norm], dim=-1)
        object_norm = to_tensor(self.normalize_data(object_traj.numpy(), "object"))

        # traj_noisy_norm = to_tensor(
        #     self.normalize_data(window_noisy["object"], "object")
        # )

        # Add noisy object trajectory to conditioning if specified
        # if self.opt.condition.noisy_obj:
        #     condition = torch.cat([condition, traj_noisy_norm], dim=-1)

        target = object_norm
        condition = torch.zeros([self.window_size, 0])
        if self.opt.hand == 'out':
            target = torch.cat([target, hand_norm], dim=-1)
        if self.opt.output.contact:
            target = torch.cat([target, window["contact"]], dim=-1)
        
        if self.opt.hand == 'cond':
            condition = torch.cat([condition, hand_norm], dim=-1)
        
        oMesh = self.object_library_mesh[window["object_id"]]
        newMesh = mesh_utils.apply_transform(oMesh, window["newTo"])    

        return {
            "condition": condition,  # [T, 2*D] - left and right hand trajectories
            "target": target,  # [T, D] - object trajectory to denoise
            
            # "traj_noisy_raw": to_tensor(window_noisy["object"]),
            "hand_raw": torch.cat(
                [left_hand, right_hand], dim=-1
            ),  # [T, 2*D] - left and right hand trajectories
            "left_hand_params": window["left_hand_params"],
            "right_hand_params": window["right_hand_params"],
            "target_raw": object_traj,  # [T, D] - unnormalized for evaluation
            "demo_id": window["demo_id"],
            "object_id": str(window["object_id"]),
            "is_motion": window["is_motion"],
            "object_valid": window["object_valid"],
            "intr":  to_tensor(window["intr"]),
            "wTc": to_tensor(window["wTc"]),
            "newTo": window["newTo"],
            "newPoints": window["newPoints"][0],
            "newMesh": newMesh,

            "start_idx": window["start_idx"],
            "end_idx": window["end_idx"],

            "contact": window["contact"],
        }
    
    def transform_wTo_traj(self, wTo, newTo):
        wTo = geom_utils.se3_to_matrix_v2(torch.FloatTensor(wTo)) # (T, 4, 4)

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
        if newTo is None:
            newTo = geom_utils.random_rotations(1, )  # (1, 3, 3)
            newTo = geom_utils.rt_to_homo(newTo) # (1, 4, 4)
        
        window['object'] = self.transform_wTo_traj(window['object'], newTo)

        # Load object geometry
        oPoints = self.object_library[window['object_id']]
        newPoints = mesh_utils.apply_transform(oPoints, newTo)  # (1, P, 3)
        
        # Randomly sample points for visualization
        nP = min(5000, newPoints.shape[1])
        index = torch.randint(0, newPoints.shape[1], (nP, ))
        newPoints = newPoints[:, index, :]
        
        window['newTo'] = newTo
        window['newPoints'] = newPoints
        
        return window, newTo


    def add_noise_data(self, can_window_dict):
        """Add noise to the data for training."""
        if self.noise_scheme == "real":
            can_window_dict_noisy = deepcopy(can_window_dict)
            can_window_dict_noisy["object"] = can_window_dict["object_shelf"]  # [T, D]
            # Apply mask
            mask = can_window_dict_noisy["shelf_valid"]  # [T, ]
            can_window_dict_noisy["object"] = (
                can_window_dict_noisy["object"] * mask[:, None]
            )
            return can_window_dict_noisy

        # Add noise to parameters

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

            # Add constant random translation noise to object trajectory
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
        save_dir="outputs/debug_vis",
        mano_models_dir="assets/mano",
        object_mesh_dir=opt.paths.object_mesh_dir,
    )

    mano_model_folder = "assets/mano"
    device = 'cuda:0'

    cnt = 0
    sided_mano_model = HandWrapper(mano_model_folder).to(device)

    train_dataset = HandToObjectDataset(
            is_train=False,
            data_path=opt.traindata.data_path,
            window_size=opt.model.window,
            # single_demo="P0001_624f2ba9",
            # single_object="225397651484143",
            sampling_strategy="random",
            split=opt.datasets.split,
            noise_scheme="syn",
            split_file=opt.traindata.split_file,
            **opt.datasets.augument,
            opt=opt,
            data_cfg=opt.traindata,
        )
    train_dataset.set_metadata()
    video_list = []
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=False, num_workers=1)
    for b, batch in enumerate(dataloader):
        batch = model_utils.to_cuda(batch, device)
        print('motion', batch['is_motion'])

        wTo = batch['target_raw']  # (1, T, D)
        hand_raw = batch['hand_raw']  # (1, T, 2*D)

        left_hand_param, left_hand_shape = sided_mano_model.dict2para(batch['left_hand_params'], side='left')
        right_hand_param, right_hand_shape = sided_mano_model.dict2para(batch['right_hand_params'], side='right')
        
        left_hand_verts, left_hand_faces, left_hand_joints = sided_mano_model.hand_para2verts_faces_joints(left_hand_param.float(), left_hand_shape.float(), side='left')
        right_hand_verts, right_hand_faces, right_hand_joints = sided_mano_model.hand_para2verts_faces_joints(right_hand_param.float(), right_hand_shape.float(), side='right')
        # left_hand, right_hand = torch.split(hand_raw, 21 * 3, dim=-1)
        # left_hand_verts, left_hand_faces = sided_mano_model.joint2verts_faces(left_hand)
        # right_hand_verts, right_hand_faces = sided_mano_model.joint2verts_faces(right_hand)
        left_hand_meshes = Meshes(verts=left_hand_verts, faces=left_hand_faces).to(device)
        left_hand_meshes.textures = mesh_utils.pad_texture(left_hand_meshes, 'white')
        right_hand_meshes = Meshes(verts=right_hand_verts, faces=right_hand_faces).to(device)
        right_hand_meshes.textures = mesh_utils.pad_texture(right_hand_meshes, 'blue')
        wTo_list = [wTo]
        color_list = ['red']
        newPoints = batch['newPoints'][None]
        print('newPoints shape', newPoints.shape)
        newPoints_mesh = plot_utils.pc_to_cubic_meshes(newPoints[:, :1000])
        image_list = pt3d_viz.log_training_step(left_hand_meshes, right_hand_meshes, wTo_list, color_list, newPoints_mesh, step=b, pref="debug_vis_clip", save_to_file=False)
        video_list.append(image_list)

        if b >= 10:
            break
    video_list = torch.cat(video_list, axis=0)
    image_utils.save_gif(video_list.unsqueeze(1), f"outputs/debug_vis_{opt.coord}/video_{cnt}", fps=30, ext=".mp4")
    print('saved video', f"outputs/debug_vis_clip/video_{cnt}.mp4")
        


# Global configuration variables
th = 0.05
use_cache = False
aug_cano = True

if __name__ == "__main__":
    from jutils import plot_utils
    
    # vis_traj()
    vis_clip()

    
    # compute_norm_stats()


