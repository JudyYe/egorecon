

# hand: {side}_theta: (3aa+3tsl+pcax15)
# hand: {side}_shape: (10,)

# [] estimate scale???
# [] save only center point in camera frame, with camera 
# [] FP: check and reinitialize. 
# [] handle missing frame 
import plotly.graph_objects as go
import subprocess
import json
import logging
from scipy.spatial.transform import Rotation as R
from hot3d.hot3d.data_loaders.loader_hand_poses import Handedness
import matplotlib
from hot3d.hot3d.data_loaders.mano_layer import loadManoHandModel
matplotlib.use("Agg")

# Configure logging to suppress info messages
logging.basicConfig(level=logging.WARNING)
from jutils import hand_utils
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict
from glob import glob

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import nvdiffrast.torch as dr
import PIL
import rerun as rr
import torch
import trimesh
from fire import Fire
from PIL import Image
from projectaria_tools.core import calibration
from projectaria_tools.core.sensor_data import TimeDomain  # @manual
from projectaria_tools.core.sensor_data import TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId  # @manual
from sam2.build_sam import build_sam2_video_predictor_hf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.append('thirdparty')
from FoundationPose.estimater import (FoundationPose, PoseRefinePredictor,
                                      ScorePredictor)
from FoundationPose.Utils import (draw_posed_3d_box,
                                  draw_xyz_axis)
from hot3d.hot3d.data_loaders.loader_object_library import load_object_library
from hot3d.hot3d.dataset_api import Hot3dDataProvider

device = "cuda:0"

root_dir = "hot3d/hot3d/dataset/"

def set_device():
    global device
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
    If conversion occurs, the returned mesh has only vertex and face data.
    Also handles texture extraction and ensures consistent mesh properties.
    """
    import trimesh

    if isinstance(scene_or_mesh, trimesh.Scene):
        # trimesh.Scene.dump() will apply all transforms and merge geometry
        mesh = trimesh.util.concatenate(
            [g for g in scene_or_mesh.dump(concatenate=False)]
        )
    else:
        mesh = scene_or_mesh

    # Ensure mesh has vertex normals
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.compute_vertex_normals()

    # Handle texture extraction for PBR materials
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
        material = mesh.visual.material
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            # Extract texture from PBR material
            try:
                texture_img = np.array(material.baseColorTexture.convert('RGB'))
                # Store texture for later use if needed
                mesh._texture_image = texture_img
            except Exception as e:
                print(f"Warning: Could not extract texture: {e}")
                mesh._texture_image = None
        elif hasattr(material, 'baseColorFactor'):
            # Use solid color from baseColorFactor
            color = material.baseColorFactor[:3]  # RGB
            mesh._texture_image = np.full((512, 512, 3), color, dtype=np.uint8)
        else:
            mesh._texture_image = None
    else:
        mesh._texture_image = None

    # Ensure vertex colors are available if possible
    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        # Create default vertex colors if none exist
        if hasattr(mesh, '_texture_image') and mesh._texture_image is not None:
            # Use average color from texture
            avg_color = np.mean(mesh._texture_image.reshape(-1, 3), axis=0)
            mesh.visual.vertex_colors = np.tile(avg_color, (len(mesh.vertices), 1))
        else:
            # Default gray color
            mesh.visual.vertex_colors = np.full((len(mesh.vertices), 3), [128, 128, 128], dtype=np.uint8)

    return mesh


class UniDepthWrapper():
    def __init__(self):
        version = "v2"
        backbone = "vitl14"
        self.depth_model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True)
        self.depth_model.eval()
        self.depth_model.to(device)

    def unproject(self, mask, intrinsic, depth):
        fx, fy, cx, cy = intrinsic
        intrinsic = torch.FloatTensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        depth = torch.from_numpy(depth).permute(2, 0, 1)

        # unproject depth map to points in camera frame
        
    def __call__(self, rgb, intrinsic):
        fx, fy, cx, cy = intrinsic
        intrinsic = torch.FloatTensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        # from unidepth.utils.camera import Pinhole
        # print(intrinsics.shape)
        rgb = torch.from_numpy(rgb/255.0).permute(2, 0, 1) # C, H, W

        # prediction = self.depth_model.infer(rgb, Pinhole(K=intrinsics))
        prediction = self.depth_model.infer(rgb, intrinsic)
        depth = prediction["depth"]

        depth = depth.squeeze().cpu().numpy().copy()
        return depth, prediction

        
class PoseWrapper:
    def __init__(
        self, mesh_file, debug=False, intrinsic=None, mask_list=None, bbox_list=None, depth_list=None
    ):
        if depth_model == 'metric3d':
            from Metric3D.metric3d_wrapper import Metric3D
            self.depth_model = Metric3D("Metric3D/weight/metric_depth_vit_large_800k.pth")
        elif depth_model == 'unidepth':
            self.depth_model = UniDepthWrapper()

        self.alpha = alpha
        mesh = as_mesh(trimesh.load(mesh_file))
        self.uid = mesh_file.split('/')[-1].split('.')[0]

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.pose_model = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir="outputs/",
            debug=debug,
            glctx=glctx,
        )
        self.reset_posetracker(mesh_file, mask_list, bbox_list)

        self.need_init = True
        self.intrinsic = intrinsic
        self.depth_list = depth_list  # depth list

    def reset_posetracker(self, mesh_file=None, mask_list=None, bbox_list=None):
        if mesh_file is not None:
            print('reset_posetracker', mesh_file)
            mesh = as_mesh(trimesh.load(mesh_file))
        else:
            mesh = self.mesh
        self.pose_model.reset_object(mesh.vertices, mesh.vertex_normals, mesh=mesh)
        self.mask_list = mask_list
        self.bbox_list = bbox_list
        self.need_init = True

        self.mesh = mesh
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        self.mask_list = mask_list  # mask list
        self.to_origin = to_origin
        self.bbox = bbox

    def read_image(self, rgb):
        if isinstance(rgb, str):
            rgb = cv2.imread(rgb)[..., ::-1].copy()
        elif isinstance(rgb, PIL.Image.Image):
            rgb = np.array(rgb)
        else:
            rgb = rgb
        return rgb

    def predict_depth(self, rgb, intrinsic=None, alpha=1.0):
        if self.depth_list is not None:
            depth = self.depth_list[self.index]
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
            zfar = np.inf
            depth[(depth<0.001) | (depth>=zfar)] = 0    
            return self.depth_list[self.index], intrinsic
        rgb = self.read_image(rgb)
        H, W = rgb.shape[:2]

        if intrinsic is None:
            intrinsic = self.intrinsic
        pred_depth, prediction = self.depth_model(rgb, intrinsic)
        
        pred_depth = cv2.resize(pred_depth, (W, H))
        pred_depth = pred_depth * alpha

        return pred_depth, intrinsic, prediction

    def track_one(self, rgb, intrinsic=None):
        depth, _, _ = self.predict_depth(rgb, intrinsic, self.alpha)
        if depth_zero:
            depth = np.zeros_like(depth)
        fx, fy, cx, cy = intrinsic
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.reshape(3, 3)

        pose = self.pose_model.track_one(rgb=rgb.copy(), depth=depth, K=K, iteration=2)
        return pose

    def register_one(self, rgb, intrinsic=None, obj_mask=None, ):
        """
        :param rgb: (H, W, 3) numpy
        :param intrinsic: _description_, defaults to None
        :param obj_mask: (H, W)
        :return: _description_
        """
        if self.process_check_mask(obj_mask) is False:
            return None
        # sam mask
        depth, intrinsic, _ = self.predict_depth(rgb, intrinsic, self.alpha)
        # depth: (H, W) numpy
        fx, fy, cx, cy = intrinsic
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        downsample = 4
        K[0:2, :] /= downsample
        rgb = cv2.resize(rgb, (rgb.shape[1]//downsample, rgb.shape[0]//downsample))
        depth = cv2.resize(depth, (depth.shape[1]//downsample, depth.shape[0]//downsample))
        
        obj_mask = obj_mask.astype(np.uint8) * 255
        obj_mask = cv2.resize(obj_mask, (obj_mask.shape[1]//downsample, obj_mask.shape[0]//downsample))

        obj_mask = obj_mask.astype(bool)
        pose = self.pose_model.register(
            K=K, rgb=rgb.copy(), depth=depth, ob_mask=obj_mask, iteration=5, zero_depth=zero_depth
        )
        return pose

    @staticmethod
    def process_check_mask(mask):
        # if mask is at the edge of the image (suggest truncation), return False
        # otherwise, erode it and return the eroded mask
        mask = mask.astype(np.uint8) * 255
        H, W = mask.shape[:2]
        if mask[0, :].sum() > 0 or mask[-1, :].sum() > 0 or mask[:, 0].sum() > 0 or mask[:, -1].sum() > 0:
            return False
        if mask.sum() < 10:  # too small
            return False
        mask = cv2.erode(mask, np.ones((5, 5)), iterations=1)
        mask = mask.astype(bool)
        if mask.sum() < 10:  # too small
            return False
        return mask

    def estimate_alpha(self, image_list, intrinsic):
        # randomly select K frames.
        K = 20
        inds = np.random.choice(len(self.mask_list), K)
        scales = []

        for ind in inds:
            rgb = self.read_image(image_list[ind])
            _, _, prediction = self.predict_depth(rgb, intrinsic, 1)

            mask = self.process_check_mask(self.mask_list[ind][0])
            if mask is False:
                print('object is not detected / truncated in frame', ind)
                continue
            mask = torch.from_numpy(mask).to(device)[None, None]  # (1, 1, H, W)
            mask_exp = mask.repeat(1, 3, 1, 1)
            cPoints = prediction['points']  # (1, 3, H, W)
            cPoints = cPoints[mask_exp] 
            cPoints = cPoints.detach().cpu().numpy()  # (P, 3)
            cPoints = cPoints.reshape(3, -1)
            cPoints = cPoints.T

            # Use the new robust scaling method
            template_points = self.mesh.vertices
            if cPoints.shape[0] == 0:
                print('whyw?', mask.sum(), cPoints.shape)
            print('cPoints', cPoints.shape, template_points.shape)

            scale = compute_alignment_scale(cPoints, template_points, method='auto')
            scales.append(scale)
        if len(scales) == 0:
            print('object is not detected in any frame', inds, len(scales))
            self.alpha = 1.0
        else:    
            self.alpha = np.mean(scales)
        return scales

    def track_video(self, image_list, intrinsic):
        # self.estimate_alpha(image_list, intrinsic)
        for i, img in enumerate(image_list):
            self.index = i
            img = self.read_image(img)
            if self.need_init:
                pose = self.register_one(img, intrinsic, self.mask_list[i][0].copy())
                fname = f'outputs/register_{i}_{self.uid}'
                imageio.imwrite(fname + '_mask.png', cv2.resize(self.mask_list[i][0].copy() * 255, (256, 256)))
                imageio.imwrite(fname + '_image.png', cv2.resize(img, (256, 256)))
                self.need_init = False
            else:
                pose = self.track_one(img, intrinsic)
            
            if pose is not None:
                valid = self.check_consistency(pose, self.bbox_list[i], intrinsic)
            else:
                valid = False
            print('valid', valid)
            self.need_init = not valid
            yield pose, valid
        return
    
    def check_consistency(self, pose, bbox_2d, intrinsic):
        # project center of bbox to 2D, and compare with bbox_2d
        K = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
        print('pose', pose.shape, self.to_origin.shape)
        center_pose = pose@np.linalg.inv(self.to_origin)  # cTo
        center_3d = center_pose[:3, 3]

        center_2d = K@center_3d
        center_2d = center_2d[:2] / center_2d[2]
        
        # if inside of bbox_2d, return True
        x1, y1, x2, y2 = bbox_2d
        inside = x1 <= center_2d[0] <= x2 and y1 <= center_2d[1] <= y2
        return inside
        


class HOT3DLoader:
    def __init__(
        self,
        seq_path,
        object_library_folder,
        # mano_hand_model=None,
        fail_on_missing_data=False,
    ):
        mano_hand_model = loadManoHandModel(mano_model_folder)
        object_library = load_object_library(
            object_library_folderpath=object_library_folder
        )
        self.data_provider = hot3d_data_provider = Hot3dDataProvider(
            sequence_folder=seq_path,
            object_library=object_library,
            mano_hand_model=mano_hand_model,
            fail_on_missing_data=fail_on_missing_data,
        )
        self.image_stream_id = StreamId("214-1")

        self._object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
        self._object_box2d_data_provider = (
            hot3d_data_provider.object_box2d_data_provider
        )
        # Object library
        self._object_library = hot3d_data_provider.object_library
        self._device_data_provider = hot3d_data_provider.device_data_provider
        self._device_pose_provider = hot3d_data_provider.device_pose_data_provider
        self._hand_pose_data_provider = hot3d_data_provider.mano_hand_data_provider


    def get_intr(self, rgb_camera_calibration, down_sampling_factor=1):
        calib = self.get_calibration(rgb_camera_calibration)
        intr = np.array(
            [
                [calib.get_focal_lengths()[0], 0, calib.get_principal_point()[0]],
                [0, calib.get_focal_lengths()[1], calib.get_principal_point()[1]],
                [0, 0, 1],
            ]
        )
        intr[0] /= down_sampling_factor
        intr[1] /= down_sampling_factor
        return intr


    def get_calibration(
        self,
        rgb_camera_calibration,
        should_rectify_image=True,
        should_rotate_image=True,
    ):
        """
        get intrinsics of pinhole camera, (and T_device_camera from get_transform_device_camera())

        devcie_points -> pixels: get_camera_projection_from_device_point(p, camera_calibration)
        :param rgb_camera_calibration: _description_
        :param should_rectify_image: _description_, defaults to True
        :param should_rotate_image: _description_, defaults to True
        :raises NotImplementedError: _description_
        :return: _description_
        """
        if should_rectify_image:
            rgb_linear_camera_calibration = calibration.get_linear_camera_calibration(
                int(rgb_camera_calibration.get_image_size()[0]),
                int(rgb_camera_calibration.get_image_size()[1]),
                rgb_camera_calibration.get_focal_lengths()[0],
                "pinhole",
                rgb_camera_calibration.get_transform_device_camera(),
            )
            if should_rotate_image:
                rgb_rotated_linear_camera_calibration = (
                    calibration.rotate_camera_calib_cw90deg(
                        rgb_linear_camera_calibration
                    )
                )
                camera_calibration = rgb_rotated_linear_camera_calibration
            else:
                camera_calibration = rgb_linear_camera_calibration
        else:  # No rectification
            if should_rotate_image:
                raise NotImplementedError(
                    "Showing upright-rotated image without rectification is not currently supported.\n"
                    "Please use --no_rotate_image_upright and --no_rectify_image together."
                )
            else:
                camera_calibration = rgb_camera_calibration
        return camera_calibration


    def extract_all(self, num=-1, decode_image=True, ):
        # save T_w_o, T_w_c
        # get intrinsics
        _, rgb_calibration = self._device_data_provider.get_camera_calibration(
            self.image_stream_id
        )
        intrinsics = self.get_calibration(rgb_calibration)
        
        deviceTcamera = intrinsics.get_transform_device_camera()
        focal_length = float(intrinsics.get_focal_lengths()[0])
        cx, cy = intrinsics.get_principal_point()
        W, H = intrinsics.get_image_size()
        focal = [focal_length, focal_length, cx, cy]
        # save all images
        to_save = defaultdict(list)

        timestamps = self.data_provider.device_data_provider.get_sequence_timestamps()
        for timestamp_ns in tqdm(timestamps, desc="extract all"):
            if num > 0 and len(to_save["image"]) >= num:
                break
            if decode_image:
                image_data = self._device_data_provider.get_undistorted_image(
                    timestamp_ns, self.image_stream_id
                )

                image = np.rot90(image_data, k=3)
                H, W = image.shape[:2]

                to_save["image"].append(image)
            box2d_collection_with_dt = \
                self._object_box2d_data_provider.get_bbox_at_timestamp(
                    stream_id=self.image_stream_id,
                    timestamp_ns=timestamp_ns,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                )
            box2d_collection = box2d_collection_with_dt.box2d_collection
            # tolerate 100ms: 0.1s
            dt = box2d_collection_with_dt.time_delta_ns

            def box2d_to_xyxy(box2d):
                if box2d is None or dt > 0.1 * 1e9:
                    return None
                x1, y1, x2, y2 = [box2d.left, box2d.top, box2d.right, box2d.bottom]

                x1, y1, x2, y2 = y1, x1, y2, x2
                x1 = W - x1
                x2 = W - x2
                x1, x2 = x2, x1
                return [x1, y1, x2, y2]
            xyxy = {
                uid: box2d_to_xyxy(box2d.box2d)
                for uid, box2d in box2d_collection.box2ds.items()
            }
            to_save["box_xyxy"].append(xyxy)

            # T_w_c: T_world_device
            T_w_c = self._device_pose_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            T_w_device = T_w_c.pose3d.T_world_device.to_matrix()
            T_w_c = T_w_device @ deviceTcamera.to_matrix()
            to_save["T_w_c"].append(T_w_c)

            T_w_o_collection = self._object_pose_data_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            T_w_o_collection = T_w_o_collection.pose3d_collection.poses  # dict of uid to T_world_object
            T_w_o = {
                uid: T_w_o_collection[uid].T_world_object.to_matrix()
                for uid in T_w_o_collection.keys()
            }

            to_save["T_w_o"].append(T_w_o)

            # hand pose
            # left hand 
            hand_pose_collection = self._hand_pose_data_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            ).pose3d_collection
            for handedness, hand_pose_data in hand_pose_collection.poses.items():
                if handedness == Handedness.Left:
                    key = 'left_hand'
                elif handedness == Handedness.Right:
                    key = 'right_hand'
                
                mano_shape = self._hand_pose_data_provider._mano_shape_params
                hA = hand_pose_data.joint_angles
                wrist = hand_pose_data.wrist_pose.to_matrix()
                global_aa = R.from_matrix(wrist[:3, :3]).as_rotvec()
                global_t = wrist[:3, 3]
                hand_vec = np.concatenate([global_aa, global_t, hA], axis=-1)

                to_save[f'{key}_theta'].append(hand_vec)  # R, T, hA
                to_save[f'{key}_shape'].append(mano_shape.cpu().numpy())

        # save a intrincis
        to_save["intrinsic"] = focal
        return to_save


@torch.no_grad()
def get_all_masks(seq="P0001_624f2ba9", max_T=240, num=-1, **kwargs):
    # set_device()
    video_dir = osp.join(root_dir, "pred_pose", seq, "images")
    frame_names = sorted(glob(osp.join(video_dir, "*.jpg")))
    bbox_list = pickle.load(
        open(osp.join(root_dir, "pred_pose", seq, "box_xyxy.pkl"), "rb")
    )
    if num > 0:
        frame_names = frame_names[:num]
        
    # total_frames = len(frame_names)
    # print(f"Total frames: {total_frames}, max_T: {max_T}")
    
    # # Initialize SAM2 predictor once (this takes time)
    # print("Initializing SAM2 predictor...")
    # predictor = build_sam2_video_predictor_hf('facebook/sam2-hiera-large', device=device)
    # inference_state = predictor.init_state(
    #     video_path=video_dir, 
    #     async_loading_frames=True, 
    #     offload_video_to_cpu=True, 
    #     offload_state_to_cpu=True
    # )
    # print("SAM2 predictor initialized successfully!")
    
    # # If video is shorter than max_T, process normally
    # if total_frames <= max_T:
    #     video_segments = process_video_chunk(
    #         predictor, inference_state, video_dir, bbox_list, frame_names, start_frame=0, end_frame=total_frames
    #     )
    # else:
    #     # Split video into chunks and process each separately
    #     video_segments = {}
    #     chunk_idx = 0
        
    #     for start_frame in range(0, total_frames, max_T):
    #         end_frame = min(start_frame + max_T, total_frames)
    #         print(f"Processing chunk {chunk_idx}: frames {start_frame} to {end_frame-1}")
            
    #         chunk_segments = process_video_chunk(
    #             predictor, inference_state, video_dir, bbox_list, frame_names, start_frame, end_frame
    #         )
            
    #         # Merge chunk results into main video_segments
    #         for frame_idx, frame_data in chunk_segments.items():
    #             video_segments[frame_idx] = frame_data
            
    #         chunk_idx += 1
    
    # print(f"Final video_segments keys: {sorted(video_segments.keys())}")
    
    # to_save = segments2save(video_segments, bbox_list)
    fname = osp.join(root_dir, "pred_pose", seq, "video_segments.pkl")

    # with open(fname, "wb") as f:
    #     pickle.dump(to_save, f)

    with open(fname, "rb") as f:
        video_segments = pickle.load(f)
        
    save_dir = osp.join(root_dir, "pred_pose", seq, "masks")
    vis_sam2(frame_names, video_segments, bbox_list, save_dir)


def process_video_chunk(predictor, inference_state, video_dir, bbox_list, frame_names, start_frame, end_frame):
    """Process a chunk of the video with SAM2 using pre-initialized predictor"""
    print(f"Processing chunk: frames {start_frame} to {end_frame-1}")
    
    # Initialize state for this chunk (much faster than initializing predictor)
    predictor.reset_state(inference_state)

    # Add bounding boxes for this chunk
    for i in range(start_frame, end_frame):
        if i >= len(bbox_list):
            break
            
        bboxes = bbox_list[i]
        for uid, bbox in bboxes.items():
            if bbox is None:
                print(f'bbox is None for frame {i} and object {uid} {uid2name[int(uid)]}')
                continue
            if i <= 30:  # Skip first 30 frames of each video
                continue

            box = np.array(bbox, dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=i,  # Relative frame index within chunk
                obj_id=int(uid),
                box=box,
            )
    
    # Run propagation for this chunk
    video_segments = {}
    T = end_frame - start_frame
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=start_frame, max_frame_num_to_track=T,
    ):
        # Convert relative frame index back to absolute frame index
        absolute_frame_idx = out_frame_idx
        print(f'Processing frame {absolute_frame_idx} (chunk {start_frame} to {end_frame-1})')
        
        video_segments[absolute_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    print(f"Chunk processed: {len(video_segments)} frames")
    return video_segments
                
            

def segments2save(video_segments, bbox_list):
    to_save = {}
    # change to {'uid': {frame_idx: [ ,], mask: [ ,]}}
    for out_frame_idx, out_obj_ids in video_segments.items():
        for out_obj_id, out_mask_logits in out_obj_ids.items():
            if out_obj_id not in to_save:
                to_save[out_obj_id] = defaultdict(list)
            to_save[out_obj_id]["frame_idx"].append(out_frame_idx)
            to_save[out_obj_id]["mask"].append(out_mask_logits)
            to_save[out_obj_id]["bbox"].append(
                bbox_list[out_frame_idx][str(out_obj_id)]
            )
            valid_box = bbox_list[out_frame_idx][str(out_obj_id)] is not None
            valid_mask = out_mask_logits.sum() > 0
            to_save[out_obj_id]["valid_box"].append(valid_box)
            to_save[out_obj_id]["valid_mask"].append(valid_mask)

    return to_save


def vis_sam2(frame_names, save, bbox_list, save_dir):
    vis_frame_stride = 1
    os.makedirs(save_dir, exist_ok=True)
    cmd = 'rm -rf ' + save_dir + '/*.png'
    os.system(cmd)
    for i, out_frame_idx in enumerate(range(0, len(frame_names), vis_frame_stride)):
        plt.close("all")
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        image = Image.open(osp.join(f"{frame_names[out_frame_idx]}"))
        print(f'vis frame {out_frame_idx}', 'image', frame_names[out_frame_idx])
        # import pdb; pdb.set_trace()
        W, H = image.size
        plt.imshow(image)
        for out_obj_id, preds in save.items():
            show_mask(preds['mask'][out_frame_idx], plt.gca(), obj_id=out_obj_id, random_color=False)
        # if video_segments is not None and out_frame_idx in video_segments:
        #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id, random_color=True)
    
        for out_obj_id, _ in bbox_list[out_frame_idx].items():
            if bbox_list[out_frame_idx][str(out_obj_id)] is None:
                print(f'bbox is None for frame {out_frame_idx} and object {out_obj_id} {uid2name[int(out_obj_id)]}')
                continue
            print(f'bbox {out_obj_id} {uid2name[int(out_obj_id)]} {bbox_list[out_frame_idx][str(out_obj_id)]}')
            x1, y1, x2, y2 = bbox_list[out_frame_idx][str(out_obj_id)]
            show_box([x1, y1, x2, y2], plt.gca())

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(osp.join(save_dir, f"{out_frame_idx:05d}.png"))
    
    fname = osp.join(save_dir, "../masks.mp4")
    # Use pattern matching for non-consecutive frame numbers
    cmd = f'ffmpeg -pattern_type glob -i "{save_dir}/*.png" -c:v libx264 -pix_fmt yuv420p {fname} -y'
    print(f"Creating masks video: {cmd}")
    os.system(cmd)
    print(f"Saved video to {fname}")
    
    

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx % 10)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def batch_save_one(decode_image=False):

    return 
def save_one(seq="P0001_624f2ba9", num=-1, decode_image=True):
    root_data = root_dir
    seq_dir = osp.join(root_data, seq)
    object_library_folder = osp.join(root_data, "assets/")
    loader = HOT3DLoader(
        seq_dir,
        object_library_folder,
    )
    to_save = loader.extract_all(num=num, decode_image=decode_image)
    save_dir = osp.join(root_data, "pred_pose", seq)

    if 'image' in to_save:
        os.makedirs(osp.join(save_dir, "images"), exist_ok=True)
        print("save dir", save_dir, osp.abspath(save_dir))
        if num > 0:
            to_save["image"] = to_save["image"][:num]
        for i in tqdm(range(len(to_save["image"]))):
            cv2.imwrite(f"{save_dir}/images/{i:05d}.jpg", to_save["image"][i][..., ::-1])
        to_save.pop("image")

    # save intrincis
    with open(f"{save_dir}/intrinsic.pkl", "wb") as f:
        pickle.dump(to_save["intrinsic"], f)

    to_save.pop("intrinsic")

    # save box_xyxy
    with open(f"{save_dir}/box_xyxy.pkl", "wb") as f:
        pickle.dump(to_save["box_xyxy"], f)
    to_save.pop("box_xyxy")
    
    # save the rest: T_w_c, T_w_o
    with open(f"{save_dir}/meta.pkl", "wb") as f:
        pickle.dump(to_save,f)


@torch.no_grad()
def esitimate_alpha(seq="P0001_624f2ba9", pose_wrapper=None):
    np.random.seed(123)
    # for each frame 
    video_segments = pickle.load(
        open(osp.join(root_dir, "pred_pose", seq, "video_segments.pkl"), "rb")
    )
    uid_list = list(video_segments.keys())

    # for each obect, get the estimation alpha --> get median
    alpha_list = []
    for uid in uid_list:
        mesh_file = osp.join(root_dir, "assets", f"{uid}.glb")
        intrinsic = pickle.load(
            open(osp.join(root_dir, "pred_pose", seq, "intrinsic.pkl"), "rb")
        )
        bbox_list = video_segments[uid]["bbox"]
        mask_list = video_segments[uid]["mask"]
        image_list = sorted(glob(osp.join(root_dir, "pred_pose", seq, "images", "*.jpg")))

        if pose_wrapper is None:
            pose_wrapper = PoseWrapper(
                mesh_file=mesh_file,
                debug=False,
                intrinsic=intrinsic,
                mask_list=mask_list,
                bbox_list=bbox_list,
                depth_list=None,
            )
        else:
            pose_wrapper.reset_posetracker(mesh_file, mask_list)
        
        scales = pose_wrapper.estimate_alpha(image_list, intrinsic)
        if len(scales) > 0:        
            alpha_list.append(np.median(scales))

        print('uid', uid, uid2name[int(uid)], 'alpha', scales)
    
    print(f"alpha: {np.median(alpha_list)}", alpha_list)
    

    pose_wrapper.alpha = np.median(alpha_list)
    for uid in uid_list:
        mesh_file = osp.join(root_dir, "assets", f"{uid}.glb")
        mask_list = video_segments[uid]["mask"]
        bbox_list = video_segments[uid]["bbox"]
        pose_wrapper.reset_posetracker(mesh_file, mask_list)

        for i, img in enumerate(image_list):
            img = pose_wrapper.read_image(img)
            mask = pose_wrapper.process_check_mask(mask_list[i][0])
            if mask is False:
                continue

            pose = pose_wrapper.register_one(img, intrinsic, mask_list[i][0].copy())
            if pose is None:    
                continue
            center_pose = pose@np.linalg.inv(pose_wrapper.to_origin)
            save_index = seq
            bbox = pose_wrapper.bbox
            fx, fy, cx, cy = intrinsic
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K = K.reshape(3, 3)

            color = imageio.imread(image_list[i])

            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)

            os.makedirs(f'outputs/{save_index}/{uid2name[uid]}', exist_ok=True)
            imageio.imwrite(f'outputs/{save_index}/{uid2name[uid]}/{i:05d}.png', vis)

            # save mask and 2D bbox too
            plt.close("all")
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {i}")
            plt.imshow(color)
            show_mask(mask_list[i][0], plt.gca())
            show_box(bbox_list[i], plt.gca())
            plt.savefig(f'outputs/{save_index}/{uid2name[uid]}/{i:05d}_mask.png')
            
            break

    return np.median(alpha_list), pose_wrapper

def vis_meta(seq="P0001_624f2ba9", num=-1, pred='fp', **kwargs):
    image_list = sorted(glob(osp.join(root_dir, "pred_pose", seq, "images", "*.jpg")))
    image_list = image_list[:num]
    # image_list = [imageio.imread(image) for image in image_list]
    # mp4_file = 'outputs/vis_meta_image.mp4'
    # imageio.mimsave(mp4_file, image_list)

    
    meta_file = osp.join(root_dir, "pred_pose", seq, "meta.pkl")
    intrinsic_file = osp.join(root_dir, "pred_pose", seq, "intrinsic.pkl")
    meta = pickle.load(open(meta_file, "rb"))
    intrinsic = pickle.load(open(intrinsic_file, "rb"))

    if pred == 'fp':
        pred_list = glob(osp.join('outputs/', seq, "*/pose.pkl"))
        name2pred = {}
        for pred_file in pred_list:
            with open(pred_file, 'rb') as f:
                data = pickle.load(f)
            name2pred[pred_file.split('/')[-2]] = data
    elif pred == '3d':
        pred_file = osp.join(root_dir, "pred_pose", seq, "pose_3d.pkl")
        with open(pred_file, 'rb') as f:
            save_dict = pickle.load(f)  # {uid: {"cTo": [...], "valid": [...]}}
        name2pred = {uid2name[int(uid)]: save_dict[uid] for uid in save_dict.keys()}
    else:
        raise ValueError(f"Unknown pred: {pred}")
    
    T_w_o = meta['T_w_o']
    T_w_c = meta['T_w_c']
    
    fx, fy, cx, cy = intrinsic
    W, H = cx * 2, cy * 2
    rr.init("vis_meta")
    rr.save(f"outputs/vis_meta_{pred}.rrd")
    # log_list_of_camera_poses(T_w_c[:num], W, H, fx, name='world/camera')

    sided_wrapper = {}
    for side in ['left', 'right']:
        sided_wrapper[side] = hand_utils.ManopthWrapper(mano_model_folder, side=side, )
    
    if num < 0:
        num = len(T_w_o)
    # fig = go.Figure()
    # # draw position for each object
    # for name in name2pred:
    #     cTo = name2pred[name]['cTo']
    #     valid = np.array(name2pred[name]['valid'])[:num]
    #     wTc = np.array(T_w_c)[:num]
    #     wTo = wTc @ np.array(cTo)[:num]
    #     wTo_pos = wTo[..., :3, 3]
    #     wTo_pos = wTo_pos * valid[..., None]
    #     fig.add_trace(go.Scatter3d(
    #         x=wTo_pos[:, 0],
    #         y=wTo_pos[:, 1],
    #         z=wTo_pos[:, 2],
    #         mode='markers',
    #         name=f'{name}_pred',
    #     ))
    #     # find gt
    #     wTo_gt = []
    #     name2uid = {v: k for k, v in uid2name.items()}
    #     for t in range(num):
    #         wTo_gt.append(T_w_o[t][str(name2uid[name])])
    #     wTo_gt = np.array(wTo_gt)
    #     wTo_gt_pos = wTo_gt[..., :3, 3]
    #     print(wTo_gt_pos.shape)
    #     fig.add_trace(go.Scatter3d(
    #         x=wTo_gt_pos[:, 0],
    #         y=wTo_gt_pos[:, 1],
    #         z=wTo_gt_pos[:, 2],
    #         mode='markers',
    #         name=f'{name}_gt',
    #     ))
    # fig.write_html('outputs/vis_meta.html')
    
    for t in range(num):
        rr.set_time_sequence("frame", t)
        pose = T_w_c[t]
        focal_length = intrinsic[0]
        if t == 0:
            rr.log("world/camera", rr.Pinhole(
                width=W,
                height=H,
                focal_length=float(focal_length),
            ))
        rr.log("world/camera", rr.Transform3D(
            translation=pose[:3, 3],
            rotation=rr.Quaternion(xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat())
        ))        
        for uid in T_w_o[t].keys():
            mesh_file = osp.join(root_dir, "assets", f"{uid}.glb")

            if t == 0:
                rr.log(f"world/object_pose/{uid}", rr.Asset3D(
                    path=mesh_file,
                ))
            rr.log(f"world/object_pose/{uid}", rr.Transform3D(
                translation=T_w_o[t][uid][:3, 3],
                rotation=rr.Quaternion(xyzw=Rotation.from_matrix(T_w_o[t][uid][:3, :3]).as_quat())
            ))
            if uid2name[int(uid)] in name2pred:
                if not name2pred[uid2name[int(uid)]]['valid'][t]:
                    pass
                cTo = name2pred[uid2name[int(uid)]]['cTo'][t]
                # wTc = np.linalg.inv(T_w_c[t])
                wTc = T_w_c[t]
                wTo = wTc @ cTo
                if t == 0:
                    rr.log(f"world/object_pose_pred/{uid}", rr.Asset3D(
                        path=mesh_file,
                    ))
                rr.log(f"world/object_pose_pred/{uid}", rr.Transform3D(
                    translation=wTo[:3, 3],
                    rotation=rr.Quaternion(xyzw=Rotation.from_matrix(wTo[:3, :3]).as_quat())
                ))
        
        # hand pose
        for handedness in ['left', 'right']:
            hand_pose = meta[f'{handedness}_hand_theta'][t]
            shape = meta[f'{handedness}_hand_shape'][t]
            if handedness == 'left':
                side = Handedness.Left
            else:
                side = Handedness.Right
            # faces = hand_data_provider.get_hand_mesh_faces_and_normals(side)[0]

            rot, tsl, hA = hand_pose[..., :3], hand_pose[..., 3:6], hand_pose[..., 6:]

            rot = torch.FloatTensor(rot)[None]
            tsl = torch.FloatTensor(tsl)[None]
            hA = torch.FloatTensor(hA)[None]
            shape = torch.FloatTensor(shape)[None]
            hA = sided_wrapper[handedness].pca_to_pose(hA)

            mesh, _ = sided_wrapper[handedness](None, hA, axisang=rot, trans=tsl, th_betas=shape)
            faces = sided_wrapper[handedness].hand_faces.cpu().numpy()[0]
            vertices = mesh.verts_packed().cpu().numpy()
            rr.log(f"world/hand/{handedness}", rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
            ))
    return 


@torch.no_grad()
def run_mono_depth(
    seq="P0001_624f2ba9",
    save_index=None,
    num=-1,
    vis_rr=False,
    vis_plt=False,
    strategy='median',
    **kwargs,
):
    video_segments = pickle.load(
        open(osp.join(root_dir, "pred_pose", seq, "video_segments.pkl"), "rb")
    )
    uid_list = list(video_segments.keys())
    for uid in uid_list:
        print(uid, len(video_segments[uid]['mask']))

    depth_model = UniDepthWrapper()

    intrinsic = pickle.load(
        open(osp.join(root_dir, "pred_pose", seq, "intrinsic.pkl"), "rb")
    )
    image_list = sorted(glob(osp.join(root_dir, "pred_pose", seq, "images", "*.jpg")))
    if num > 0:
        image_list = image_list[:num]

    # Initialize dicts for each uid
    cTo_dict = {uid: [] for uid in uid_list}
    valid_dict = {uid: [] for uid in uid_list}

    # Re-run the above loop, but append to dicts instead of flat lists
    for t, image in enumerate(tqdm(image_list)):
        image = imageio.imread(image)
        depth, prediction = depth_model(image, intrinsic)
        cPoints = prediction['points']
            
        for uid in uid_list:
            mask = video_segments[uid]['mask'][t][0]
            # print(mask.shape, mask.sum())
            mask = PoseWrapper.process_check_mask(mask)
            if mask is False:
                # print(f"Object {uid} is not detected / truncated in this frame.")
                cTo = np.eye(4)
                cTo_dict[uid].append(cTo)
                valid_dict[uid].append(False)
                continue

            mask = torch.from_numpy(mask).to(cPoints.device)[None, None]  # (1, 1, H, W)
            mask_exp = mask.repeat(1, 3, 1, 1)
            obj_points = cPoints[mask_exp]  # (P, 3)
            if obj_points.numel() == 0:
                print(f"No valid points for object {uid} in this frame.")
                cTo = np.eye(4)
                cTo_dict[uid].append(cTo)
                valid_dict[uid].append(False)
                continue
            obj_points = obj_points.detach().cpu().numpy().reshape(3, -1).T  # (P, 3)
            if strategy == 'median':
                median_xyz = np.median(obj_points, axis=0)
                print(f"Object {uid} median 3D position: {median_xyz}")
                cObj = median_xyz
            elif strategy == 'mean':
                mean_xyz = np.mean(obj_points, axis=0)
                print(f"Object {uid} mean 3D position: {mean_xyz}")
                cObj = mean_xyz
            else:
                print(f"Unknown strategy: {strategy}")
                raise ValueError(f"Unknown strategy: {strategy}")
            cTo = np.eye(4)
            cTo[:3, 3] = cObj
            cTo_dict[uid].append(cTo)
            valid_dict[uid].append(True)
            if t == 99 and int(uid) == 194930206998778:
                # save mask, save 
                imageio.imwrite(f'outputs/image_{t}_{uid}.png', image)
                imageio.imwrite(f'outputs/mask_{t}_{uid}.png', mask.cpu().numpy()[0, 0].astype(np.uint8) * 255)

    # Save the results in the same format as run_foundation_pose, but to pose_3d.pkl
    save_dict = {}
    for uid in uid_list:
        save_dict[uid] = {
            "cTo": cTo_dict[uid],
            "valid": valid_dict[uid]
        }

    save_path = os.path.join(root_dir, "pred_pose", seq, "pose_3d.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Saved 3D pose results to {save_path}")
    

@torch.no_grad()
def run_foundation_pose(
    seq="P0001_624f2ba9",
    save_index=None,
    num=-1,
    vis_rr=False,
    vis_plt=False,
):
    if save_index is None:
        save_index = seq
    
    depth_list = None
    video_segments = pickle.load(
        open(osp.join(root_dir, "pred_pose", seq, "video_segments.pkl"), "rb")
    )
    uid_list = list(video_segments.keys())
    alpha, pose_wrapper = esitimate_alpha(seq, )
    
    meta_file = osp.join(root_dir, "pred_pose", seq, "meta.pkl")
    meta = pickle.load(open(meta_file, "rb"))

    intrinsic = pickle.load(
        open(osp.join(root_dir, "pred_pose", seq, "intrinsic.pkl"), "rb")
    )
    image_list = sorted(glob(osp.join(root_dir, "pred_pose", seq, "images", "*.jpg")))

    print(f"Rerun logging: {'Enabled' if vis_rr else 'Disabled'}")
    
    for uid in uid_list:
        uid = int(uid)
        print(f"Processing object {uid} ({uid2name[uid]})")
        
        mesh_file = osp.join(root_dir, "assets", f"{uid}.glb")
        mask_list = video_segments[uid]["mask"]
        bbox_list = video_segments[uid]["bbox"]
    
        T_w_o = [e[str(uid)] for e in meta["T_w_o"]]
        T_w_c = [e for e in meta["T_w_c"]]

        if num > 0:
            image_list = image_list[:num]
            T_w_o = T_w_o[:num]
            T_w_c = T_w_c[:num]
            mask_list = mask_list[:num]
            bbox_list = bbox_list[:num]

        pose_wrapper.reset_posetracker(mesh_file, mask_list, bbox_list)

        #  # SE inverse T_w_c
        T_c_w = [np.linalg.inv(T_w_c[i]) for i in range(len(T_w_c))]
        pose_gt_list = T_c_o = [T_c_w[i] @ T_w_o[i] for i in range(len(T_w_o))]
        depth_list = [np.zeros_like(imageio.imread(image)[...,0]) for image in image_list]

        # Initialize Rerun recording (if enabled)
        os.makedirs(f'outputs/{save_index}/{uid2name[uid]}', exist_ok=True)
        if vis_rr:
            rr.init("foundation_pose", spawn=False)
            rr.save(f"outputs/{save_index}/{uid2name[uid]}/pose.rrd")
        
        img = pose_wrapper.read_image(image_list[0])
        H, W = img.shape[:2]
        
        valid_list = []
        cTo_list = []
        for i, (pose, valid) in enumerate(pose_wrapper.track_video(image_list, intrinsic)):
            # pred_depth, intrinsic, _ = pose_wrapper.predict_depth(image_list[i], intrinsic, pose_wrapper.alpha)
            if pose is None:
                pose = np.eye(4)
            cTo_list.append(pose.copy())
            valid_list.append(valid)
            # Log pose to Rerun (if enabled)
            if vis_rr:
                rr.set_time_sequence("frame", i)
                
                # Log the 4x4 transformation matrix as a transform
                # Extract translation and rotation from the 4x4 matrix
                T_c_o = pose
                T_w_o = T_w_c[i] @ T_c_o
                
                translation = T_w_o[:3, 3]
                rotation_matrix = T_w_o[:3, :3]
                
                # Convert rotation matrix to quaternion
                rotation = Rotation.from_matrix(rotation_matrix)
                quaternion = rotation.as_quat()  # Returns [x, y, z, w]
                
                # Log the 4x4 transformation matrix as a transform
                rr.log("world/object_pose", rr.Transform3D(
                    translation=translation,
                    rotation=rr.Quaternion(xyzw=quaternion)
                ))
                if i == 0:                
                    rr.log("world/object_pose", rr.Asset3D(
                        path=mesh_file,
                    ))

                rotation = Rotation.from_matrix(pose_gt_list[i][:3, :3])
                quaternion = rotation.as_quat()
                rr.log("world/object_pose_gt", rr.Transform3D(
                    translation=pose_gt_list[i][:3, 3],
                    rotation=rr.Quaternion(xyzw=quaternion)
                ))
                rr.log("world/object_pose_gt", rr.Asset3D(
                    path=mesh_file,
                ))

                rr.log("world/camera", rr.Pinhole(
                    width=W,
                    height=H,
                    focal_length=intrinsic[0],
                ))

                rr.log("world/camera", rr.Transform3D(
                    translation=T_w_c[i][:3, 3],
                    rotation=rr.Quaternion(xyzw=Rotation.from_matrix(T_w_c[i][:3, :3]).as_quat())
                ))
                # rr.log("world/camera/depth_gt", rr.DepthImage(depth_list[i]))
                # rr.log("world/camera/depth_pred", rr.DepthImage(pred_depth))

            bbox = pose_wrapper.bbox
            fx, fy, cx, cy = intrinsic
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K = K.reshape(3, 3)


            if vis_plt:
                color = imageio.imread(image_list[i])
                center_pose = pose@np.linalg.inv(pose_wrapper.to_origin)
                vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                os.makedirs(f'outputs/{save_index}/{uid2name[uid]}/track_vis', exist_ok=True)
                imageio.imwrite(f'outputs/{save_index}/{uid2name[uid]}/track_vis/{i:05d}.png', vis)
            
            
            if only_init:
                break
        
        fname = f'outputs/{save_index}/{uid2name[uid]}/pose.pkl'
        with open(fname, 'wb') as f:
            pickle.dump({'cTo': cTo_list, 'cTw': T_w_c, 'valid': valid_list}, f)
        print(f"Saved pose to {fname}")

        if vis_plt:
            # Create video for this object using proper ffmpeg command for non-consecutive frames
            track_vis_dir = f'outputs/{save_index}/{uid2name[uid]}/track_vis'
            output_video = f'outputs/{save_index}/{uid2name[uid]}_video.mp4'
            
            # Use pattern matching for non-consecutive frame numbers
            cmd = f'ffmpeg -pattern_type glob -i "{track_vis_dir}/*.png" -c:v libx264 -pix_fmt yuv420p {output_video} -y'
            print(f"Creating video for {uid2name[uid]}: {cmd}")
            os.system(cmd)
        
        print(f"Completed processing object {uid} ({uid2name[uid]})")
    
    print(f"Completed processing all {len(uid_list)} objects")
    return


def log_list_of_camera_poses(poses, W, H, focal_length, name='world/camera'):
    for t, pose in enumerate(poses):
        rr.set_time_sequence("frame", t)
        # log pinhole camera

        rr.log(name, rr.Pinhole(
            width=W,
            height=H,
            focal_length=float(focal_length),
        ))
        rr.log(name, rr.Transform3D(
            translation=pose[:3, 3],
            rotation=rr.Quaternion(xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat())
        ))
    return 

def log_list_of_obj_poses(poses, mesh_file, name='world/object_pose'):
    for t, pose in enumerate(poses):
        rr.set_time_sequence("frame", t)
        rr.log(name, rr.Transform3D(
            translation=pose[:3, 3],
            rotation=rr.Quaternion(xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat())
        ))
        rr.log(name, rr.Asset3D(
            path=mesh_file,
        ))
    return 

def log_list_of_hand_poses(vertices_list, faces, name='world/hand/right'):
    for t, vertices in enumerate(vertices_list):
        rr.set_time_sequence("frame", t)
        rr.log(name, rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=faces,
        ))
    return 




def check(uid='106957734975303'):
    tmp_file = 'outputs/tmp_cPoints_0.pkl'
    with open(tmp_file, 'rb') as f:
        data = pickle.load(f)
    prediction = data['prediction']
    mask = data['mask']
    mask_exp = mask.repeat(1, 3, 1, 1)
    intrinsic = data['intrinsic']
    cPoints = prediction['points']
    print(cPoints.shape, mask_exp.shape)
    cPoints = cPoints[mask_exp]
    cPoints = cPoints.reshape(3, -1)
    cPoints = cPoints.detach().cpu().numpy()
    cPoints = cPoints.T


    print(cPoints.shape)
    print(cPoints)

    mesh_file = osp.join(root_dir, "assets", f"{uid}.glb")
    mesh = trimesh.load(mesh_file)
    mesh = as_mesh(mesh)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    template_bounds = bbox

    bounds = np.stack([cPoints.min(axis=0), cPoints.max(axis=0)], axis=0)
    print(bounds)
    scale = compute_alignment_scale(cPoints, mesh.vertices, method='auto')
    print('scale', scale)
    cPoints = cPoints * scale

    import plotly.graph_objects as go
    fig = go.Figure()
    # INSERT_YOUR_CODE
    fig.add_trace(go.Scatter3d(
        x=cPoints[:, 0], y=cPoints[:, 1], z=cPoints[:, 2],
        mode='markers',
        name='Camera Points'
    ))
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        name='Mesh'
    ))
    fig.update_layout(
        legend=dict(
            # x=0.01,
            # y=0.99,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='Black',
            borderwidth=1
        ),
        showlegend=True
    )
    # 1:1 aspect ratio
    fig.update_layout(scene_aspectmode='data', scene_aspectratio=dict(x=1, y=1, z=1))
    
    # save html
    fig.write_html('outputs/tmp.html')



def compute_scale_bbox_diagonal(cPoints, template_points):
    """
    Compute scale factor using bounding box diagonal ratio
    """
    # Get bounds for both point clouds
    cPoints_bounds = np.stack([cPoints.min(axis=0), cPoints.max(axis=0)], axis=0)
    template_bounds = np.stack([template_points.min(axis=0), template_points.max(axis=0)], axis=0)
    
    # Compute diagonal lengths
    cPoints_diagonal = np.linalg.norm(cPoints_bounds[1] - cPoints_bounds[0])
    template_diagonal = np.linalg.norm(template_bounds[1] - template_bounds[0])
    
    # Scale factor
    scale = template_diagonal / cPoints_diagonal
    return scale


def compute_scale_rms_distance(cPoints, template_points):
    """
    Compute scale factor using RMS distance from centroid
    """
    # Compute centroids
    cPoints_centroid = np.mean(cPoints, axis=0)
    template_centroid = np.mean(template_points, axis=0)
    
    # Compute RMS distances
    cPoints_rms = np.sqrt(np.mean(np.sum((cPoints - cPoints_centroid)**2, axis=1)))
    template_rms = np.sqrt(np.mean(np.sum((template_points - template_centroid)**2, axis=1)))
    
    # Scale factor
    scale = template_rms / cPoints_rms
    return scale


def compute_scale_max_distance(cPoints, template_points):
    """
    Compute scale factor using maximum distance from centroid
    """
    # Compute centroids
    cPoints_centroid = np.mean(cPoints, axis=0)
    template_centroid = np.mean(template_points, axis=0)
    
    # Compute maximum distances
    cPoints_max_dist = np.max(np.linalg.norm(cPoints - cPoints_centroid, axis=1))
    template_max_dist = np.max(np.linalg.norm(template_points - template_centroid, axis=1))
    
    # Scale factor
    scale = template_max_dist / cPoints_max_dist
    return scale


def compute_scale_procrustes(cPoints, template_points):
    """
    Compute scale factor using Procrustes analysis
    """
    # Center both point clouds
    cPoints_centered = cPoints - np.mean(cPoints, axis=0)
    template_centered = template_points - np.mean(template_points, axis=0)
    
    # Compute scale factor
    cPoints_scale = np.sqrt(np.sum(cPoints_centered**2))
    template_scale = np.sqrt(np.sum(template_centered**2))
    
    scale = template_scale / cPoints_scale
    return scale


def compute_alignment_scale(cPoints, template_points, method='auto'):
    """
    Compute scaling factor to align partial points with template points
    
    Args:
        cPoints: (P, 3) partial point cloud
        template_points: (Q, 3) template point cloud  
        method: 'bbox', 'rms', 'max', 'procrustes', or 'auto'
    
    Returns:
        scale: float, scaling factor to apply to cPoints
    """
    
    if method == 'bbox':
        return compute_scale_bbox_diagonal(cPoints, template_points)
    elif method == 'rms':
        return compute_scale_rms_distance(cPoints, template_points)
    elif method == 'max':
        return compute_scale_max_distance(cPoints, template_points)
    elif method == 'procrustes':
        return compute_scale_procrustes(cPoints, template_points)
    elif method == 'auto':
        # Try multiple methods and return the median
        scales = []
        scales.append(compute_scale_bbox_diagonal(cPoints, template_points))
        scales.append(compute_scale_rms_distance(cPoints, template_points))
        scales.append(compute_scale_max_distance(cPoints, template_points))
        scales.append(compute_scale_procrustes(cPoints, template_points))
        
        return np.median(scales)
    else:
        raise ValueError(f"Unknown method: {method}")



def get_calibration(
    rgb_camera_calibration,
    should_rectify_image=True,
    should_rotate_image=True,
):
    """
    get intrinsics of pinhole camera, (and T_device_camera from get_transform_device_camera())

    devcie_points -> pixels: get_camera_projection_from_device_point(p, camera_calibration)
    :param rgb_camera_calibration: _description_
    :param should_rectify_image: _description_, defaults to True
    :param should_rotate_image: _description_, defaults to True
    :raises NotImplementedError: _description_
    :return: _description_
    """
    if should_rectify_image:
        rgb_linear_camera_calibration = calibration.get_linear_camera_calibration(
            int(rgb_camera_calibration.get_image_size()[0]),
            int(rgb_camera_calibration.get_image_size()[1]),
            rgb_camera_calibration.get_focal_lengths()[0],
            "pinhole",
            rgb_camera_calibration.get_transform_device_camera(),
        )
        if should_rotate_image:
            rgb_rotated_linear_camera_calibration = (
                calibration.rotate_camera_calib_cw90deg(
                    rgb_linear_camera_calibration
                )
            )
            camera_calibration = rgb_rotated_linear_camera_calibration
        else:
            camera_calibration = rgb_linear_camera_calibration
    else:  # No rectification
        if should_rotate_image:
            raise NotImplementedError(
                "Showing upright-rotated image without rectification is not currently supported.\n"
                "Please use --no_rotate_image_upright and --no_rectify_image together."
            )
        else:
            camera_calibration = rgb_camera_calibration
    return camera_calibration    

def debug_pose():
    pose_file = '/move/u/yufeiy2/data/HOT3D/pred_pose/P100/meta.pkl'
    with open(pose_file, 'rb') as f:
        data = pickle.load(f)
    
    T_inds = range(0, 100, 3)
    wTdevice = np.array(data['T_w_c']) 
    wTo_gt = data['T_w_o']

    hot3d_loader = HOT3DLoader(
        seq_path=osp.join(root_dir, 'P0001_624f2ba9'),
        object_library_folder=osp.join(root_dir, 'assets'),
    )           

    _, rgb_camera_calibration = hot3d_loader._device_data_provider.get_camera_calibration(
            hot3d_loader.image_stream_id
        )
    # this is the code to get deviceTcamera!!!!!
    camera_calibration = hot3d_loader.get_calibration(rgb_camera_calibration)
    
    # camera_calibration = get_calibration(rgb_camera_calibration)
    deviceTcamera = camera_calibration.get_transform_device_camera() 
    W, H = camera_calibration.get_image_size()
    fx, fy = camera_calibration.get_focal_lengths()

    wTcamera_gt = wTdevice @ deviceTcamera.to_matrix()[None]

    # ...
    pred_dir = 'outputs/P0001_624f2ba9/'


    rr.init("debug_pose")
    rr.save("outputs/debug_pose.rrd")
    for t in T_inds:
        rr.set_time_sequence("frame", t)
        # log objects 
        for uid in wTo_gt[t].keys():
            rr.log(f"world/objects_gt/{uid}", rr.Asset3D(
                path=osp.join(root_dir, 'assets', f"{uid}.glb")
            ))
            rr.log(f"world/objects_gt/{uid}", rr.Transform3D(
                translation=wTo_gt[t][uid][:3, 3],
                rotation=rr.Quaternion(xyzw=Rotation.from_matrix(wTo_gt[t][uid][:3, :3]).as_quat())
            ))

            if int(uid) not in uid2name:
                print(f"Object {uid} not found in uid2name")
                continue
            obj_name = uid2name[int(uid)]
            pose_file = osp.join(pred_dir, f"{obj_name}/pose.pkl")
            if not osp.exists(pose_file):
                print(f"Pose file {pose_file} not found")
                continue
            with open(pose_file, 'rb') as f:
                data = pickle.load(f)
            cTo_pred = data['cTo']
            # if is np.eye(4), skip
            if np.allclose(cTo_pred, np.eye(4)):
                print(f"Object {uid} is not valid at frame {t}")
                continue
            wTo_pred = wTcamera_gt[0:len(cTo_pred)] @ cTo_pred 
            rr.log(f"world/objects_pred/{uid}", rr.Asset3D(
                path=osp.join(root_dir, 'assets', f"{uid}.glb")
            ))
            rr.log(f"world/objects_pred/{uid}", rr.Transform3D(
                translation=wTo_pred[t][:3, 3],
                rotation=rr.Quaternion(xyzw=Rotation.from_matrix(wTo_pred[t][:3, :3]).as_quat())
            ))
        
        rr.log("world/camera", rr.Pinhole(
            width=W,
            height=H,
            focal_length=(float(fx), float(fy)),
        ))
        rr.log("world/camera", rr.Transform3D(
            translation=wTcamera_gt[t][:3, 3],
            rotation=rr.Quaternion(xyzw=Rotation.from_matrix(wTcamera_gt[t][:3, :3]).as_quat())
        ))
        



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
        cTo_shelf = objects[uid]['cTo_shelf']
        wTc = data['wTc']
        # cTw = np.linalg.inv(wTc)  
        wTo_shelf = wTc @ cTo_shelf
        objects[uid]['wTo_shelf'] = wTo_shelf

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


def vis_mini_dataset(data_file):
    data = load_pickle(data_file)
    seq = osp.basename(data_file).split(".")[0]
    key = list(data.keys())[0]
    data = data[key]

    rr.init(f"vis_mini_dataset_{seq}")
    rr.save(f"outputs/vis_mini_dataset_{seq}.rrd")

    # draw camera, draw objects, draw hands
    for t in range(len(data['wTc'])):
        rr.set_time_sequence("frame", t)
        if t == 0:
            rr.log("world/camera", rr.Pinhole(
                width=data['intrinsic'][0],
                height=data['intrinsic'][1],
                focal_length=data['intrinsic'][0],
            ))

        rr.log("world/camera", rr.Transform3D(
            translation=data['wTc'][t][:3, 3],
            rotation=rr.Quaternion(xyzw=Rotation.from_matrix(data['wTc'][t][:3, :3]).as_quat())
        ))
        for uid in data['objects']:
            if t == 0:

                rr.log(f"world/objects/{uid}", rr.Asset3D(
                    path=osp.join(root_dir, "assets", f"{uid}.glb")
                ))
                rr.log(f"world/objects_pred/{uid}", rr.Asset3D(
                    path=osp.join(root_dir, "assets", f"{uid}.glb")
                ))
                
            rr.log(f"world/objects/{uid}", rr.Transform3D(
                translation=data['objects'][uid]['wTo'][t][:3, 3],
                rotation=rr.Quaternion(xyzw=Rotation.from_matrix(data['objects'][uid]['wTo'][t][:3, :3]).as_quat())
            ))
            if data['objects'][uid]['shelf_valid'][t]:
                # use wTo_shelf
                wTo_pred = data['objects'][uid]['wTo_shelf'][t]
                rr.log(f"world/objects_pred/{uid}", rr.Transform3D(
                    translation=wTo_pred[:3, 3],
                    rotation=rr.Quaternion(xyzw=Rotation.from_matrix(wTo_pred[:3, :3]).as_quat())
                ))
        

def make_a_mini_dataset(seq='P0001_624f2ba9', num=-1, pred='fp'):
    # format: 
    # "objects": ['uid1', 'uid2', ...], 
    # "left_hand_theta": [T, 3+3+15], 3aa+3tsl+pcax15
    # "left_hand_shape": [T, 10],
    # "right_hand_theta": [T, 3+3+15],
    # "right_hand_shape": [T, 10],
    # "wTc": [T, 4, 4],
    # intrinsic: [3, 3],
    # "uid{i}_cTo_shelf": [T, 4, 4],
    # "uid{i}_shelf_valid": [T, ],
    # "uid{i}_wTo": [T, 4, 4],
    # "uid{i}_gt_valid": [T, ],

    meta_file = osp.join(root_dir, "pred_pose", seq, "meta.pkl")
    if pred == 'fp':
        pred_list = glob(osp.join('outputs/', seq, "*/pose.pkl"))
        uid2pred = {}
        
        for pred_file in pred_list:
            name = pred_file.split('/')[-2]
            with open(pred_file, 'rb') as f:
                data = pickle.load(f)
            uid2pred[name2uid[name]] = data
    elif pred == '3d':
        pred_file = osp.join(root_dir, "pred_pose", seq, "pose_3d.pkl")
        with open(pred_file, 'rb') as f:
            uid2pred = pickle.load(f)
    intrinsic_file = osp.join(root_dir, "pred_pose", seq, "intrinsic.pkl")
    
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    with open(intrinsic_file, 'rb') as f:
        intrinsic = pickle.load(f)

    data = {}
    data['objects'] = list(uid2pred.keys())
    for key in ['left_hand_theta', 'left_hand_shape', 'right_hand_theta', 'right_hand_shape']:
        data[key] = meta[key]
    
    fx, fy, cx, cy = intrinsic
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    data['intrinsic'] = K

    data['wTc'] = meta['T_w_c']
    for uid in data['objects']:
        shelf = uid2pred[uid]
        for t in range(len(shelf['cTo'])):
            if shelf['cTo'][t] is None:
                print('cTo is None at frame', t)
                assert shelf['valid'][t] is False
                shelf['cTo'][t] = np.eye(4)
        data[f'obj_{uid}_cTo_shelf'] = shelf['cTo']
        data[f'obj_{uid}_shelf_valid'] = shelf['valid']
        # data[f'obj_{uid}_wTo'] = 
        data[f'obj_{uid}_wTo'] = [meta['T_w_o'][t][str(uid)] for t in range(len(meta['T_w_o']))]
    
    for k, v in data.items():
        data[k] = np.array(v)
        if num > 0 and 'objects' != k:
            data[k] = data[k][:num]
        print(k, data[k].shape)
    
    save_file = osp.join(root_dir, "pred_pose", f"mini_{seq}_{pred}")
    np.savez_compressed(save_file, **data)
    print(f"Saved mini dataset to {save_file}")
    



    
depth_zero = True
zero_depth = 0
depth_model = 'unidepth' #  'metric3d'
only_init = False
# 37787722328019: keyboard
alpha = 1
# 223371871635142: mug white

mano_model_folder = 'hot3d/hot3d/mano_v1_2/models'
# P0001_624f2ba9
# P0003_c701bd11
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    library = json.load(open(osp.join(root_dir, "assets", "instance.json"), "r"))
    uid2name = {int(k): v['instance_name'] for k, v in library.items()}
    name2uid = {v: k for k, v in uid2name.items()}
    Fire(save_one)
    # Fire(get_all_masks)
    
    # alpha = 1 # 0.6  # this works for cube
    # Fire(run_foundation_pose)
    # Fire(run_mono_depth)
    # Fire(esitimate_alpha)

    # Fire(vis_meta)
    
    # Fire(make_a_mini_dataset)
    Fire(vis_mini_dataset)


    # run_foundation_pose(save_index='demo_alpha1')
    # run_foundation_pose(save_index='demo_alpha06')
    # debug_glb()


    # check()


    # debug_space()
    # debug_pose()