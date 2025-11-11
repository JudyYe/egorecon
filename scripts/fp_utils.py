from tqdm import tqdm
import imageio
import os
import os.path as osp
import pickle
import numpy as np
import cv2
import trimesh
import logging
import torch
import nvdiffrast.torch as dr
from FoundationPose.estimater import FoundationPose
from FoundationPose.Utils import nvdiffrast_render

from FoundationPose.learning.training.predict_score import ScorePredictor
from FoundationPose.learning.training.predict_pose_refine import PoseRefinePredictor
from FoundationPose.Utils import draw_posed_3d_box, draw_xyz_axis
from PIL import Image
from torchvision import transforms

depth_zero = True
zero_depth = 0
depth_model = 'metric3d' #  'metric3d'
# depth_model = 'unidepth' #  'metric3d'
only_init = False
# 37787722328019: keyboard
alpha = 1
# 223371871635142: mug white

def as_mesh(mesh):
    # scene to mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.dump(concatenate=False)]
        )
    return mesh

def setup_foundation_pose(mesh_file, debug=True, debug_dir='./realsense_debug'):
    """
    Initialize FoundationPose with the given mesh file.
    
    Args:
        mesh_file (str): Path to the object mesh file (.obj)
        debug (bool): Enable debug mode
        debug_dir (str): Directory for debug outputs
    
    Returns:
        FoundationPose: Initialized pose estimator
        dict: Object metadata (bbox, to_origin, etc.)
    """
    # Load mesh
    mesh = trimesh.load(mesh_file)
    mesh = as_mesh(mesh)
    
    # Create debug directory
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(f'{debug_dir}/poses', exist_ok=True)
    os.makedirs(f'{debug_dir}/visualizations', exist_ok=True)
    
    # Initialize FoundationPose components
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    
    # Create FoundationPose estimator
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx
    )
    
    # Get object metadata
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
    
    metadata = {
        'mesh': mesh,
        'bbox': bbox,
        'to_origin': to_origin,
        'extents': extents
    }
    
    logging.info("FoundationPose initialization completed")
    return estimator, metadata


class UniDepthWrapper:
    def __init__(self):
        version = "v2"
        backbone = "vitl14"
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.depth_model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True)
        self.depth_model.eval()
        self.depth_model.to(device)
        self.device = device

    def __call__(self, rgb, intrinsic):
        import torch
        fx, fy, cx, cy = intrinsic
        intrinsic_tensor = torch.FloatTensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        rgb_tensor = torch.from_numpy(rgb/255.0).permute(2, 0, 1)  # C, H, W
        
        prediction = self.depth_model.infer(rgb_tensor, intrinsic_tensor)
        depth = prediction["depth"]
        
        depth = depth.squeeze().cpu().numpy().copy()
        return depth


class PoseWrapper:
    def __init__(self, mesh_file, debug=False, intrinsic=None, mask_list=None, depth_list=None, debug_dir='./outputs/debug_fp'):
        if depth_model == 'metric3d':
            import sys
            sys.path.append('thirdparty/')
            from Metric3D.metric3d_wrapper import Metric3D
            self.depth_model = Metric3D("thirdparty/Metric3D/weight/metric_depth_vit_large_800k.pth")
        elif depth_model == 'unidepth':
            self.depth_model = UniDepthWrapper()

        self.alpha = alpha
        
        self.mesh_file = mesh_file
        self.debug = debug
        self.intrinsic = intrinsic
        self.mask_list = mask_list
        self.depth_list = depth_list

        self.estimator, self.metadata = setup_foundation_pose(mesh_file, debug, debug_dir)

    def read_image(self, rgb):
        """Read image from file path or numpy array."""
        if isinstance(rgb, str):
            rgb = cv2.imread(rgb)[..., ::-1].copy()  # BGR to RGB
        elif isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
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
        pred_depth = self.depth_model(rgb, intrinsic)
        
        pred_depth = cv2.resize(pred_depth, (W, H))
        pred_depth = pred_depth * alpha

        return pred_depth, intrinsic

    def track_one(self, rgb, intrinsic=None):
        """Track pose from RGB image."""
        depth, intrinsic= self.predict_depth(rgb, intrinsic, self.alpha)
        if depth_zero:
            depth = np.zeros_like(depth)
        fx, fy, cx, cy = intrinsic
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.reshape(3, 3)

        pose = self.estimator.track_one(rgb=rgb.copy(), depth=depth, K=K, iteration=2)
        return pose

    def register_one(self, rgb, intrinsic=None, obj_mask=None):
        """
        Register/initialize pose from RGB image and mask.
        
        Args:
            rgb: (H, W, 3) numpy array
            intrinsic: Camera intrinsics [fx, fy, cx, cy]
            obj_mask: (H, W) boolean or uint8 mask
        
        Returns:
            pose: 4x4 pose matrix or None if failed
        """
        # if self.process_check_mask(obj_mask) is False:
        #     return np.eye(4), False
        
        depth, intrinsic= self.predict_depth(rgb, intrinsic, self.alpha)
        fx, fy, cx, cy = intrinsic
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        downsample = 4
        K[0:2, :] /= downsample
        rgb_down = cv2.resize(rgb, (rgb.shape[1]//downsample, rgb.shape[0]//downsample))
        depth_down = cv2.resize(depth, (depth.shape[1]//downsample, depth.shape[0]//downsample))
        
        obj_mask_down = obj_mask.astype(np.uint8) * 255
        obj_mask_down = cv2.resize(obj_mask_down, (obj_mask_down.shape[1]//downsample, obj_mask_down.shape[0]//downsample))
        obj_mask_down = obj_mask_down.astype(bool)
        
        pose = self.estimator.register(
            K=K, rgb=rgb_down.copy(), depth=depth_down, ob_mask=obj_mask_down, 
            iteration=5, zero_depth=zero_depth
        )
        success = self.estimator.pose_last is not None            
        return pose, success

    @staticmethod
    def process_check_mask(mask):
        """
        Check if mask is valid for pose estimation.
        Returns False if mask is invalid, otherwise returns processed mask.
        """
        mask = mask.astype(np.uint8) * 255
        H, W = mask.shape[:2]
        
        # Check if mask is all zeros (out of screen)
        if mask.sum() < 10:
            return False
        
        return True
    
    def track_video_from_best_frame(self, image_list, intrinsic):
        """Register on the frame with maximum mask coverage, then track forward/backward."""
        if self.mask_list is None or len(self.mask_list) == 0:
            raise ValueError("mask_list is required for track_video_from_best_frame")

        num_frames = len(image_list)
        if num_frames == 0:
            raise ValueError("Empty image_list passed to track_video_from_best_frame")
        if num_frames != len(self.mask_list):
            raise ValueError("mask_list length mismatch with image_list")

        mask_sums = self.mask_list.reshape(num_frames, -1).sum(axis=1)
        best_idx = int(np.argmax(mask_sums))
        if mask_sums[best_idx] <= 0:
            logging.warning("No valid mask pixels found; falling back to sequential tracking")
            return self.track_video(image_list, intrinsic)

        fx, fy, cx, cy = intrinsic
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        cTo_list = [np.eye(4) for _ in range(num_frames)]
        valid_list = np.zeros(num_frames, dtype=bool)
        score_list = np.zeros(num_frames, dtype=float)
        tracked_list = np.zeros(num_frames, dtype=float)

        # Register on best frame
        self.index = best_idx
        rgb_best = self.read_image(image_list[best_idx])
        obj_mask_best = self.mask_list[best_idx].copy()
        pose_best, success = self.register_one(rgb_best, intrinsic, obj_mask_best)
        tracked_list[best_idx] = 0
        if not success:
            logging.warning("Registration failed on best mask frame; falling back to sequential tracking")
            return self.track_video(image_list, intrinsic)

        best_pose_last = self.estimator.pose_last.clone()
        reproj_best = self.evaluate_tracking_score(pose_best, K, rgb_best, obj_mask_best, self.estimator)
        cTo_list[best_idx] = pose_best
        valid_list[best_idx] = True
        score_list[best_idx] = reproj_best

        # Track forward from best_idx + 1 to end
        current_pose = pose_best
        for frame_idx in range(best_idx + 1, num_frames):
            self.index = frame_idx
            rgb = self.read_image(image_list[frame_idx])
            obj_mask = self.mask_list[frame_idx]
            current_pose = self.track_one(rgb, intrinsic)
            reproj_iou = self.evaluate_tracking_score(current_pose, K, rgb, obj_mask, self.estimator)
            cTo_list[frame_idx] = current_pose
            valid_list[frame_idx] = mask_sums[frame_idx] > 0
            score_list[frame_idx] = reproj_iou
            tracked_list[frame_idx] = 1

        # Track backward from best_idx - 1 to start
        self.estimator.pose_last = best_pose_last.clone()
        current_pose = pose_best
        for frame_idx in range(best_idx - 1, -1, -1):
            self.index = frame_idx
            rgb = self.read_image(image_list[frame_idx])
            obj_mask = self.mask_list[frame_idx]
            current_pose = self.track_one(rgb, intrinsic)
            reproj_iou = self.evaluate_tracking_score(current_pose, K, rgb, obj_mask, self.estimator)
            cTo_list[frame_idx] = current_pose
            valid_list[frame_idx] = mask_sums[frame_idx] > 0
            score_list[frame_idx] = reproj_iou
            tracked_list[frame_idx] = 1

        cTo = np.array(cTo_list)
        valid = valid_list
        score = score_list
        tracked = np.array(tracked_list)
        return cTo, valid, score, tracked


    def track_video(self, image_list, intrinsic):
        """
        Track pose through video sequence with automatic reinitialization.
        
        Reinitialization conditions:
        1. Tracking fails (pose is None)
        2. Mask is all zeros (object out of screen)
        
        Args:
            image_list: List of image paths or numpy arrays
            intrinsic: Camera intrinsics [fx, fy, cx, cy]
        
        Returns:
            cTo: (T, 4, 4) numpy array of poses
            valid: (T,) numpy array of validity flags
        """
        pose_initialized = False
        current_pose = None
        cTo_list = []
        valid_list = []
        score_list = []
        tracked_list = []
        fx, fy, cx, cy = intrinsic
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        for i, img_path in enumerate(tqdm(image_list)):
            self.index = i
            rgb = self.read_image(img_path)
            
            # Get mask for current frame (assume mask_list is (T, H, W) numpy array)
            obj_mask = self.mask_list[i]
            
            # # Check if mask is valid - if not, skip registration and continue
            # mask_valid = self.process_check_mask(obj_mask) if obj_mask is not None else False
            
            # if not mask_valid:
            #     # Mask is invalid - don't try to register, just append identity pose
            #     logging.info(f"Frame {i}: Mask is invalid, skipping...")
            #     cTo_list.append(np.eye(4))
            #     valid_list.append(False)
            #     score_list.append(0.0)
            #     continue
            
            if not pose_initialized:
                # Initialize pose
                print(f"Frame {i}: Initializing pose...")
                current_pose, success = self.register_one(rgb, intrinsic, obj_mask.copy())
                tracked_list.append(0)
                
                if success:
                    pose_initialized = True
                    reproj_iou = self.evaluate_tracking_score(current_pose, K, rgb, obj_mask, self.estimator)
                    logging.info(f"Frame {i}: Pose initialization successful")
                    cTo_list.append(current_pose)
                    valid_list.append(True)
                    score_list.append(reproj_iou)
                else:
                    logging.warning(f"Frame {i}: Pose initialization failed")
                    cTo_list.append(np.eye(4))
                    valid_list.append(False)
                    score_list.append(0.0)
            else:
                current_pose = self.track_one(rgb, intrinsic)
                # get reprojection IoU for reset
                reproj_iou = self.evaluate_tracking_score(current_pose, K, rgb, obj_mask, self.estimator)
                th = 0.
                pose_initialized = reproj_iou >= th

                logging.info(f"Frame {i}: Tracking successful")
                cTo_list.append(current_pose)
                valid_list.append(pose_initialized)
                score_list.append(reproj_iou)
                tracked_list.append(1)
        
        # Convert to numpy arrays
        cTo = np.array(cTo_list)  # (T, 4, 4)
        valid = np.array(valid_list)  # (T,)
        score = np.array(score_list)  # (T,)
        tracked = np.array(tracked_list)  # (T,)

        return cTo, valid, score, tracked

    def evaluate_tracking_score(self, pose, K, color, current_mask, estimator=None):
        """
        Evaluate tracking score using reprojection error between projected mask and detected mask.
        
        Args:
            pose: 4x4 pose matrix
            K: Camera intrinsics matrix
            color: Color image
            mask_selector: Text2MaskPredictor instance
            estimator: FoundationPose estimator (optional, for mesh rendering)
        
        Returns:
            float: Tracking score (0.0 to 1.0, higher is better)
        """
        H, W = color.shape[:2]
        # Convert pose to centered mesh coordinates for rendering
        # pose_centered = pose @ np.linalg.inv(estimator.get_tf_to_centered_mesh().cpu().numpy())
        pose_centered = pose.astype(np.float32)
        # Render the mesh using nvdiffrast
        
        # Render mesh to get depth and mask
        # rendered_depth, rendered_mask 
        rendered_rgb, rendered_depth, rendered_normal_map = nvdiffrast_render(
            K=K, H=H, W=W, 
            ob_in_cams=torch.tensor(pose_centered.reshape(1, 4, 4), device='cuda'),
            glctx=estimator.glctx,
            mesh=estimator.mesh,
            mesh_tensors=estimator.mesh_tensors,
            # projection_mat=projection_mat,
            output_size=(H, W)
        )
        projected_mask = (rendered_depth[0] > 0)
        projected_mask = projected_mask.detach().cpu().numpy()
        
        # canvas = np.concatenate([current_mask, projected_mask], axis=1).astype(np.uint8) * 255
        # cv2.imwrite(save_pref + 'mask_projected.png', canvas)

        # Calculate IoU (Intersection over Union) between masks
        intersection = np.logical_and(projected_mask, current_mask).sum()
        union = np.logical_or(projected_mask, current_mask).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        
        # Convert IoU to a score (0.0 to 1.0)
        # IoU of 0.5+ is considered good tracking
        score = min(1.0, iou * 2.0)  # Scale IoU to get better score range
        
        return float(score)
