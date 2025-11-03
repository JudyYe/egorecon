
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
from FoundationPose.learning.training.predict_score import ScorePredictor
from FoundationPose.learning.training.predict_pose_refine import PoseRefinePredictor
from FoundationPose.Utils import draw_posed_3d_box, draw_xyz_axis
from PIL import Image
from torchvision import transforms

depth_zero = True
zero_depth = 0
depth_model = 'metric3d' #  'metric3d'
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
        return depth, prediction


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
        if self.process_check_mask(obj_mask) is False:
            return None
        
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
        return pose

    @staticmethod
    def process_check_mask(mask):
        """
        Check if mask is valid for pose estimation.
        Returns False if mask is invalid, otherwise returns processed mask.
        """
        print('mask', mask.shape)
        mask = mask.astype(np.uint8) * 255
        H, W = mask.shape[:2]
        
        # Check if mask is all zeros (out of screen)
        if mask.sum() < 10:
            return False
        
        return True

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
        
        for i, img_path in enumerate(image_list):
            self.index = i
            rgb = self.read_image(img_path)
            
            # Get mask for current frame (assume mask_list is (T, H, W) numpy array)
            if self.mask_list is not None and i < len(self.mask_list):
                obj_mask = self.mask_list[i]
            else:
                obj_mask = None
            
            # Check if mask is valid - if not, skip registration and continue
            mask_valid = self.process_check_mask(obj_mask) if obj_mask is not None else False
            
            if not mask_valid:
                # Mask is invalid - don't try to register, just append identity pose
                logging.info(f"Frame {i}: Mask is invalid, skipping...")
                cTo_list.append(np.eye(4))
                valid_list.append(False)
                continue
            
            if not pose_initialized:
                # Initialize pose
                logging.info(f"Frame {i}: Initializing pose...")
                current_pose = self.register_one(rgb, intrinsic, obj_mask.copy())
                
                if current_pose is not None:
                    pose_initialized = True
                    logging.info(f"Frame {i}: Pose initialization successful")
                    cTo_list.append(current_pose)
                    valid_list.append(True)
                else:
                    logging.warning(f"Frame {i}: Pose initialization failed")
                    cTo_list.append(np.eye(4))
                    valid_list.append(False)
            else:
                # Track pose
                current_pose = self.track_one(rgb, intrinsic)
                
                if current_pose is not None:
                    logging.info(f"Frame {i}: Tracking successful")
                    cTo_list.append(current_pose)
                    valid_list.append(True)
                else:
                    # Tracking failed - check if mask is valid before reinitializing
                    logging.warning(f"Frame {i}: Tracking failed, checking mask for reinitialization...")
                    if mask_valid:
                        pose_initialized = False
                        current_pose = self.register_one(rgb, intrinsic, obj_mask.copy())
                        
                        if current_pose is not None:
                            pose_initialized = True
                            cTo_list.append(current_pose)
                            valid_list.append(True)
                        else:
                            cTo_list.append(np.eye(4))
                            valid_list.append(False)
                    else:
                        # Mask is invalid, can't reinitialize
                        cTo_list.append(np.eye(4))
                        valid_list.append(False)
        
        # Convert to numpy arrays
        cTo = np.array(cTo_list)  # (T, 4, 4)
        valid = np.array(valid_list)  # (T,)
        
        return cTo, valid