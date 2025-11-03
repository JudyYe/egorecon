# use grounded sam 2 to find the mask, instead of user selection
# monitor pose score, if it's below a threshold, re initialize pose

import imageio
import os
import os.path as osp
import pickle
import pyrealsense2 as rs
import numpy as np
import cv2
import trimesh
import logging
import torch
import nvdiffrast.torch as dr
from FoundationPose.Utils import nvdiffrast_render
from FoundationPose.estimater import FoundationPose
from FoundationPose.learning.training.predict_score import ScorePredictor
from FoundationPose.learning.training.predict_pose_refine import PoseRefinePredictor
from FoundationPose.Utils import draw_posed_3d_box, draw_xyz_axis
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from torchvision import transforms

import sys
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Grounded-SAM-2'))
print(f"Adding {path_to_add} to sys.path")
sys.path.insert(0, path_to_add)
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert

class Text2MaskPredictor:
    """Text-to-mask predictor using GroundingDINO + SAM2."""
    
    def __init__(self, device='cuda:0', text_prompt='yellow bottle', 
                 box_threshold=0.35, text_threshold=0.25):
        self.device = device
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
    
        # Initialize SAM2
        
        # Store functions as instance variables for later use
        self.predict_fn = predict
        self.box_convert = box_convert
        
        # Build SAM2 model
        sam2_model = build_sam2_hf('facebook/sam2-hiera-large', device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        # Build GroundingDINO model
        grounding_config = "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_checkpoint = "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        
        self.grounding_model = load_model(
            model_config_path=grounding_config,
            model_checkpoint_path=grounding_checkpoint,
            device=device
        )
        
        # Setup autocast for better performance
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        logging.info("Text2MaskPredictor initialized successfully")
        self.initialized = True

    def predict(self, image, text_prompt=None):
        """
        Predict mask from text prompt using GroundingDINO + SAM2.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            text_prompt (str): Text description of object to segment
            
        Returns:
            np.ndarray: Boolean mask of the detected object
        """
        if not self.initialized:
            logging.error("Text2MaskPredictor not initialized")
            return None
            
        if text_prompt is None:
            text_prompt = self.text_prompt
            
        # Convert BGR to RGB for GroundingDINO
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Load image for GroundingDINO
        image_source, image_tensor = self.load_image_from_array(rgb_image)
        
        # Set image for SAM2
        self.sam2_predictor.set_image(image_source)
        
        # Predict bounding boxes using GroundingDINO
        boxes, confidences, labels = self.predict_fn(
            model=self.grounding_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )
        
        if len(boxes) == 0:
            logging.warning(f"No objects detected for prompt: {text_prompt}")
            return None
        
        # Convert boxes to SAM2 format
        h, w, _ = image_source.shape
        print('boxes.device', boxes.device, 'self.device', self.device, confidences.device,  image_tensor.device)
        boxes = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
        input_boxes = self.box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        # Enable autocast for better performance
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            # Predict masks using SAM2
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        
        # Convert to numpy and select best mask
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Select the mask with highest confidence
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx].astype(bool)
        
        logging.info(f"Generated mask for '{text_prompt}' with confidence {scores[best_mask_idx]:.3f}")
        return best_mask
        

    @staticmethod
    def load_image_from_array(image_array):
        """Load image from numpy array for GroundingDINO compatibility."""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(pil_image)
        
        return image_array, image_tensor

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

def get_camera_intrinsics(intr):
    """
    Convert RealSense intrinsics to camera matrix K.
    
    Args:
        intr: RealSense intrinsics object
    
    Returns:
        np.ndarray: 3x3 camera matrix K
    """
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    return K


def pose_initialization_waiting_stage(estimator, metadata, K, color, depth, mask_selector):
    """
    Waiting stage for pose initialization with text-based mask prediction.
    
    Args:
        estimator: FoundationPose estimator
        metadata: Object metadata (bbox, to_origin, etc.)
        K: Camera intrinsics matrix
        color: Color image
        depth: Depth image
        mask_selector: Text2MaskPredictor instance
    
    Returns:
        tuple: (success, pose, mask) - success flag, initial pose, and object mask
    """

    # Step 2: Generate mask using text prompt
    logging.info("Starting text-based mask prediction...")
    object_mask = mask_selector.predict(color)
    
    if object_mask is None:
        logging.warning("Text-based mask prediction failed")
        return False, None, None
    
    # Step 3: Initialize pose with the selected mask
    logging.info("Initializing pose with selected mask...")

    # Convert BGR to RGB for FoundationPose
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    # Initialize pose
    initial_pose = estimator.register(
        K=K,
        rgb=rgb,
        depth=depth,
        ob_mask=object_mask,
        iteration=5
    )
    
    if initial_pose is None:
        logging.error("Pose initialization failed")
        return False, None, object_mask
    
    logging.info("Pose initialization successful")
    return True, initial_pose, object_mask
        

def evaluate_tracking_score(pose, K, color, mask_selector, estimator=None, save_pref='realsense_output/'):
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
    # Get current mask from mask selector
    print(type(color), mask_selector.device)
    current_mask = mask_selector.predict(color)
    if current_mask is None:
        print('current_mask is None')
        return 0.0
    
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


def create_simple_projected_mask(pose, K, H, W):
    """Create a simple projected mask using pose center and estimated size."""
    projected_mask = np.zeros((H, W), dtype=bool)
    
    # Project the pose center to 2D
    center_3d = pose[:3, 3]
    center_2d = K @ center_3d
    center_2d = center_2d[:2] / center_2d[2]
    
    # Create a simple circular mask around the projected center
    if 0 <= center_2d[0] < W and 0 <= center_2d[1] < H:
        # Estimate object size based on pose scale (simplified)
        scale = np.linalg.norm(pose[:3, :3], axis=0).mean()
        radius = max(20, int(50 * scale))  # Adaptive radius
        
        # Create circular mask
        y, x = np.ogrid[:H, :W]
        mask_circle = (x - center_2d[0])**2 + (y - center_2d[1])**2 <= radius**2
        projected_mask = mask_circle
    
    return projected_mask

def draw_pose_visualization(color, pose, K, bbox, to_origin, scale=0.1):
    """
    Draw pose visualization on the color image using FoundationPose visualization functions.
    
    Args:
        color (np.ndarray): Color image
        pose (np.ndarray): 4x4 pose matrix
        K (np.ndarray): Camera intrinsics
        bbox (np.ndarray): Object bounding box
        to_origin (np.ndarray): Transform to origin
        scale (float): Scale for visualization
    
    Returns:
        np.ndarray: Image with pose visualization
    """
    vis_img = color.copy()
    
    # Transform pose to centered mesh coordinates
    center_pose = pose @ np.linalg.inv(to_origin)
    
    # Use FoundationPose visualization functions
    # Draw 3D bounding box
    vis_img = draw_posed_3d_box(K, img=vis_img, ob_in_cam=center_pose, bbox=bbox)
    
    # Draw coordinate axes
    vis_img = draw_xyz_axis(vis_img, ob_in_cam=center_pose, scale=scale, K=K, 
                           thickness=3, transparency=0, is_input_rgb=True)
    
    return vis_img

def stream_pose_estimation(mesh_file, output_dir='./realsense_output', 
                          est_refine_iter=5, track_refine_iter=2, 
                          save_poses=True, save_visualizations=False, 
                          show_preview=True, device='cuda:0', text_prompt='yellow bottle',
                          tracking_threshold=0.5):
    """
    Main function to stream RGBD from RealSense and estimate object poses.
    
    Args:
        mesh_file (str): Path to object mesh file
        output_dir (str): Directory to save outputs
        est_refine_iter (int): Iterations for initial pose estimation
        track_refine_iter (int): Iterations for pose tracking
        save_poses (bool): Whether to save pose files
        save_visualizations (bool): Whether to save visualization images
        show_preview (bool): Whether to show real-time preview
        device (str): Device for SAM2 (cuda:0, cpu, etc.)
        text_prompt (str): Text prompt for object detection and segmentation
        tracking_threshold (float): Tracking score threshold for reinitialization (0.0-1.0)
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize FoundationPose
    estimator, metadata = setup_foundation_pose(mesh_file, debug=0, debug_dir=output_dir)
    
    # Initialize text-to-mask predictor
    mask_selector = Text2MaskPredictor(device=device, text_prompt=text_prompt)
    if not mask_selector.initialized:
        logging.error("Failed to initialize Text2MaskPredictor. Cannot proceed without mask prediction.")
        return
    
    # Setup RealSense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = get_camera_intrinsics(intr)
    
    print(f"Camera intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, ppx={intr.ppx:.2f}, ppy={intr.ppy:.2f}")
    print(f"Depth scale: {depth_scale}")
    
    # Tracking variables
    frame_count = 0
    pose_initialized = False
    current_pose = None
    tracking_loop_active = False
    tracking_score_threshold = tracking_threshold  # Threshold for tracking quality
    
    try:
        while True:
            frames = pipe.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            
            # Convert BGR to RGB for FoundationPose
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            
            if not tracking_loop_active:
                # Show waiting message and wait for user to press key to start tracking loop
                vis_img = color.copy()
                cv2.putText(vis_img, "Press 'i' to start tracking loop", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(vis_img, "Press ESC to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(vis_img, f"Frame: {frame_count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if show_preview:
                    cv2.imshow("RealSense Pose Estimation", vis_img)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('i'):
                        # User pressed 'i' - start tracking loop
                        logging.info(f"User started tracking loop (frame {frame_count})")
                        tracking_loop_active = True
                        pose_initialized = False  # Start with pose initialization
                    elif key == 27:  # ESC to quit
                        break
                else:
                    # If no preview, just wait a bit and continue
                    import time
                    time.sleep(0.1)
            else:
                tracking_score = -1
                # Tracking loop is active
                if not pose_initialized:
                    # Step 2: Initialize pose if not initialized
                    logging.info(f"Initializing pose (frame {frame_count})")
                    
                    if mask_selector is not None:
                        # Use text-based mask prediction
                        
                        success, initial_pose, selected_mask = pose_initialization_waiting_stage(
                            estimator, metadata, K, color, depth.astype(np.float32) * depth_scale, mask_selector
                        )

                        tracking_score = evaluate_tracking_score(initial_pose, K, color, mask_selector, estimator, 
                                                                save_pref=osp.join(output_dir, f'mask/initial_{frame_count:06d}'))

                        if success and initial_pose is not None:
                            current_pose = initial_pose
                            pose_initialized = True
                            logging.info("Pose initialization completed successfully")
                        else:
                            logging.warning("Pose initialization failed, will retry next frame")
                            # Continue to next frame to retry
                            continue
                else:
                    print(rgb.shape)
                    new_rgb, new_depth, new_K = resize(rgb, depth, K, MAX_SIZE)
                    current_pose = estimator.track_one(
                        rgb=new_rgb, 
                        depth=new_depth.astype(np.float32) * depth_scale, 
                        K=new_K, 
                        iteration=track_refine_iter
                    )
                    os.makedirs(osp.join(output_dir, f'mask'), exist_ok=True)
                    tracking_score = evaluate_tracking_score(current_pose, K, color, mask_selector, estimator, 
                                                            save_pref=osp.join(output_dir, f'mask/{frame_count:06d}'))
                    
                    # Step 4: Check tracking quality and reinitialize if needed
                    if current_pose is not None:
                        
                        if tracking_score < tracking_score_threshold:
                            logging.warning(f"Tracking score {tracking_score:.3f} below threshold {tracking_score_threshold}, reinitializing...")
                            pose_initialized = False
                        else:
                            logging.info(f"Tracking successful, score: {tracking_score:.3f}")
                    else:
                        logging.warning("Tracking failed, reinitializing...")
                        pose_initialized = False
                        current_pose = None
                        
                    # except Exception as e:
                    #     logging.warning(f"Pose tracking failed: {e}, reinitializing...")
                    #     # Reset to initialization mode
                    #     pose_initialized = False
                    #     current_pose = None
            
            # Save pose if available
            if current_pose is not None and save_poses:
                pose_file = os.path.join(output_dir, 'poses', f'pose_{frame_count:06d}.txt')
                np.savetxt(pose_file, current_pose.reshape(4, 4))
            
            # Create visualization
            vis_img = color.copy()
            if tracking_loop_active:
                if current_pose is not None:
                    vis_img = draw_pose_visualization(
                        vis_img, current_pose, K, metadata['bbox'], metadata['to_origin']
                    )
                    
                    # Add pose info text
                    center_3d = current_pose[:3, 3]
                    cv2.putText(vis_img, f"Pos: [{center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f}]", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(vis_img, f"Frame: {frame_count}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if pose_initialized:
                        cv2.putText(vis_img, f"Tracking Active, score: {tracking_score:.3f}", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(vis_img, "Initializing Pose...", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    if pose_initialized:
                        cv2.putText(vis_img, "Tracking Failed - Reinitializing...", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(vis_img, "Initializing Pose...", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(vis_img, f"Frame: {frame_count}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(vis_img, "Press 'i' to start tracking loop", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis_img, "Press ESC to quit", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save visualization if requested
            if save_visualizations and current_pose is not None:
                vis_file = os.path.join(output_dir, 'visualizations', f'vis_{frame_count:06d}.png')
                cv2.imwrite(vis_file, vis_img)
            
            # Show preview
            if show_preview:
                cv2.imshow("RealSense Pose Estimation", vis_img)
                if cv2.waitKey(1) == 27:  # ESC to quit
                    break
            
            frame_count += 1
            
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        pipe.stop()
        if show_preview:
            cv2.destroyAllWindows()
        logging.info(f"Streaming completed. Processed {frame_count} frames.")
        logging.info(f"Outputs saved to: {output_dir}")


MAX_SIZE = 320
def resize(rgb, depth, K, max_length=640):
    K = K.copy()
    H, W = rgb.shape[:2]
    if H > max_length or W > max_length:
        scale = max_length / max(H, W)
        new_H = int(H * scale)
        new_W = int(W * scale)
        rgb = cv2.resize(rgb, (new_W, new_H))
        depth = cv2.resize(depth, (new_W, new_H))
        K[0,0] = K[0,0] * scale
        K[1,1] = K[1,1] * scale
        K[0,2] = K[0,2] * scale
        K[1,2] = K[1,2] * scale
    return rgb, depth, K

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='RealSense FoundationPose Streaming')
    parser.add_argument('--mesh_file', type=str, 
                       default='./FoundationPose/demo_data/mustard0/mesh/textured_simple.obj',
                       help='Path to object mesh file')
    parser.add_argument('--output_dir', type=str, default='./realsense_output',
                       help='Output directory for poses and visualizations')
    parser.add_argument('--est_refine_iter', type=int, default=5,
                       help='Iterations for initial pose estimation')
    parser.add_argument('--track_refine_iter', type=int, default=2,
                       help='Iterations for pose tracking')
    parser.add_argument('--no_save_poses', action='store_true',
                       help='Disable saving pose files')
    parser.add_argument('--save_vis', action='store_true',
                       help=' saving visualization images')
    parser.add_argument('--no_preview', action='store_true',
                       help='Disable real-time preview')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for SAM2 (cuda:0, cpu, etc.)')
    parser.add_argument('--text_prompt', type=str, default='yellow bottle',
                       help='Text prompt for object detection and segmentation')
    parser.add_argument('--tracking_threshold', type=float, default=0.5,
                       help='Tracking score threshold for reinitialization (0.0-1.0)')
    
    args = parser.parse_args()
    
    stream_pose_estimation(
        mesh_file=args.mesh_file,
        output_dir=args.output_dir,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        save_poses=not args.no_save_poses,
        save_visualizations=args.save_vis,
        show_preview=not args.no_preview,
        device=args.device,
        text_prompt=args.text_prompt,
        tracking_threshold=args.tracking_threshold
    )