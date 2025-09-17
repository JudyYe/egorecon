import os
import numpy as np
import torch
import trimesh
from src.utils.rotation_utils import transforms
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Import existing components
try:
    from human_body_prior.body_model.body_model import BodyModel
    SMPLX_AVAILABLE = True
except ImportError:
    print("Warning: SMPL-X not available. Full body visualization will be limited.")
    SMPLX_AVAILABLE = False

# Import MANO components if available
try:
    # Try different MANO import paths
    try:
        from hot3d.data_loaders.mano_layer import MANOHandModel
    except ImportError:
        # Fallback for direct SMPLX installation
        import smplx
        MANOHandModel = None  # We'll use smplx directly
    MANO_AVAILABLE = True
except ImportError:
    print("Warning: MANO not available. Hand visualization will be limited.")
    MANO_AVAILABLE = False
    MANOHandModel = None

from manip.vis.blender_vis_mesh_motion import (
    run_blender_rendering_and_save2video,
    save_verts_faces_to_mesh_file_w_object
)


class TrajectoryMeshVisualizer:
    """
    Comprehensive visualizer that converts hand and object trajectories to meshes
    and renders them using the existing Blender pipeline.
    """
    
    def __init__(self, 
                 data_root_folder: str = "data",
                 smplx_model_path: Optional[str] = None,
                 mano_model_path: Optional[str] = None,
                 object_geometry_path: Optional[str] = None):
        """
        Initialize the trajectory mesh visualizer.
        
        Args:
            data_root_folder: Root folder for data files
            smplx_model_path: Path to SMPL-X models
            mano_model_path: Path to MANO models  
            object_geometry_path: Path to object geometry files
        """
        self.data_root_folder = data_root_folder
        self.smplx_model_path = smplx_model_path or os.path.join(data_root_folder, "smpl_all_models")
        self.mano_model_path = mano_model_path
        self.object_geometry_path = object_geometry_path or os.path.join(data_root_folder, "objects_geometry")
        
        # Initialize models
        self._setup_body_models()
        self._setup_hand_models()
        self._setup_object_library()
        
    def _setup_body_models(self):
        """Setup SMPL-X body models for full body mesh generation."""
        self.body_models = {}
        
        if SMPLX_AVAILABLE and os.path.exists(self.smplx_model_path):
            try:
                # Setup male and female models
                for gender in ['male', 'female']:
                    model_path = os.path.join(self.smplx_model_path, f"SMPLX_{gender.upper()}.npz")
                    if os.path.exists(model_path):
                        self.body_models[gender] = BodyModel(
                            bm_fname=model_path,
                            num_betas=16,
                            num_expressions=None,
                            num_dmpls=None,
                            dmpl_fname=None
                        ).cuda()
                        
                        # Freeze parameters
                        for p in self.body_models[gender].parameters():
                            p.requires_grad = False
                            
                print(f"Loaded SMPL-X models: {list(self.body_models.keys())}")
            except Exception as e:
                print(f"Failed to load SMPL-X models: {e}")
                self.body_models = {}
        
    def _setup_hand_models(self):
        """Setup MANO hand models for detailed hand mesh generation."""
        self.hand_models = {}
        
        if MANO_AVAILABLE and self.mano_model_path and os.path.exists(self.mano_model_path):
            try:
                if MANOHandModel is not None:
                    self.hand_models['mano'] = MANOHandModel(
                        mano_model_files_dir=self.mano_model_path
                    )
                    print("Loaded MANO hand models")
                else:
                    # Use direct SMPLX for MANO models
                    import smplx
                    self.hand_models['smplx_mano'] = {
                        'left': smplx.create(os.path.join(self.mano_model_path, 'MANO_LEFT.pkl'), 'mano', is_rhand=False),
                        'right': smplx.create(os.path.join(self.mano_model_path, 'MANO_RIGHT.pkl'), 'mano', is_rhand=True)
                    }
                    print("Loaded MANO hand models via SMPLX")
            except Exception as e:
                print(f"Failed to load MANO models: {e}")
                
    def _setup_object_library(self):
        """Setup HOT3D object library for mapping object names to IDs."""
        self.object_library = {}
        
        # Try to load HOT3D object library
        instance_path = os.path.join(self.object_geometry_path, "instance.json")
        if os.path.exists(instance_path):
            try:
                import json
                with open(instance_path, 'r') as f:
                    instance_data = json.load(f)
                
                # Create mapping from object names to IDs
                if 'objects' in instance_data:
                    for obj_id, obj_info in instance_data['objects'].items():
                        if 'name' in obj_info:
                            object_name = obj_info['name']
                            self.object_library[object_name] = obj_id
                            # Also add lowercase version
                            self.object_library[object_name.lower()] = obj_id
                
                print(f"Loaded HOT3D object library with {len(self.object_library)} objects")
            except Exception as e:
                print(f"Failed to load HOT3D object library: {e}")
        else:
            print("HOT3D object library not found, using filename-based object loading")
                
    def trajectory_to_hand_meshes(self, 
                                  hand_trajectories: np.ndarray,
                                  hand_type: str = "position_only",
                                  use_mano: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert hand trajectories to hand meshes.
        
        Args:
            hand_trajectories: [T, 2, D] - left and right hand trajectories
            hand_type: Type of hand data ("position_only", "pose_9d", "pose_12d")
            use_mano: Whether to use MANO for detailed hand meshes
            
        Returns:
            Tuple of (vertices, faces) for hand meshes
        """
        T = hand_trajectories.shape[0]
        
        if use_mano and ('mano' in self.hand_models or 'smplx_mano' in self.hand_models):
            return self._generate_mano_hand_meshes(hand_trajectories, hand_type)
        else:
            return self._generate_simple_hand_meshes(hand_trajectories, hand_type)
    
    def _generate_mano_hand_meshes(self, 
                                   hand_trajectories: np.ndarray,
                                   hand_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate detailed MANO hand meshes."""
        T = hand_trajectories.shape[0]
        
        # Check which MANO model type we have
        if 'mano' in self.hand_models:
            mano_model = self.hand_models['mano']
            use_hot3d_mano = True
        elif 'smplx_mano' in self.hand_models:
            mano_models = self.hand_models['smplx_mano']
            use_hot3d_mano = False
        else:
            return self._generate_simple_hand_meshes(hand_trajectories, hand_type)
        
        all_vertices = []
        faces = None
        
        for t in range(T):
            frame_vertices = []
            
            for hand_idx in range(2):  # Left (0) and right (1) hands
                hand_data = hand_trajectories[t, hand_idx]
                
                # Extract position and pose from trajectory data
                if hand_type == "position_only":
                    # Use default pose with given position
                    position = hand_data[:3]
                    pose_params = torch.zeros(15)  # Default hand pose
                    global_orient = torch.zeros(3)  # Default orientation
                elif hand_type == "pose_9d":
                    position = hand_data[:3]
                    rotation_6d = hand_data[3:9]
                    # Convert 6D rotation to axis-angle
                    rotation_matrix = transforms.rotation_6d_to_matrix(
                        torch.tensor(rotation_6d).unsqueeze(0)
                    )
                    global_orient = transforms.matrix_to_axis_angle(rotation_matrix).squeeze(0)
                    pose_params = torch.zeros(15)  # Default finger poses
                elif hand_type == "pose_12d":
                    position = hand_data[:3]
                    rotation_6d = hand_data[6:12]  # Skip velocity
                    rotation_matrix = transforms.rotation_6d_to_matrix(
                        torch.tensor(rotation_6d).unsqueeze(0)
                    )
                    global_orient = transforms.matrix_to_axis_angle(rotation_matrix).squeeze(0)
                    pose_params = torch.zeros(15)
                
                # Generate hand mesh using MANO
                if use_hot3d_mano:
                    # Use HOT3D MANO wrapper
                    global_xform = torch.cat([global_orient, torch.tensor(position)])
                    shape_params = torch.zeros(10)  # Default hand shape
                    
                    vertices, landmarks = mano_model.forward_kinematics(
                        shape_params=shape_params,
                        joint_angles=pose_params.unsqueeze(0),
                        global_xfrom=global_xform.unsqueeze(0),
                        is_right_hand=torch.tensor([hand_idx == 1])
                    )
                    
                    frame_vertices.append(vertices[0].cpu().numpy())
                    
                    # Get faces (same for all frames)
                    if faces is None:
                        if hand_idx == 0:
                            faces = mano_model.mano_layer_left.faces
                        else:
                            faces = mano_model.mano_layer_right.faces
                else:
                    # Use direct SMPLX MANO
                    hand_side = 'left' if hand_idx == 0 else 'right'
                    mano_layer = mano_models[hand_side]
                    
                    # Prepare parameters
                    betas = torch.zeros(1, 10)  # Hand shape
                    hand_pose = pose_params.unsqueeze(0)  # Hand pose
                    global_orient_batch = global_orient.unsqueeze(0)  # Global orientation
                    transl = torch.tensor(position).unsqueeze(0)  # Translation
                    
                    # Generate mesh
                    output = mano_layer(
                        betas=betas,
                        hand_pose=hand_pose,
                        global_orient=global_orient_batch,
                        transl=transl,
                        return_verts=True
                    )
                    
                    frame_vertices.append(output.vertices[0].cpu().numpy())
                    
                    # Get faces (same for all frames)
                    if faces is None:
                        faces = mano_layer.faces
            
            all_vertices.append(np.stack(frame_vertices))
        
        return np.array(all_vertices), faces
    
    def _generate_simple_hand_meshes(self, 
                                     hand_trajectories: np.ndarray,
                                     hand_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simple hand meshes using basic shapes."""
        T = hand_trajectories.shape[0]
        
        # Create simple hand representation using spheres or boxes
        hand_vertices = []
        
        for t in range(T):
            frame_vertices = []
            for hand_idx in range(2):
                hand_data = hand_trajectories[t, hand_idx]
                position = hand_data[:3]
                
                # Create a simple hand mesh (box or sphere)
                hand_mesh = trimesh.creation.box(extents=[0.08, 0.12, 0.04])
                hand_mesh.vertices += position
                
                frame_vertices.append(hand_mesh.vertices)
            
            hand_vertices.append(frame_vertices)
        
        # Use faces from the first mesh
        faces = hand_mesh.faces
        
        return np.array(hand_vertices), faces
    
    def trajectory_to_object_meshes(self, 
                                    object_trajectories: np.ndarray,
                                    object_name: str,
                                    object_geometry_data: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert object trajectories to object meshes.
        
        Args:
            object_trajectories: [T, D] - object trajectory data
            object_name: Name of the object
            object_geometry_data: Optional pre-loaded object geometry
            
        Returns:
            Tuple of (vertices, faces) for object meshes
        """
        T = object_trajectories.shape[0]
        
        # Load base object geometry
        if object_geometry_data is None:
            object_geometry_data = self._load_object_geometry(object_name)
        
        base_vertices = object_geometry_data['vertices']
        faces = object_geometry_data['faces']
        
        all_vertices = []
        
        for t in range(T):
            traj_data = object_trajectories[t]
            
            # Extract transformation from trajectory
            position = traj_data[:3]
            
            if len(traj_data) >= 9:  # Has rotation information
                if len(traj_data) == 9:  # 9D format: pos(3) + rot(6)
                    rotation_6d = traj_data[3:9]
                elif len(traj_data) == 12:  # 12D format: pos(3) + vel(3) + rot(6)
                    rotation_6d = traj_data[6:12]
                else:
                    rotation_6d = np.array([1, 0, 0, 0, 1, 0])  # Default rotation
                
                # Convert 6D rotation to matrix
                rotation_6d_tensor = torch.tensor(rotation_6d).unsqueeze(0)
                rotation_matrix = transforms.rotation_6d_to_matrix(rotation_6d_tensor)[0].numpy()
            else:
                rotation_matrix = np.eye(3)  # No rotation
            
            # Apply transformation to base vertices
            transformed_vertices = np.dot(base_vertices, rotation_matrix.T) + position
            all_vertices.append(transformed_vertices)
        
        return np.array(all_vertices), faces
    
    def _load_object_geometry(self, object_name: str) -> Dict:
        """Load object geometry from file."""
        # Check if we have a HOT3D object library mapping
        object_id = object_name
        if hasattr(self, 'object_library') and object_name in self.object_library:
            object_id = self.object_library[object_name]
            print(f"Mapped object '{object_name}' to ID '{object_id}'")
        elif hasattr(self, 'object_library') and object_name.lower() in self.object_library:
            object_id = self.object_library[object_name.lower()]
            print(f"Mapped object '{object_name}' to ID '{object_id}'")
        
        # Try different file extensions and paths
        possible_paths = [
            # HOT3D .glb format (using mapped object ID)
            os.path.join(self.object_geometry_path, f"{object_id}.glb"),
            # Traditional formats (using original object name)
            os.path.join(self.object_geometry_path, f"{object_name}_cleaned_simplified.obj"),
            os.path.join(self.object_geometry_path, f"{object_name}.obj"),
            os.path.join(self.object_geometry_path, f"{object_name}.ply"),
        ]
        
        for obj_path in possible_paths:
            if os.path.exists(obj_path):
                try:
                    if obj_path.endswith('.glb'):
                        # Load GLB file and convert to mesh
                        scene = trimesh.load_mesh(obj_path, process=True, merge_primitives=True, file_type="glb")
                        if hasattr(scene, 'to_mesh'):
                            mesh = scene.to_mesh()
                        else:
                            mesh = scene
                    else:
                        mesh = trimesh.load_mesh(obj_path)
                    
                    return {
                        'vertices': np.array(mesh.vertices),
                        'faces': np.array(mesh.faces)
                    }
                except Exception as e:
                    print(f"Failed to load {obj_path}: {e}")
                    continue
        
        # Fallback: create a simple box
        print(f"Could not load geometry for {object_name}, using default box")
        box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        return {
            'vertices': np.array(box.vertices),
            'faces': np.array(box.faces)
        }
    
    def visualize_trajectories(self, 
                               hand_trajectories: np.ndarray,
                               object_trajectories: np.ndarray,
                               object_name: str,
                               output_dir: str,
                               sequence_name: str = "trajectory",
                               hand_type: str = "pose_9d",
                               render_video: bool = True,
                               vis_hands: bool = True,
                               vis_objects: bool = True) -> str:
        """
        Complete pipeline to visualize hand and object trajectories.
        
        Args:
            hand_trajectories: [T, 2, D] - left and right hand trajectories
            object_trajectories: [T, D] - object trajectory
            object_name: Name of the object
            output_dir: Output directory for results
            sequence_name: Name for this sequence
            hand_type: Type of hand trajectory data
            render_video: Whether to render final video
            vis_hands: Whether to visualize hands
            vis_objects: Whether to visualize objects
            
        Returns:
            Path to the generated video file
        """
        print(f"Visualizing trajectories for {sequence_name}")
        
        # Create output directories
        mesh_dir = os.path.join(output_dir, "meshes", sequence_name)
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        
        T = hand_trajectories.shape[0]
        
        # Generate meshes
        hand_vertices, hand_faces = None, None
        object_vertices, object_faces = None, None
        
        if vis_hands and hand_trajectories is not None:
            print("Generating hand meshes...")
            hand_vertices, hand_faces = self.trajectory_to_hand_meshes(
                hand_trajectories, hand_type=hand_type
            )
        
        if vis_objects and object_trajectories is not None:
            print("Generating object meshes...")
            object_vertices, object_faces = self.trajectory_to_object_meshes(
                object_trajectories, object_name
            )
        
        # Save meshes to files for Blender rendering
        print("Saving mesh files...")
        for t in range(T):
            frame_name = f"{t:05d}"
            
            if vis_hands and hand_vertices is not None:
                # Save left and right hands separately or combined
                if hand_vertices.shape[1] == 2:  # Separate left/right hands
                    # Combine left and right hand vertices
                    combined_vertices = np.vstack([
                        hand_vertices[t, 0],  # Left hand
                        hand_vertices[t, 1]   # Right hand
                    ])
                    combined_faces = np.vstack([
                        hand_faces,
                        hand_faces + len(hand_vertices[t, 0])  # Offset for right hand
                    ])
                else:
                    combined_vertices = hand_vertices[t]
                    combined_faces = hand_faces
                
                # Save human mesh
                human_mesh = trimesh.Trimesh(
                    vertices=combined_vertices,
                    faces=combined_faces
                )
                human_mesh.export(os.path.join(mesh_dir, f"{frame_name}.ply"))
            
            if vis_objects and object_vertices is not None:
                # Save object mesh
                object_mesh = trimesh.Trimesh(
                    vertices=object_vertices[t],
                    faces=object_faces
                )
                object_mesh.export(os.path.join(mesh_dir, f"{frame_name}_object.ply"))
        
        # Render video using existing Blender pipeline
        if render_video:
            print("Rendering video with Blender...")
            video_path = os.path.join(video_dir, f"{sequence_name}.mp4")
            
            run_blender_rendering_and_save2video(
                obj_folder_path=mesh_dir,
                out_folder_path=os.path.join(output_dir, "frames", sequence_name),
                out_vid_path=video_path,
                vis_object=vis_objects,
                vis_human=vis_hands,
                vis_hand_and_object=vis_hands and vis_objects
            )
            
            print(f"Video saved to: {video_path}")
            return video_path
        
        return mesh_dir
    
    def visualize_from_trainer_results(self, 
                                       results_dir: str,
                                       sequence_name: str = "generated",
                                       render_video: bool = True) -> str:
        """
        Visualize results directly from trainer output files.
        
        Args:
            results_dir: Directory containing trainer results
            sequence_name: Name for this visualization
            render_video: Whether to render video
            
        Returns:
            Path to generated content
        """
        print(f"Loading trainer results from {results_dir}")
        
        # Find result files
        result_files = list(Path(results_dir).glob("*.npy")) + list(Path(results_dir).glob("*.npz"))
        
        hand_trajectories = None
        object_trajectories = None
        object_name = "unknown"
        
        for file_path in result_files:
            if "hands_" in file_path.name:
                # Load hand data
                hand_data = np.load(file_path)
                if isinstance(hand_data, np.lib.npyio.NpzFile):
                    left_hand = hand_data['left_hand']
                    right_hand = hand_data['right_hand']
                    hand_trajectories = np.stack([left_hand, right_hand], axis=1)  # [T, 2, D]
                else:
                    hand_trajectories = hand_data
                
                # Extract object name from filename
                parts = file_path.name.split('_')
                if len(parts) >= 3:
                    object_name = parts[2]
                    
            elif "sampled_" in file_path.name:
                # Load sampled object trajectory
                object_trajectories = np.load(file_path)
                
                # Extract object name from filename
                parts = file_path.name.split('_')
                if len(parts) >= 3:
                    object_name = parts[2]
        
        if hand_trajectories is None and object_trajectories is None:
            raise ValueError(f"No valid trajectory data found in {results_dir}")
        
        # Visualize trajectories
        output_dir = os.path.join(results_dir, "visualization")
        
        return self.visualize_trajectories(
            hand_trajectories=hand_trajectories,
            object_trajectories=object_trajectories,
            object_name=object_name,
            output_dir=output_dir,
            sequence_name=sequence_name,
            render_video=render_video
        )


def create_visualizer_from_config(config_path: Optional[str] = None) -> TrajectoryMeshVisualizer:
    """Create a visualizer with default or config-based settings."""
    
    # Default paths - adjust these for your setup
    default_config = {
        'data_root_folder': 'data',
        'smplx_model_path': 'data/smpl_all_models',
        'mano_model_path': 'data/mano_models',  # Set this if you have MANO models
        'object_geometry_path': 'data/objects_geometry'
    }
    
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = default_config
    
    return TrajectoryMeshVisualizer(**config)


# Convenience function for quick visualization
def visualize_trajectory_results(results_dir: str, 
                                 sequence_name: str = "demo",
                                 config_path: Optional[str] = None) -> str:
    """
    Quick function to visualize trajectory results.
    
    Args:
        results_dir: Directory with trajectory results from trainer
        sequence_name: Name for output files
        config_path: Optional config file path
        
    Returns:
        Path to generated video or mesh directory
    """
    visualizer = create_visualizer_from_config(config_path)
    return visualizer.visualize_from_trainer_results(
        results_dir=results_dir,
        sequence_name=sequence_name,
        render_video=True
    ) 