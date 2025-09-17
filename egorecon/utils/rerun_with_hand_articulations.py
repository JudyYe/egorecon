#!/usr/bin/env python3
"""Create enhanced Rerun visualization with coordinate transformations and detailed hand articulations"""

import os
import pickle
import numpy as np
import rerun as rr
import torch
import shutil
from pathlib import Path

# Constants for MANO model paths
MANO_LEFT_SRC = "/home/yufeiy2_egorecon/omomo_release/data/mano_models/MANO_LEFT.pkl"
MANO_RIGHT_SRC = "/home/yufeiy2_egorecon/omomo_release/data/mano_models/MANO_RIGHT.pkl"
MANO_TEMP_DIR = "/tmp/mano_models"

def load_mano_models():
    """Load MANO models for hand mesh generation"""
    try:
        import smplx
        
        # Create temporary directory structure that SMPLX expects
        os.makedirs(MANO_TEMP_DIR, exist_ok=True)
        
        # Check if source files exist
        if not os.path.exists(MANO_LEFT_SRC) or not os.path.exists(MANO_RIGHT_SRC):
            raise FileNotFoundError(f"MANO model files not found at {MANO_LEFT_SRC} or {MANO_RIGHT_SRC}")
        
        # Copy MANO model files to expected locations
        mano_files = {
            "MANO_LEFT.pkl": MANO_LEFT_SRC,
            "MANO_RIGHT.pkl": MANO_RIGHT_SRC
        }
        
        for filename, src_path in mano_files.items():
            dst_path = os.path.join(MANO_TEMP_DIR, filename)
            shutil.copy2(src_path, dst_path)
        
        # Load MANO models
        left_mano = smplx.MANO(
            model_path=os.path.join(MANO_TEMP_DIR, "MANO_LEFT.pkl"),
            is_rhand=False,
            use_pca=False,
            flat_hand_mean=True
        )
        
        right_mano = smplx.MANO(
            model_path=os.path.join(MANO_TEMP_DIR, "MANO_RIGHT.pkl"),
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=True
        )
        
        print("✓ MANO models loaded successfully")
        return left_mano, right_mano
        
    except Exception as e:
        print(f"✗ Failed to load MANO models: {e}")
        return None, None

def get_available_object_assets():
    """Get list of available object GLB files"""
    assets_dir = "/home/yufeiy2_egorecon/omomo_release/hot3d/hot3d/dataset/assets"
    if not os.path.exists(assets_dir):
        print(f"Assets directory not found: {assets_dir}")
        return {}
    
    object_assets = {}
    for filename in os.listdir(assets_dir):
        if filename.endswith('.glb'):
            object_uid = filename.replace('.glb', '')
            object_assets[object_uid] = os.path.join(assets_dir, filename)
    
    print(f"✓ Found {len(object_assets)} object GLB files")
    return object_assets

def wxyz_to_rotation_matrix(wxyz_quat):
    """Convert quaternion [w, x, y, z] (WXYZ format) to 3x3 rotation matrix"""
    w, x, y, z = wxyz_quat
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

def apply_coordinate_transform(position, rotation_matrix=None):
    """Apply coordinate system transformation
    
    The HOT3D dataset appears to use a coordinate system where:
    - Objects are oriented vertically (Y-axis is up)
    - We want to rotate them to be horizontal (Z-axis is up)
    
    This transformation rotates the scene 90 degrees around the X-axis
    to convert from Y-up (vertical) to Z-up (horizontal) coordinate system.
    """
    
    # Transform matrix: 90-degree rotation around X-axis
    # This converts from Y-up (vertical) to Z-up (horizontal) coordinate system
    transform_matrix = np.array([
        [1,  0,  0],  # X stays the same
        [0,  1,  0],  # Y becomes -Z (rotated 90° around X)
        [0,  0,  1]   # Z becomes Y (rotated 90° around X)
    ])
    
    # Apply transformation to position
    transformed_position = transform_matrix @ position
    
    # Apply transformation to rotation matrix if provided
    transformed_rotation = None
    if rotation_matrix is not None:
        # R' = T * R * T^T where T is the transform matrix
        transformed_rotation = transform_matrix @ rotation_matrix @ transform_matrix.T
    
    return transformed_position, transformed_rotation

def load_hand_articulations():
    """Load the extracted hand articulation data"""
    file_path = "data/hand_articulations.pkl"
    if not os.path.exists(file_path):
        print(f"⚠️  Hand articulations file not found: {file_path}")
        print("   Will fall back to basic hand poses from generation.pkl")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded hand articulations from {len(data)} sequences")
        return data
    except Exception as e:
        print(f"⚠️  Error loading hand articulations: {e}")
        print("   Will fall back to basic hand poses from generation.pkl")
        return None

def create_enhanced_rerun_file():
    """Create enhanced Rerun file with coordinate transformations and detailed hand articulations"""
    
    # Load MANO models
    left_mano, right_mano = load_mano_models()
    
    # Get available object assets
    object_assets = get_available_object_assets()
    
    # Try to load hand articulations first
    hand_articulations = load_hand_articulations()
    
    # Load basic data as fallback
    with open("/home/yufeiy2_egorecon/omomo_release/data/generation.pkl", 'rb') as f:
        basic_data = pickle.load(f)
    
    # Try to find a sequence that has detailed hand articulations
    detailed_sequence_key = None
    for seq_key in basic_data.keys():
        if hand_articulations and seq_key in hand_articulations:
            detailed_sequence_key = seq_key
            break
    
    # Use detailed sequence if available, otherwise fall back to first
    if detailed_sequence_key:
        first_seq_key = detailed_sequence_key
        print(f"✓ Found sequence with detailed hand articulations: {first_seq_key}")
    else:
        first_seq_key = list(basic_data.keys())[0]
        print(f"⚠️  No detailed hand articulations found, using: {first_seq_key}")
    
    sequence_data = basic_data[first_seq_key]
    print(f"Creating enhanced visualization for: {first_seq_key}")
    
    # Create output
    output_dir = "/home/yufeiy2_egorecon/omomo_release/rerun_files"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{first_seq_key}_enhanced_hands.rrd")
    
    # Initialize rerun
    rr.init(f"ProjectAria_EnhancedHands")
    rr.save(output_file)
    
    # Get basic data
    left_hand_data = sequence_data['left_hand']
    right_hand_data = sequence_data['right_hand']
    object_data = sequence_data['object_pose']
    
    # Check if we have detailed hand articulations for this sequence
    detailed_hands = None
    if hand_articulations and first_seq_key in hand_articulations:
        detailed_hands = hand_articulations[first_seq_key]
        print(f"✓ Using detailed hand articulations with {len(detailed_hands['left_hand'])} left and {len(detailed_hands['right_hand'])} right hand frames")
    else:
        print(f"⚠️  No detailed hand articulations found for {first_seq_key}, using basic poses")
    
    # Determine number of frames to process
    if detailed_hands:
        num_frames = min(500, len(detailed_hands['left_hand']), len(detailed_hands['right_hand']))
    else:
        num_frames = min(500, len(left_hand_data), len(right_hand_data))
    
    print(f"Processing {num_frames} frames with enhanced hand visualization...")
    
    # Track which objects we've loaded as static assets
    loaded_objects = set()
    
    # Collect trajectories
    left_hand_pose_translations = []
    right_hand_pose_translations = []
    
    # Log coordinate system reference (after transformation)
    origin_transformed, _ = apply_coordinate_transform(np.array([0, 0, 0]))
    x_axis_transformed, _ = apply_coordinate_transform(np.array([0.1, 0, 0]))
    y_axis_transformed, _ = apply_coordinate_transform(np.array([0, 0.1, 0]))
    z_axis_transformed, _ = apply_coordinate_transform(np.array([0, 0, 0.1]))
    
    rr.log("world/reference/origin", 
           rr.Points3D([origin_transformed], colors=[[255, 255, 255]], radii=[0.02]), 
           static=True)
    rr.log("world/reference/x_axis", 
           rr.Points3D([x_axis_transformed], colors=[[255, 0, 0]], radii=[0.015]), 
           static=True)
    rr.log("world/reference/y_axis", 
           rr.Points3D([y_axis_transformed], colors=[[0, 255, 0]], radii=[0.015]), 
           static=True)
    rr.log("world/reference/z_axis", 
           rr.Points3D([z_axis_transformed], colors=[[0, 0, 255]], radii=[0.015]), 
           static=True)
    
    # Add a ground plane reference (horizontal plane in transformed coordinates)
    ground_points = []
    for x in np.linspace(-0.5, 0.5, 5):
        for y in np.linspace(-0.5, 0.5, 5):
            # In original coordinates: Y=0 (ground level), X and Z vary
            # After transformation: Z=0 (ground level), X and Y vary
            point, _ = apply_coordinate_transform(np.array([x, 0, y]))
            ground_points.append(point)
    
    rr.log("world/reference/ground_plane", 
           rr.Points3D(ground_points, colors=[[128, 128, 128]], radii=[0.005]), 
           static=True)
    
    # Process frames
    for frame_idx in range(num_frames):
        rr.set_time("frame", sequence=frame_idx)
        
        # === ENHANCED HAND VISUALIZATION ===
        if detailed_hands and left_mano is not None:
            # Use detailed hand articulations
            if frame_idx < len(detailed_hands['left_hand']):
                left_frame = detailed_hands['left_hand'][frame_idx]
                
                # Get detailed landmarks and mesh
                landmarks = left_frame['landmarks_21']  # Shape: [20, 3]
                mesh_vertices = left_frame['mesh_vertices']  # Shape: [778, 3]
                mesh_faces = left_frame['mesh_faces']  # Shape: [1538, 3]
                
                if landmarks is not None and mesh_vertices is not None:
                    # Apply coordinate transformation to landmarks
                    landmarks_transformed = []
                    for landmark in landmarks:
                        transformed_pos, _ = apply_coordinate_transform(landmark)
                        landmarks_transformed.append(transformed_pos)
                    
                    # Apply coordinate transformation to mesh vertices
                    mesh_vertices_transformed = []
                    for vertex in mesh_vertices:
                        transformed_vertex, _ = apply_coordinate_transform(vertex)
                        mesh_vertices_transformed.append(transformed_vertex)
                    
                    # Log detailed hand mesh
                    rr.log(
                        "world/hands/left_detailed_mesh",
                        rr.Mesh3D(
                            vertex_positions=mesh_vertices_transformed,
                            triangle_indices=mesh_faces,
                            vertex_colors=[0.3, 0.8, 0.3]  # Green for left hand
                        ),
                    )
                    
                    # Log hand skeleton (landmarks connected by bones)
                    if 'landmark_connectivity' in detailed_hands:
                        connectivity = detailed_hands['landmark_connectivity']
                        bone_lines = []
                        for connection in connectivity:
                            start_idx, end_idx = connection
                            if start_idx < len(landmarks_transformed) and end_idx < len(landmarks_transformed):
                                bone_lines.append([landmarks_transformed[start_idx], landmarks_transformed[end_idx]])
                        
                        if bone_lines:
                            rr.log(
                                "world/hands/left_skeleton",
                                rr.LineStrips3D(bone_lines, colors=[[0, 255, 0]], radii=[0.003])
                            )
                    
                    # Store position for trajectory
                    if landmarks_transformed:
                        left_hand_pose_translations.append(landmarks_transformed[5])  # Wrist joint
                
            if frame_idx < len(detailed_hands['right_hand']):
                right_frame = detailed_hands['right_hand'][frame_idx]
                
                # Get detailed landmarks and mesh
                landmarks = right_frame['landmarks_21']
                mesh_vertices = right_frame['mesh_vertices']
                mesh_faces = right_frame['mesh_faces']
                
                if landmarks is not None and mesh_vertices is not None:
                    # Apply coordinate transformation
                    landmarks_transformed = []
                    for landmark in landmarks:
                        transformed_pos, _ = apply_coordinate_transform(landmark)
                        landmarks_transformed.append(transformed_pos)
                    
                    mesh_vertices_transformed = []
                    for vertex in mesh_vertices:
                        transformed_vertex, _ = apply_coordinate_transform(vertex)
                        mesh_vertices_transformed.append(transformed_vertex)
                    
                    # Log detailed hand mesh
                    rr.log(
                        "world/hands/right_detailed_mesh",
                        rr.Mesh3D(
                            vertex_positions=mesh_vertices_transformed,
                            triangle_indices=mesh_faces,
                            vertex_colors=[0.3, 0.3, 0.8]  # Blue for right hand
                        ),
                    )
                    
                    # Log hand skeleton
                    if 'landmark_connectivity' in detailed_hands:
                        connectivity = detailed_hands['landmark_connectivity']
                        bone_lines = []
                        for connection in connectivity:
                            start_idx, end_idx = connection
                            if start_idx < len(landmarks_transformed) and end_idx < len(landmarks_transformed):
                                bone_lines.append([landmarks_transformed[start_idx], landmarks_transformed[end_idx]])
                        
                        if bone_lines:
                            rr.log(
                                "world/hands/right_skeleton",
                                rr.LineStrips3D(bone_lines, colors=[[0, 0, 255]], radii=[0.003])
                            )
                    
                    # Store position for trajectory
                    if landmarks_transformed:
                        right_hand_pose_translations.append(landmarks_transformed[5])  # Wrist joint
        
        else:
            # Fallback to basic hand visualization
            if frame_idx < len(left_hand_data) and left_mano is not None:
                left_frame = left_hand_data[frame_idx]
                left_translation = np.array(left_frame['translation'])
                left_rotation_wxyz = left_frame['rotation'][0]  # WXYZ format
                
                # Apply coordinate transformation
                left_translation_transformed, left_rotation_transformed = apply_coordinate_transform(
                    left_translation, wxyz_to_rotation_matrix(left_rotation_wxyz)
                )
                
                left_hand_pose_translations.append(left_translation_transformed)
                
                # Generate MANO mesh
                with torch.no_grad():
                    left_output = left_mano()
                    hand_mesh_vertices = left_output.vertices[0].numpy()
                    hand_triangles = left_mano.faces
                
                # Apply transformation to vertices
                if left_rotation_transformed is not None:
                    hand_mesh_vertices_transformed = (left_rotation_transformed @ hand_mesh_vertices.T).T + left_translation_transformed
                else:
                    hand_mesh_vertices_transformed = hand_mesh_vertices + left_translation_transformed
                
                # Log mesh
                rr.log(
                    "world/hands/left_basic_mesh",
                    rr.Mesh3D(
                        vertex_positions=hand_mesh_vertices_transformed,
                        triangle_indices=hand_triangles,
                        vertex_colors=[0.3, 0.8, 0.3]  # Green for left hand
                    ),
                )
            
            if frame_idx < len(right_hand_data) and right_mano is not None:
                right_frame = right_hand_data[frame_idx]
                right_translation = np.array(right_frame['translation'])
                right_rotation_wxyz = right_frame['rotation'][0]  # WXYZ format
                
                # Apply coordinate transformation
                right_translation_transformed, right_rotation_transformed = apply_coordinate_transform(
                    right_translation, wxyz_to_rotation_matrix(right_rotation_wxyz)
                )
                
                right_hand_pose_translations.append(right_translation_transformed)
                
                # Generate MANO mesh
                with torch.no_grad():
                    right_output = right_mano()
                    hand_mesh_vertices = right_output.vertices[0].numpy()
                    hand_triangles = right_mano.faces
                
                # Apply transformation to vertices
                if right_rotation_transformed is not None:
                    hand_mesh_vertices_transformed = (right_rotation_transformed @ hand_mesh_vertices.T).T + right_translation_transformed
                else:
                    hand_mesh_vertices_transformed = hand_mesh_vertices + right_translation_transformed
                
                # Log mesh
                rr.log(
                    "world/hands/right_basic_mesh",
                    rr.Mesh3D(
                        vertex_positions=hand_mesh_vertices_transformed,
                        triangle_indices=hand_triangles,
                        vertex_colors=[0.3, 0.3, 0.8]  # Blue for right hand
                    ),
                )
        
        # === OBJECTS WITH COORDINATE TRANSFORMATION ===
        if frame_idx < len(object_data) and len(object_data[frame_idx]['poses']) > 0:
            for obj_data in object_data[frame_idx]['poses']:
                object_uid = obj_data['object_uid']
                obj_translation = np.array(obj_data['translation'])
                obj_rotation_wxyz = obj_data['rotation'][0]  # WXYZ format
                
                object_name = f"object_{object_uid}"
                
                # Load the 3D asset if we haven't already and if it exists
                if object_uid not in loaded_objects and object_uid in object_assets:
                    glb_path = object_assets[object_uid]
                    rr.log(
                        f"world/objects/{object_name}",
                        rr.Asset3D(path=glb_path),
                        static=True
                    )
                    loaded_objects.add(object_uid)
                    print(f"  ✓ Loaded 3D asset for object {object_uid}")
                
                # Convert WXYZ quaternion to rotation matrix and apply coordinate transformation
                obj_rotation_matrix = wxyz_to_rotation_matrix(obj_rotation_wxyz)
                obj_translation_transformed, obj_rotation_transformed = apply_coordinate_transform(
                    obj_translation, obj_rotation_matrix
                )
                
                # Use rotation matrix directly with mat3x3
                rr.log(
                    f"world/objects/{object_name}",
                    rr.Transform3D(
                        translation=obj_translation_transformed,
                        mat3x3=obj_rotation_transformed
                    )
                )
    
    # Log hand trajectories
    if left_hand_pose_translations:
        rr.log("world/trajectories/left_hand", 
               rr.LineStrips3D([left_hand_pose_translations], colors=[[0, 255, 0]]), 
               static=True)
    
    if right_hand_pose_translations:
        rr.log("world/trajectories/right_hand", 
               rr.LineStrips3D([right_hand_pose_translations], colors=[[0, 0, 255]]), 
               static=True)
    
    print(f"✓ Created enhanced rerun file: {output_file}")
    
    # Check file size
    file_size = os.path.getsize(output_file)
    print(f"File size: {file_size} bytes ({file_size/1024/1024:.1f} MB)")
    
    has_mano = left_mano is not None and right_mano is not None
    has_detailed_hands = detailed_hands is not None
    print(f"\nEnhanced scene includes:")
    print(f"- {'✓' if has_mano else '✗'} MANO hand models")
    print(f"- {'✓' if has_detailed_hands else '✗'} Detailed hand articulations with individual finger movements")
    print(f"- ✓ {len(loaded_objects)} 3D object assets with transformed positions and rotations")
    print(f"- ✓ Ground plane reference (gray dots)")
    print(f"- ✓ Transformed coordinate system (X=red, Y=green, Z=blue)")
    print(f"- ✓ Objects use rotation matrices (mat3x3) for precise control")
    
    if has_detailed_hands:
        print(f"\n🎯 Hand Articulation Features:")
        print(f"- Individual finger joint movements (20 landmarks per hand)")
        print(f"- Detailed hand mesh with 778 vertices and 1,538 faces")
        print(f"- Hand skeleton visualization with bone connections")
    
    return output_file

if __name__ == "__main__":
    output_file = create_enhanced_rerun_file()
    print(f"\nTo download and view:")
    print(f"scp egorecon:{output_file} ./")
    print(f"rerun {os.path.basename(output_file)} --web-viewer --port 9877")
    print(f"\nEnhanced visualization features:")
    print(f"- Detailed hand articulations with individual finger movements")
    print(f"- Rich hand mesh geometry (778 vertices per hand)")