#!/usr/bin/env python3
"""Extract detailed hand articulations from HOT3D dataset"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

hot3d_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "hot3d", "hot3d")
sys.path.append(hot3d_path)

try:
    from dataset_api import Hot3dDataProvider
    from data_loaders.loader_object_library import load_object_library
    from data_loaders.mano_layer import MANOHandModel
    from data_loaders.loader_hand_poses import HandType, Handedness
    from data_loaders.hand_common import LANDMARK_INDEX_TO_NAMING, LANDMARK_CONNECTIVITY
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    print("✓ Successfully imported HOT3D modules")
except ImportError as e:
    print(f"❌ Error importing HOT3D modules: {e}")
    print("Please ensure you're running from the correct directory and using pixi")
    sys.exit(1)

def load_mano_models():
    """Load MANO models for hand articulation extraction"""
    try:
        mano_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "mano_models")
        if not os.path.exists(mano_dir):
            print(f"❌ MANO models directory not found: {mano_dir}")
            return None, None
        
        # Check if MANO models exist
        left_model_path = os.path.join(mano_dir, "MANO_LEFT.pkl")
        right_model_path = os.path.join(mano_dir, "MANO_RIGHT.pkl")
        
        if not os.path.exists(left_model_path) or not os.path.exists(right_model_path):
            print(f"❌ MANO model files not found in {mano_dir}")
            return None, None
        
        # Load MANO models
        left_mano = MANOHandModel(mano_dir)
        right_mano = MANOHandModel(mano_dir)
        
        print("✓ MANO models loaded successfully")
        return left_mano, right_mano
        
    except Exception as e:
        print(f"❌ Failed to load MANO models: {e}")
        return None, None

def extract_hand_articulations_from_sequence(sequence_folder, object_library_folder, mano_hand_model=None):
    """
    Extract detailed hand articulations from a HOT3D sequence.
    
    Args:
        sequence_folder: Path to the sequence folder
        object_library_folder: Path to the object library folder
        mano_hand_model: Optional MANO hand model for enhanced articulation
    
    Returns:
        dict with detailed hand articulation data
    """
    
    print(f"\n🔍 Processing sequence: {os.path.basename(sequence_folder)}")
    
    # Validate paths
    if not os.path.exists(sequence_folder):
        raise RuntimeError(f"Sequence folder {sequence_folder} does not exist")
    if not os.path.exists(object_library_folder):
        raise RuntimeError(f"Object library folder {object_library_folder} does not exist")
    
    # Load object library
    print("Loading object library...")
    object_library = load_object_library(object_library_folderpath=object_library_folder)
    
    # Initialize data provider
    print("Initializing HOT3D data provider...")
    data_provider = Hot3dDataProvider(
        sequence_folder=sequence_folder,
        object_library=object_library,
        mano_hand_model=mano_hand_model,
        fail_on_missing_data=False,
    )
    
    print(f"Data provider statistics: {data_provider.get_data_statistics()}")
    
    # Get hand data provider (prefer MANO if available, otherwise UmeTrack)
    hand_data_provider = None
    hand_model_type = "Unknown"
    
    if data_provider.mano_hand_data_provider is not None:
        hand_data_provider = data_provider.mano_hand_data_provider
        hand_model_type = "MANO"
        print("✓ Using MANO hand data provider")
    elif data_provider.umetrack_hand_data_provider is not None:
        hand_data_provider = data_provider.umetrack_hand_data_provider
        hand_model_type = "UmeTrack"
        print("✓ Using UmeTrack hand data provider")
    else:
        raise RuntimeError("No hand data provider available")
    
    # Get timestamps
    timestamps = data_provider.device_data_provider.get_sequence_timestamps()
    print(f"Found {len(timestamps)} timestamps")
    
    # Initialize storage for detailed hand data
    left_hand_data = []
    right_hand_data = []
    
    # Process each timestamp
    print(f"\nExtracting hand articulations using {hand_model_type} model...")
    for timestamp_ns in tqdm(timestamps, desc="Processing frames"):
        
        # Get hand poses for this timestamp
        hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        
        if hand_poses_with_dt is None:
            continue
        
        hand_pose_collection = hand_poses_with_dt.pose3d_collection
        
        # Process each hand
        for hand_pose_data in hand_pose_collection.poses.values():
            
            # Get detailed hand landmarks (21 joint positions)
            hand_landmarks = hand_data_provider.get_hand_landmarks(hand_pose_data)
            
            if hand_landmarks is not None:
                # Convert landmarks to numpy array
                landmarks_array = hand_landmarks.numpy() if hasattr(hand_landmarks, 'numpy') else np.array(hand_landmarks)
                
                # Get hand mesh vertices for additional detail
                hand_mesh_vertices = hand_data_provider.get_hand_mesh_vertices(hand_pose_data)
                mesh_vertices = None
                if hand_mesh_vertices is not None:
                    mesh_vertices = hand_mesh_vertices.numpy() if hasattr(hand_mesh_vertices, 'numpy') else np.array(hand_mesh_vertices)
                
                # Get hand mesh faces and normals
                mesh_faces_normals = hand_data_provider.get_hand_mesh_faces_and_normals(hand_pose_data)
                mesh_faces = None
                mesh_normals = None
                if mesh_faces_normals is not None:
                    mesh_faces, mesh_normals = mesh_faces_normals
                
                # Create detailed frame data
                frame_data = {
                    'timestamp_ns': timestamp_ns,
                    'handedness': hand_pose_data.handedness_label(),
                    'wrist_pose': {
                        'translation': hand_pose_data.wrist_pose.translation()[0] if hand_pose_data.wrist_pose else None,
                        'rotation': hand_pose_data.wrist_pose.rotation().to_matrix() if hand_pose_data.wrist_pose else None,
                    },
                    'joint_angles': hand_pose_data.joint_angles if hasattr(hand_pose_data, 'joint_angles') else None,
                    'landmarks_21': landmarks_array,  # 21 joint positions
                    'mesh_vertices': mesh_vertices,
                    'mesh_faces': mesh_faces,
                    'mesh_normals': mesh_normals,
                    'hand_model_type': hand_model_type
                }
                
                # Store based on handedness
                if hand_pose_data.is_left_hand():
                    left_hand_data.append(frame_data)
                elif hand_pose_data.is_right_hand():
                    right_hand_data.append(frame_data)
    
    print(f"✓ Extracted {len(left_hand_data)} left hand frames")
    print(f"✓ Extracted {len(right_hand_data)} right hand frames")
    
    return {
        'sequence_name': os.path.basename(sequence_folder),
        'hand_model_type': hand_model_type,
        'total_frames': len(timestamps),
        'left_hand': left_hand_data,
        'right_hand': right_hand_data,
        'landmark_names': [landmark.value for landmark in LANDMARK_INDEX_TO_NAMING],
        'landmark_connectivity': LANDMARK_CONNECTIVITY
    }

def main():
    """Main function to extract hand articulations from HOT3D dataset"""
    
    print("🖐️  HOT3D Hand Articulation Extractor")
    print("=" * 50)
    
    # Configuration
    sequences_folder = "hot3d/hot3d/dataset"
    object_library_folder = "hot3d/hot3d/dataset/assets"
    output_file = "data/hand_articulations.pkl"
    
    # Check if directories exist
    if not os.path.exists(sequences_folder):
        print(f"❌ Sequences folder not found: {sequences_folder}")
        return
    
    if not os.path.exists(object_library_folder):
        print(f"❌ Object library folder not found: {object_library_folder}")
        return
    
    # Load MANO models if available
    mano_hand_model = None
    try:
        left_mano, right_mano = load_mano_models()
        if left_mano is not None and right_mano is not None:
            mano_hand_model = left_mano  # Use left as default, both will be available
    except Exception as e:
        print(f"⚠️  MANO models not available: {e}")
        print("Will use UmeTrack hand model instead")
    
    # Get list of sequences
    sequence_dirs = [d for d in os.listdir(sequences_folder) 
                    if os.path.isdir(os.path.join(sequences_folder, d)) and d.startswith('P')]
    
    print(f"Found {len(sequence_dirs)} sequences")
    
    # Process first few sequences for testing
    max_sequences = 300  # Limit for testing
    processed_sequences = {}
    
    for i, seq_dir in enumerate(sequence_dirs[:max_sequences]):
        sequence_path = os.path.join(sequences_folder, seq_dir)
        
        try:
            print(f"\n{'='*60}")
            print(f"Processing sequence {i+1}/{min(max_sequences, len(sequence_dirs))}: {seq_dir}")
            print(f"{'='*60}")
            
            # Extract hand articulations
            sequence_data = extract_hand_articulations_from_sequence(
                sequence_path, 
                object_library_folder, 
                mano_hand_model
            )
            
            processed_sequences[seq_dir] = sequence_data
            
        except Exception as e:
            print(f"❌ Error processing sequence {seq_dir}: {e}")
            continue
    
    # Save results
    if processed_sequences:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(processed_sequences, f)
        
        print(f"\n✅ Successfully extracted hand articulations from {len(processed_sequences)} sequences")
        print(f"📁 Saved to: {output_file}")
        
        # Show summary
        print(f"\n📊 Summary:")
        for seq_name, seq_data in processed_sequences.items():
            print(f"  {seq_name}:")
            print(f"    - Left hand: {len(seq_data['left_hand'])} frames")
            print(f"    - Right hand: {len(seq_data['right_hand'])} frames")
            print(f"    - Model: {seq_data['hand_model_type']}")
            print(f"    - Landmarks: {len(seq_data['landmark_names'])} joints")
        
        print(f"\n🔍 Landmark names:")
        for i, name in enumerate(processed_sequences[list(processed_sequences.keys())[0]]['landmark_names']):
            print(f"  {i:2d}: {name}")
            
    else:
        print("❌ No sequences were processed successfully")

if __name__ == "__main__":
    main()