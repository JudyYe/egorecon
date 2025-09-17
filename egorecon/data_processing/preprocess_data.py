#!/usr/bin/env python3
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from ..utils.quaternion import quat_to_6v
import os

def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def convert_trajectory_to_6d(trajectory_data):
    """
    Convert trajectory data from quaternion to 6D rotation format with velocity.
    trajectory_data: list of dicts with 'timestamp_ns', 'translation', 'rotation'
    Returns: dict with 'timestamps', 'translations', 'rotations_6d', 'poses_12d'
    """
    timestamps = []
    translations = []
    rotations = []
    
    for frame in trajectory_data:
        timestamps.append(frame['timestamp_ns'])
        translations.append(frame['translation'])  # [3]
        rotations.append(frame['rotation'][0])  # [4] - quaternion is nested in a list
    
    # Convert to tensors
    timestamps = np.array(timestamps)
    translations = torch.tensor(translations, dtype=torch.float32)  # [T, 3]
    rotations = torch.tensor(rotations, dtype=torch.float32)  # [T, 4]
    
    # Convert quaternions to 6D rotation
    rotations_6d = quat_to_6v(rotations)  # [T, 6]
    
    # Compute velocities (like reference script)
    velocities = translations[1:] - translations[:-1]  # [T-1, 3]
    # Pad velocity for last frame (like reference does)
    velocities = torch.cat([velocities, torch.zeros(1, 3)], dim=0)  # [T, 3]
    
    # Combine translation, velocity, and rotation (like reference: pos + vel + rot)
    poses_12d = torch.cat([translations, velocities, rotations_6d], dim=-1)  # [T, 12]
    poses_9d = torch.cat([translations, rotations_6d], dim=-1)  # [T, 9] - keep for compatibility
    
    return {
        'timestamps': timestamps,
        'poses_9d': poses_9d.numpy(),   # [T, 9] - translation (3) + rotation_6d (6)
        'poses_12d': poses_12d.numpy()  # [T, 12] - translation (3) + velocity (3) + rotation_6d (6)
    }

def convert_object_trajectory_to_6d(object_data):
    """
    Convert object trajectory data from quaternion to 6D rotation format.
    object_data: list of dicts with 'timestamp_ns', 'poses' (list of objects per frame)
    Returns: dict with object trajectories
    """
    if not object_data:
        return {}
    
    # Group by object_uid
    object_trajectories = {}
    
    for frame in object_data:
        timestamp = frame['timestamp_ns']
        for obj in frame['poses']:
            obj_uid = obj['object_uid']
            if obj_uid not in object_trajectories:
                object_trajectories[obj_uid] = {
                    'timestamps': [],
                    'translations': [],
                    'rotations': []
                }
            
            object_trajectories[obj_uid]['timestamps'].append(timestamp)
            object_trajectories[obj_uid]['translations'].append(obj['translation'])
            object_trajectories[obj_uid]['rotations'].append(obj['rotation'][0])  # quaternion is nested
    
    # Convert each object trajectory to 6D format
    processed_objects = {}
    for obj_uid, obj_data in object_trajectories.items():
        timestamps = np.array(obj_data['timestamps'])
        translations = torch.tensor(obj_data['translations'], dtype=torch.float32)  # [T, 3]
        rotations = torch.tensor(obj_data['rotations'], dtype=torch.float32)  # [T, 4]
        
        # Convert quaternions to 6D rotation
        rotations_6d = quat_to_6v(rotations)  # [T, 6]
        
        # Compute velocities (like reference script)
        velocities = translations[1:] - translations[:-1]  # [T-1, 3]
        # Pad velocity for last frame (like reference does)
        velocities = torch.cat([velocities, torch.zeros(1, 3)], dim=0)  # [T, 3]
        
        # Combine translation, velocity, and rotation (like reference: pos + vel + rot)
        poses_12d = torch.cat([translations, velocities, rotations_6d], dim=-1)  # [T, 12]
        poses_9d = torch.cat([translations, rotations_6d], dim=-1)  # [T, 9] - keep for compatibility
        
        processed_objects[obj_uid] = {
            'timestamps': timestamps,
            'poses_9d': poses_9d.numpy(),   # [T, 9] - translation (3) + rotation_6d (6)
            'poses_12d': poses_12d.numpy()  # [T, 12] - translation (3) + velocity (3) + rotation_6d (6)
        }
    
    return processed_objects

def preprocess_data(input_path, output_path):
    """Preprocess the raw pickle data and save it in 6D rotation format."""
    print(f"Loading data from {input_path}...")
    raw_data = load_pickle(input_path)
    
    print(f"Found {len(raw_data)} demonstrations")
    
    processed_data = {}
    
    for demo_id, demo_data in raw_data.items():
        print(f"Processing demonstration: {demo_id}")
        
        processed_demo = {}
        
        # Process left hand trajectory
        if 'left_hand' in demo_data:
            print(f"  Processing left hand trajectory ({len(demo_data['left_hand'])} frames)")
            processed_demo['left_hand'] = convert_trajectory_to_6d(demo_data['left_hand'])
        
        # Process right hand trajectory
        if 'right_hand' in demo_data:
            print(f"  Processing right hand trajectory ({len(demo_data['right_hand'])} frames)")
            processed_demo['right_hand'] = convert_trajectory_to_6d(demo_data['right_hand'])
        
        # Process object trajectories
        if 'object_pose' in demo_data:
            print(f"  Processing object trajectories ({len(demo_data['object_pose'])} frames)")
            processed_demo['objects'] = convert_object_trajectory_to_6d(demo_data['object_pose'])
            print(f"    Found {len(processed_demo['objects'])} unique objects")
        
        processed_data[demo_id] = processed_demo
    
    # Save processed data
    print(f"Saving processed data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("Preprocessing completed!")
    
    # Print summary
    print("\nSummary:")
    for demo_id, demo_data in processed_data.items():
        print(f"Demo {demo_id}:")
        if 'left_hand' in demo_data:
            print(f"  Left hand: {demo_data['left_hand']['poses_9d'].shape}")
        if 'right_hand' in demo_data:
            print(f"  Right hand: {demo_data['right_hand']['poses_9d'].shape}")
        if 'objects' in demo_data:
            print(f"  Objects: {len(demo_data['objects'])} unique objects")
            for obj_id, obj_data in demo_data['objects'].items():
                print(f"    {obj_id}: {obj_data['poses_9d'].shape}")
        break  # Just show first demo as example
    
    return processed_data

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw trajectory data by converting quaternions to 6D rotation format."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/generation.pkl",
        help="Path to input pickle file"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str,
        default="data/processed_data_with_velocity.pkl",
        help="Path to output processed pickle file"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
        
    preprocess_data(args.input, args.output)

if __name__ == "__main__":
    main() 