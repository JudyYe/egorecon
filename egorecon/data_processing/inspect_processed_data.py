#!/usr/bin/env python3
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def analyze_velocity_distribution(processed_data, velocity_thresholds=[0.01, 0.02, 0.03, 0.05, 0.1]):
    """Analyze velocity distribution across all demonstrations and objects."""
    print(f"\nAnalyzing velocity distribution across {len(processed_data)} demonstrations...")
    
    all_velocities = []
    moving_segments = []
    stationary_segments = []
    
    for demo_id, demo_data in processed_data.items():
        if 'objects' not in demo_data:
            continue
            
        objects_data = demo_data['objects']
        
        for obj_id, obj_data in objects_data.items():
            object_poses = obj_data['poses_9d']
            
            # Calculate velocities
            object_positions = object_poses[:, :3]  # Extract 3D positions
            velocities = np.linalg.norm(np.diff(object_positions, axis=0), axis=1)
            all_velocities.extend(velocities)
            
            # Analyze different thresholds
            for threshold in velocity_thresholds:
                in_motion = velocities > threshold
                
                # Find contiguous blocks
                motion_blocks = []
                stationary_blocks = []
                
                current_state = in_motion[0]
                start_idx = 0
                
                for i in range(1, len(in_motion)):
                    if in_motion[i] != current_state:
                        if current_state:
                            motion_blocks.append((start_idx, i))
                        else:
                            stationary_blocks.append((start_idx, i))
                        start_idx = i
                        current_state = in_motion[i]
                
                # Handle the last block
                if current_state:
                    motion_blocks.append((start_idx, len(in_motion)))
                else:
                    stationary_blocks.append((start_idx, len(in_motion)))
                
                # Store segment lengths
                moving_segments.extend([end - start for start, end in motion_blocks])
                stationary_segments.extend([end - start for start, end in stationary_blocks])
    
    # Statistics
    all_velocities = np.array(all_velocities)
    
    print(f"Total velocity samples: {len(all_velocities)}")
    print(f"Velocity statistics:")
    print(f"  Mean: {np.mean(all_velocities):.6f}")
    print(f"  Median: {np.median(all_velocities):.6f}")
    print(f"  Std: {np.std(all_velocities):.6f}")
    print(f"  Min: {np.min(all_velocities):.6f}")
    print(f"  Max: {np.max(all_velocities):.6f}")
    
    # Percentile analysis
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nVelocity percentiles:")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(all_velocities, p):.6f}")
    
    # Threshold analysis
    print(f"\nThreshold analysis:")
    for threshold in velocity_thresholds:
        moving_ratio = np.mean(all_velocities > threshold)
        print(f"  Threshold {threshold:.3f}: {moving_ratio*100:.1f}% moving, {(1-moving_ratio)*100:.1f}% stationary")
    
    return all_velocities, velocity_thresholds

def analyze_interaction_windows(processed_data, velocity_threshold=0.02, min_window_size=30, max_window_size=128):
    """Analyze how many interaction windows we get with different parameters."""
    print(f"\nAnalyzing interaction windows (threshold={velocity_threshold}, min_size={min_window_size})...")
    
    total_moving_windows = 0
    total_stationary_windows = 0
    moving_window_lengths = []
    stationary_window_lengths = []
    
    for demo_id, demo_data in processed_data.items():
        if 'objects' not in demo_data:
            continue
            
        objects_data = demo_data['objects']
        
        for obj_id, obj_data in objects_data.items():
            object_poses = obj_data['poses_9d']
            
            # Calculate velocities
            object_positions = object_poses[:, :3]
            velocities = np.linalg.norm(np.diff(object_positions, axis=0), axis=1)
            in_motion = velocities > velocity_threshold
            
            # Find contiguous blocks of motion
            motion_blocks = []
            start_idx = None
            for i, moving in enumerate(in_motion):
                if moving and start_idx is None:
                    start_idx = i
                elif not moving and start_idx is not None:
                    motion_blocks.append((start_idx, i))
                    start_idx = None
            if start_idx is not None:
                motion_blocks.append((start_idx, len(in_motion)))
            
            # Count moving windows
            for start, end in motion_blocks:
                actual_len = end - start
                if actual_len >= min_window_size:
                    # Count how many windows we can extract
                    if actual_len <= max_window_size:
                        total_moving_windows += 1
                        moving_window_lengths.append(actual_len)
                    else:
                        # For long sequences, we can extract multiple windows
                        num_windows = actual_len // max_window_size
                        total_moving_windows += num_windows
                        moving_window_lengths.extend([max_window_size] * num_windows)
            
            # Find stationary blocks
            stationary_blocks = []
            start_idx = None
            for i, moving in enumerate(in_motion):
                if not moving and start_idx is None:
                    start_idx = i
                elif moving and start_idx is not None:
                    stationary_blocks.append((start_idx, i))
                    start_idx = None
            if start_idx is not None:
                stationary_blocks.append((start_idx, len(in_motion)))
            
            # Count stationary windows
            for start, end in stationary_blocks:
                actual_len = end - start
                if actual_len >= min_window_size:
                    if actual_len <= max_window_size:
                        total_stationary_windows += 1
                        stationary_window_lengths.append(actual_len)
                    else:
                        num_windows = actual_len // max_window_size
                        total_stationary_windows += num_windows
                        stationary_window_lengths.extend([max_window_size] * num_windows)
    
    print(f"Window analysis results:")
    print(f"  Moving windows: {total_moving_windows}")
    print(f"  Stationary windows: {total_stationary_windows}")
    print(f"  Total windows: {total_moving_windows + total_stationary_windows}")
    print(f"  Moving ratio: {total_moving_windows / (total_moving_windows + total_stationary_windows) * 100:.1f}%")
    print(f"  Stationary ratio: {total_stationary_windows / (total_moving_windows + total_stationary_windows) * 100:.1f}%")
    
    if moving_window_lengths:
        print(f"  Moving window lengths - Mean: {np.mean(moving_window_lengths):.1f}, Std: {np.std(moving_window_lengths):.1f}")
    if stationary_window_lengths:
        print(f"  Stationary window lengths - Mean: {np.mean(stationary_window_lengths):.1f}, Std: {np.std(stationary_window_lengths):.1f}")
    
    return total_moving_windows, total_stationary_windows

def inspect_processed_data(data_path):
    """Inspect the processed data and show available demonstrations and objects."""
    print(f"Loading processed data from {data_path}...")
    processed_data = load_pickle(data_path)
    
    print(f"\nFound {len(processed_data)} demonstrations:")
    
    for demo_id, demo_data in processed_data.items():
        print(f"\nDemo ID: {demo_id}")
        
        # Left hand info
        if 'left_hand' in demo_data:
            left_shape = demo_data['left_hand']['poses_9d'].shape
            print(f"  Left hand: {left_shape} (T={left_shape[0]}, D={left_shape[1]})")
        
        # Right hand info  
        if 'right_hand' in demo_data:
            right_shape = demo_data['right_hand']['poses_9d'].shape
            print(f"  Right hand: {right_shape} (T={right_shape[0]}, D={right_shape[1]})")
        
        # Object info
        if 'objects' in demo_data:
            objects = demo_data['objects']
            print(f"  Objects ({len(objects)} total):")
            for obj_id, obj_data in objects.items():
                obj_shape = obj_data['poses_9d'].shape
                print(f"    {obj_id}: {obj_shape} (T={obj_shape[0]}, D={obj_shape[1]})")
        
        # Show only first 3 demos in detail to avoid too much output
        if list(processed_data.keys()).index(demo_id) >= 2:
            print(f"\n... and {len(processed_data) - 3} more demonstrations")
            break
    
    # Analyze velocity distribution
    all_velocities, thresholds = analyze_velocity_distribution(processed_data)
    
    # Analyze interaction windows with different thresholds
    print(f"\n" + "="*60)
    print("INTERACTION WINDOW ANALYSIS")
    print("="*60)
    
    for threshold in [0.01, 0.02, 0.03, 0.05]:
        print(f"\nThreshold: {threshold}")
        moving_count, stationary_count = analyze_interaction_windows(processed_data, velocity_threshold=threshold)
        
    # Summary of first demo for training
    first_demo_id = list(processed_data.keys())[0]
    first_demo = processed_data[first_demo_id]
    first_object_id = list(first_demo['objects'].keys())[0]
    
    print(f"\n" + "="*50)
    print("RECOMMENDED FOR OVERFITTING:")
    print(f"Demo ID: {first_demo_id}")
    print(f"Object ID: {first_object_id}")
    print(f"Left hand shape: {first_demo['left_hand']['poses_9d'].shape}")
    print(f"Right hand shape: {first_demo['right_hand']['poses_9d'].shape}")
    print(f"Object shape: {first_demo['objects'][first_object_id]['poses_9d'].shape}")
    
    # Recommend velocity threshold based on analysis
    moving_ratios = []
    for threshold in [0.01, 0.02, 0.03, 0.05]:
        moving_count, stationary_count = analyze_interaction_windows(processed_data, velocity_threshold=threshold)
        if moving_count + stationary_count > 0:
            moving_ratio = moving_count / (moving_count + stationary_count)
            moving_ratios.append((threshold, moving_ratio))
    
    # Find threshold that gives closest to 50% balance
    best_threshold = min(moving_ratios, key=lambda x: abs(x[1] - 0.5))
    print(f"\nRECOMMENDED VELOCITY THRESHOLD: {best_threshold[0]} (gives {best_threshold[1]*100:.1f}% moving windows)")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(
        description="Inspect processed data and analyze velocity distributions for balancing dataset."
    )
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default="data/processed_data.pkl",
        help="Path to processed data pickle file"
    )
    
    args = parser.parse_args()
    inspect_processed_data(args.data_path)

if __name__ == "__main__":
    main() 