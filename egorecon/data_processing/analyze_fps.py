#!/usr/bin/env python3
"""
Script to analyze FPS and timestep information from processed_data.pkl
"""
import pickle
import numpy as np
import os
from pprint import pprint

def analyze_timestamps_and_fps(timestamps):
    """Analyze timestamps to determine FPS and timestep information"""
    if len(timestamps) < 2:
        return {"error": "Need at least 2 timestamps"}
    
    # Convert to numpy array if needed
    timestamps = np.array(timestamps)
    
    # Calculate time differences in nanoseconds
    time_diffs_ns = np.diff(timestamps)
    
    # Convert to seconds
    time_diffs_s = time_diffs_ns / 1e9
    
    # Calculate FPS statistics
    fps_values = 1.0 / time_diffs_s
    
    analysis = {
        "total_frames": len(timestamps),
        "total_duration_s": (timestamps[-1] - timestamps[0]) / 1e9,
        "mean_fps": np.mean(fps_values),
        "median_fps": np.median(fps_values),
        "min_fps": np.min(fps_values),
        "max_fps": np.max(fps_values),
        "std_fps": np.std(fps_values),
        "mean_timestep_s": np.mean(time_diffs_s),
        "median_timestep_s": np.median(time_diffs_s),
        "min_timestep_s": np.min(time_diffs_s),
        "max_timestep_s": np.max(time_diffs_s),
        "std_timestep_s": np.std(time_diffs_s),
    }
    
    return analysis

def inspect_processed_data(pkl_path):
    """Inspect the processed data and analyze FPS"""
    print(f"Inspecting: {pkl_path}")
    print("=" * 60)
    
    if not os.path.exists(pkl_path):
        print(f"ERROR: File {pkl_path} does not exist!")
        return
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Number of sequences: {len(data)}")
            print(f"Sequence keys: {list(data.keys())}")
            print()
            
            # Analyze first sequence in detail
            first_seq_key = list(data.keys())[0]
            first_seq = data[first_seq_key]
            print(f"Analyzing first sequence: {first_seq_key}")
            print(f"Sequence keys: {list(first_seq.keys())}")
            print()
            
            # Analyze each hand/object in the sequence
            for key, value in first_seq.items():
                print(f"=== {key.upper()} ===")
                if isinstance(value, dict) and 'timestamps' in value:
                    timestamps = value['timestamps']
                    poses = value['poses_9d']
                    
                    print(f"Number of frames: {len(timestamps)}")
                    print(f"Pose shape: {poses.shape}")
                    
                    # Analyze FPS
                    fps_analysis = analyze_timestamps_and_fps(timestamps)
                    print("FPS Analysis:")
                    for k, v in fps_analysis.items():
                        if isinstance(v, float):
                            print(f"  {k}: {v:.3f}")
                        else:
                            print(f"  {k}: {v}")
                    
                    # Show sample timestamps
                    print(f"First 5 timestamps (ns): {timestamps[:5]}")
                    print(f"Last 5 timestamps (ns): {timestamps[-5:]}")
                    
                    # Show sample poses
                    print(f"Sample pose (first frame, first joint): {poses[0, 0, :]}")
                    print()
                    
                elif isinstance(value, dict):
                    # This might be objects dict
                    print(f"Number of objects: {len(value)}")
                    for obj_id, obj_data in value.items():
                        if isinstance(obj_data, dict) and 'timestamps' in obj_data:
                            timestamps = obj_data['timestamps']
                            poses = obj_data['poses_9d']
                            
                            print(f"  Object {obj_id}:")
                            print(f"    Frames: {len(timestamps)}")
                            print(f"    Pose shape: {poses.shape}")
                            
                            fps_analysis = analyze_timestamps_and_fps(timestamps)
                            print(f"    Mean FPS: {fps_analysis['mean_fps']:.3f}")
                            print()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Overall summary across all sequences
        all_fps_values = []
        all_durations = []
        
        for seq_key, seq_data in data.items():
            for key, value in seq_data.items():
                if isinstance(value, dict) and 'timestamps' in value:
                    timestamps = value['timestamps']
                    fps_analysis = analyze_timestamps_and_fps(timestamps)
                    all_fps_values.append(fps_analysis['mean_fps'])
                    all_durations.append(fps_analysis['total_duration_s'])
        
        if all_fps_values:
            print(f"Overall FPS statistics across all sequences:")
            print(f"  Mean FPS: {np.mean(all_fps_values):.3f}")
            print(f"  Median FPS: {np.median(all_fps_values):.3f}")
            print(f"  Min FPS: {np.min(all_fps_values):.3f}")
            print(f"  Max FPS: {np.max(all_fps_values):.3f}")
            print(f"  Std FPS: {np.std(all_fps_values):.3f}")
            print()
            print(f"Duration statistics:")
            print(f"  Mean duration: {np.mean(all_durations):.3f}s")
            print(f"  Median duration: {np.median(all_durations):.3f}s")
            print(f"  Min duration: {np.min(all_durations):.3f}s")
            print(f"  Max duration: {np.max(all_durations):.3f}s")
            
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for processed_data.pkl
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    pkl_path = os.path.join(data_dir, "processed_data.pkl")
    
    inspect_processed_data(pkl_path)
