#!/usr/bin/env python3
"""
Simple script to inspect processed_data.pkl without numpy dependencies
"""
import pickle
import os

def simple_inspect(pkl_path):
    """Simple inspection without numpy"""
    print(f"Inspecting: {pkl_path}")
    print("=" * 50)
    
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
            
            # Analyze first sequence
            first_seq_key = list(data.keys())[0]
            first_seq = data[first_seq_key]
            print(f"First sequence: {first_seq_key}")
            print(f"Sequence keys: {list(first_seq.keys())}")
            print()
            
            # Analyze each component
            for key, value in first_seq.items():
                print(f"=== {key.upper()} ===")
                if isinstance(value, dict) and 'timestamps' in value:
                    timestamps = value['timestamps']
                    poses = value['poses_9d']
                    
                    print(f"Number of frames: {len(timestamps)}")
                    print(f"Pose shape: {poses.shape}")
                    
                    # Calculate FPS manually
                    if len(timestamps) > 1:
                        # Calculate time differences in nanoseconds
                        time_diffs = []
                        for i in range(1, len(timestamps)):
                            diff_ns = timestamps[i] - timestamps[i-1]
                            time_diffs.append(diff_ns)
                        
                        # Convert to seconds and calculate FPS
                        time_diffs_s = [diff / 1e9 for diff in time_diffs]
                        fps_values = [1.0 / diff_s for diff_s in time_diffs_s]
                        
                        # Calculate statistics
                        total_duration_s = (timestamps[-1] - timestamps[0]) / 1e9
                        mean_fps = sum(fps_values) / len(fps_values)
                        min_fps = min(fps_values)
                        max_fps = max(fps_values)
                        
                        print(f"Total duration: {total_duration_s:.3f}s")
                        print(f"Mean FPS: {mean_fps:.3f}")
                        print(f"Min FPS: {min_fps:.3f}")
                        print(f"Max FPS: {max_fps:.3f}")
                        print(f"Mean timestep: {total_duration_s / len(timestamps):.6f}s")
                        
                        # Show sample timestamps
                        print(f"First 5 timestamps (ns): {timestamps[:5]}")
                        print(f"Last 5 timestamps (ns): {timestamps[-5:]}")
                        
                        # Show sample time differences
                        print(f"First 5 time differences (ns): {time_diffs[:5]}")
                        print(f"First 5 time differences (s): {time_diffs_s[:5]}")
                        print(f"First 5 FPS values: {fps_values[:5]}")
                        
                        # Show sample pose - handle different shapes
                        if len(poses.shape) == 2:
                            # Shape is (T, 9) - flattened poses
                            print(f"Sample pose (first frame): {poses[0, :]}")
                        elif len(poses.shape) == 3:
                            # Shape is (T, 21, 9) - poses per joint
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
                            
                            if len(timestamps) > 1:
                                total_duration_s = (timestamps[-1] - timestamps[0]) / 1e9
                                mean_timestep = total_duration_s / len(timestamps)
                                mean_fps = 1.0 / mean_timestep
                                print(f"    Mean FPS: {mean_fps:.3f}")
                                print(f"    Mean timestep: {mean_timestep:.6f}s")
                            print()
        
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        
        # Overall summary
        all_fps_values = []
        all_durations = []
        
        for seq_key, seq_data in data.items():
            for key, value in seq_data.items():
                if isinstance(value, dict) and 'timestamps' in value:
                    timestamps = value['timestamps']
                    if len(timestamps) > 1:
                        total_duration_s = (timestamps[-1] - timestamps[0]) / 1e9
                        mean_timestep = total_duration_s / len(timestamps)
                        mean_fps = 1.0 / mean_timestep
                        all_fps_values.append(mean_fps)
                        all_durations.append(total_duration_s)
        
        if all_fps_values:
            print(f"Overall statistics across all sequences:")
            print(f"  Mean FPS: {sum(all_fps_values) / len(all_fps_values):.3f}")
            print(f"  Min FPS: {min(all_fps_values):.3f}")
            print(f"  Max FPS: {max(all_fps_values):.3f}")
            print(f"  Mean duration: {sum(all_durations) / len(all_durations):.3f}s")
            print(f"  Min duration: {min(all_durations):.3f}s")
            print(f"  Max duration: {max(all_durations):.3f}s")
            
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for processed_data.pkl
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    pkl_path = os.path.join(data_dir, "processed_data.pkl")
    
    simple_inspect(pkl_path)
