#!/usr/bin/env python3
"""
Script to extract a sample data point from processed_data.pkl and display in JSON format
"""
import pickle
import json
import numpy as np
import os

def convert_numpy_to_json(obj):
    """Convert numpy arrays and other numpy types to JSON-serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj

def extract_sample_data_point(pkl_path, max_samples=3):
    """Extract sample data points from processed_data.pkl"""
    print(f"Extracting sample data from: {pkl_path}")
    print("=" * 60)
    
    if not os.path.exists(pkl_path):
        print(f"ERROR: File {pkl_path} does not exist!")
        return
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Number of sequences: {len(data)}")
        print(f"Sequence keys: {list(data.keys())}")
        print()
        
        # Get first sequence
        first_seq_key = list(data.keys())[0]
        first_seq = data[first_seq_key]
        print(f"Analyzing sequence: {first_seq_key}")
        print(f"Sequence keys: {list(first_seq.keys())}")
        print()
        
        # Extract sample data points
        sample_data = {}
        
        for key, value in first_seq.items():
            print(f"=== {key.upper()} ===")
            
            if isinstance(value, dict) and 'timestamps' in value:
                timestamps = value['timestamps']
                poses = value['poses_9d']
                
                print(f"Number of frames: {len(timestamps)}")
                print(f"Pose shape: {poses.shape}")
                
                # Extract first few frames as samples
                num_samples = min(max_samples, len(timestamps))
                sample_data[key] = {
                    'num_frames': len(timestamps),
                    'pose_shape': list(poses.shape),
                    'samples': []
                }
                
                for i in range(num_samples):
                    sample = {
                        'frame_index': i,
                        'timestamp_ns': int(timestamps[i]),
                        'pose_9d': poses[i].tolist() if hasattr(poses[i], 'tolist') else poses[i]
                    }
                    sample_data[key]['samples'].append(sample)
                    
                    # Show first frame details
                    if i == 0:
                        print(f"  Frame {i}:")
                        print(f"    Timestamp: {timestamps[i]} ns")
                        print(f"    Pose shape: {poses[i].shape if hasattr(poses[i], 'shape') else 'N/A'}")
                        if len(poses[i].shape) == 1:
                            print(f"    Pose (first 5 values): {poses[i][:5]}")
                        elif len(poses[i].shape) == 2:
                            print(f"    Pose (first joint, first 5 values): {poses[i][0, :5]}")
                
                print()
                
            elif isinstance(value, dict):
                # This might be objects dict
                print(f"Number of objects: {len(value)}")
                sample_data[key] = {
                    'num_objects': len(value),
                    'objects': {}
                }
                
                # Sample first few objects
                for i, (obj_id, obj_data) in enumerate(value.items()):
                    if i >= max_samples:
                        break
                        
                    if isinstance(obj_data, dict) and 'timestamps' in obj_data:
                        timestamps = obj_data['timestamps']
                        poses = obj_data['poses_9d']
                        
                        print(f"  Object {obj_id}:")
                        print(f"    Frames: {len(timestamps)}")
                        print(f"    Pose shape: {poses.shape}")
                        
                        # Extract sample data
                        sample_data[key]['objects'][str(obj_id)] = {
                            'num_frames': len(timestamps),
                            'pose_shape': list(poses.shape),
                            'samples': []
                        }
                        
                        # First few frames
                        num_samples = min(2, len(timestamps))
                        for j in range(num_samples):
                            sample = {
                                'frame_index': j,
                                'timestamp_ns': int(timestamps[j]),
                                'pose_9d': poses[j].tolist() if hasattr(poses[j], 'tolist') else poses[j]
                            }
                            sample_data[key]['objects'][str(obj_id)]['samples'].append(sample)
                        
                        print(f"    Sample pose (first frame): {poses[0][:5] if len(poses[0]) >= 5 else poses[0]}")
                        print()
        
        # Convert to JSON-serializable format
        json_data = convert_numpy_to_json(sample_data)
        
        # Save sample data to JSON file
        output_file = os.path.join(os.path.dirname(pkl_path), "sample_data_point.json")
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nSample data saved to: {output_file}")
        
        # Display first sample in a clean format
        print("\n" + "="*60)
        print("SAMPLE DATA POINT (First Frame)")
        print("="*60)
        
        if 'left_hand' in sample_data and sample_data['left_hand']['samples']:
            first_sample = sample_data['left_hand']['samples'][0]
            print("Left Hand - First Frame:")
            print(f"  Timestamp: {first_sample['timestamp_ns']} ns")
            print(f"  Pose 9D: {first_sample['pose_9d'][:5]}... (showing first 5 values)")
            print()
        
        if 'right_hand' in sample_data and sample_data['right_hand']['samples']:
            first_sample = sample_data['right_hand']['samples'][0]
            print("Right Hand - First Frame:")
            print(f"  Timestamp: {first_sample['timestamp_ns']} ns")
            print(f"  Pose 9D: {first_sample['pose_9d'][:5]}... (showing first 5 values)")
            print()
        
        if 'objects' in sample_data and sample_data['objects']['objects']:
            first_obj_id = list(sample_data['objects']['objects'].keys())[0]
            first_obj = sample_data['objects']['objects'][first_obj_id]
            if first_obj['samples']:
                first_sample = first_obj['samples'][0]
                print(f"Object {first_obj_id} - First Frame:")
                print(f"  Timestamp: {first_sample['timestamp_ns']} ns")
                print(f"  Pose 9D: {first_sample['pose_9d'][:5]}... (showing first 5 values)")
                print()
        
        return json_data
        
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for processed_data.pkl
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    pkl_path = os.path.join(data_dir, "processed_data.pkl")
    
    extract_sample_data_point(pkl_path)
