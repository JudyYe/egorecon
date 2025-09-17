#!/usr/bin/env python3
"""
Script to extract a raw data point directly from processed_data.pkl
"""
import pickle
import os
import pprint

def extract_raw_data_point(pkl_path):
    """Extract raw data point directly from processed_data.pkl"""
    print(f"Extracting raw data from: {pkl_path}")
    print("=" * 60)
    
    if not os.path.exists(pkl_path):
        print(f"ERROR: File {pkl_path} does not exist!")
        return
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Number of sequences: {len(data)}")
        print(f"Sequence keys (first 5): {list(data.keys())[:5]}")
        print()
        
        # Get first sequence
        first_seq_key = list(data.keys())[0]
        first_seq = data[first_seq_key]
        print(f"First sequence key: {first_seq_key}")
        print(f"First sequence type: {type(first_seq)}")
        print(f"First sequence keys: {list(first_seq.keys())}")
        print()
        
        # Extract raw data for each component
        raw_data = {}
        
        for key, value in first_seq.items():
            print(f"=== {key.upper()} ===")
            print(f"Type: {type(value)}")
            
            if isinstance(value, dict) and 'timestamps' in value:
                timestamps = value['timestamps']
                poses = value['poses_9d']
                
                print(f"Timestamps type: {type(timestamps)}")
                print(f"Timestamps shape: {timestamps.shape if hasattr(timestamps, 'shape') else 'N/A'}")
                print(f"Poses type: {type(poses)}")
                print(f"Poses shape: {poses.shape if hasattr(poses, 'shape') else 'N/A'}")
                
                # Store raw data
                raw_data[key] = {
                    'timestamps': timestamps,
                    'poses_9d': poses
                }
                
                # Show first few raw values
                print(f"First 3 timestamps: {timestamps[:3]}")
                if hasattr(poses, 'shape') and len(poses.shape) == 2:
                    print(f"First pose (first 5 values): {poses[0, :5]}")
                    print(f"Second pose (first 5 values): {poses[1, :5]}")
                elif hasattr(poses, 'shape') and len(poses.shape) == 3:
                    print(f"First pose (first joint, first 5 values): {poses[0, 0, :5]}")
                    print(f"Second pose (first joint, first 5 values): {poses[1, 0, :5]}")
                print()
                
            elif isinstance(value, dict):
                # This might be objects dict
                print(f"Number of objects: {len(value)}")
                raw_data[key] = {}
                
                # Sample first few objects
                for i, (obj_id, obj_data) in enumerate(value.items()):
                    if i >= 3:  # Only show first 3 objects
                        break
                        
                    if isinstance(obj_data, dict) and 'timestamps' in obj_data:
                        timestamps = obj_data['timestamps']
                        poses = obj_data['poses_9d']
                        
                        print(f"  Object {obj_id}:")
                        print(f"    Timestamps type: {type(timestamps)}")
                        print(f"    Timestamps shape: {timestamps.shape if hasattr(timestamps, 'shape') else 'N/A'}")
                        print(f"    Poses type: {type(poses)}")
                        print(f"    Poses shape: {poses.shape if hasattr(poses, 'shape') else 'N/A'}")
                        
                        # Store raw data
                        raw_data[key][obj_id] = {
                            'timestamps': timestamps,
                            'poses_9d': poses
                        }
                        
                        # Show raw values
                        print(f"    First 3 timestamps: {timestamps[:3]}")
                        if hasattr(poses, 'shape') and len(poses.shape) == 2:
                            print(f"    First pose (first 5 values): {poses[0, :5]}")
                        print()
        
        # Display raw data structure
        print("\n" + "="*60)
        print("RAW DATA STRUCTURE")
        print("="*60)
        
        # Show the actual raw data types and values
        print("Raw data types and sample values:")
        print("-" * 40)
        
        for key, value in raw_data.items():
            if key == 'objects':
                print(f"\n{key}:")
                for obj_id, obj_data in value.items():
                    print(f"  Object {obj_id}:")
                    print(f"    Timestamps: {type(obj_data['timestamps'])} with shape {obj_data['timestamps'].shape}")
                    print(f"    Poses: {type(obj_data['poses_9d'])} with shape {obj_data['poses_9d'].shape}")
                    print(f"    First timestamp: {obj_data['timestamps'][0]}")
                    print(f"    First pose: {obj_data['poses_9d'][0]}")
            else:
                print(f"\n{key}:")
                print(f"  Timestamps: {type(value['timestamps'])} with shape {value['timestamps'].shape}")
                print(f"  Poses: {type(value['poses_9d'])} with shape {value['poses_9d'].shape}")
                print(f"  First timestamp: {value['timestamps'][0]}")
                print(f"  First pose: {value['poses_9d'][0]}")
        
        # Show the actual raw data as it appears in the pickle
        print("\n" + "="*60)
        print("ACTUAL RAW DATA (First Frame)")
        print("="*60)
        
        if 'left_hand' in raw_data:
            print("Left Hand Raw Data:")
            print(f"  Timestamp: {raw_data['left_hand']['timestamps'][0]}")
            print(f"  Pose: {raw_data['left_hand']['poses_9d'][0]}")
            print()
        
        if 'right_hand' in raw_data:
            print("Right Hand Raw Data:")
            print(f"  Timestamp: {raw_data['right_hand']['timestamps'][0]}")
            print(f"  Pose: {raw_data['right_hand']['poses_9d'][0]}")
            print()
        
        if 'objects' in raw_data:
            first_obj_id = list(raw_data['objects'].keys())[0]
            print(f"Object {first_obj_id} Raw Data:")
            print(f"  Timestamp: {raw_data['objects'][first_obj_id]['timestamps'][0]}")
            print(f"  Pose: {raw_data['objects'][first_obj_id]['poses_9d'][0]}")
            print()
        
        return raw_data
        
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for processed_data.pkl
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    pkl_path = os.path.join(data_dir, "processed_data.pkl")
    
    extract_raw_data_point(pkl_path)
