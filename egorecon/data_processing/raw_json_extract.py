#!/usr/bin/env python3
"""
Script to extract raw data from processed_data.pkl and convert to JSON
"""
import pickle
import json
import numpy as np
import os

def numpy_to_json(obj):
    """Convert numpy objects to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(key): numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json(item) for item in obj]
    else:
        return obj

def extract_raw_json(pkl_path, max_frames=5):
    """Extract raw data and convert to JSON"""
    print(f"Extracting raw JSON from: {pkl_path}")
    print("=" * 60)
    
    if not os.path.exists(pkl_path):
        print(f"ERROR: File {pkl_path} does not exist!")
        return
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Number of sequences: {len(data)}")
        
        # Get first sequence
        first_seq_key = list(data.keys())[0]
        first_seq = data[first_seq_key]
        print(f"First sequence: {first_seq_key}")
        
        # Extract raw data structure
        raw_json_data = {
            "sequence_id": first_seq_key,
            "data_structure": {}
        }
        
        for key, value in first_seq.items():
            print(f"\nProcessing: {key}")
            
            if isinstance(value, dict) and 'timestamps' in value:
                timestamps = value['timestamps']
                poses = value['poses_9d']
                
                print(f"  Timestamps: {type(timestamps)} with shape {timestamps.shape}")
                print(f"  Poses: {type(poses)} with shape {poses.shape}")
                
                # Extract raw data
                raw_json_data["data_structure"][key] = {
                    "data_type": str(type(poses)),
                    "shape": poses.shape,
                    "num_frames": len(timestamps),
                    "raw_timestamps": timestamps[:max_frames].tolist(),
                    "raw_poses": poses[:max_frames].tolist()
                }
                
            elif isinstance(value, dict):
                # Objects dict
                print(f"  Objects: {len(value)} objects")
                raw_json_data["data_structure"][key] = {
                    "num_objects": len(value),
                    "objects": {}
                }
                
                # Sample first few objects
                for i, (obj_id, obj_data) in enumerate(value.items()):
                    if i >= 3:  # Only show first 3 objects
                        break
                        
                    if isinstance(obj_data, dict) and 'timestamps' in obj_data:
                        timestamps = obj_data['timestamps']
                        poses = obj_data['poses_9d']
                        
                        print(f"    Object {obj_id}: {type(poses)} with shape {poses.shape}")
                        
                        raw_json_data["data_structure"][key]["objects"][str(obj_id)] = {
                            "data_type": str(type(poses)),
                            "shape": poses.shape,
                            "num_frames": len(timestamps),
                            "raw_timestamps": timestamps[:max_frames].tolist(),
                            "raw_poses": poses[:max_frames].tolist()
                        }
        
        # Convert to JSON-serializable format
        json_data = numpy_to_json(raw_json_data)
        
        # Save raw JSON
        output_file = os.path.join(os.path.dirname(pkl_path), "raw_data.json")
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nRaw JSON saved to: {output_file}")
        
        # Display the raw JSON structure
        print("\n" + "="*60)
        print("RAW JSON STRUCTURE")
        print("="*60)
        print(json.dumps(json_data, indent=2))
        
        return json_data
        
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for processed_data.pkl
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    pkl_path = os.path.join(data_dir, "processed_data.pkl")
    
    extract_raw_json(pkl_path)
