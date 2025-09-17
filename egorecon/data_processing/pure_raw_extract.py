#!/usr/bin/env python3
"""
Script to extract PURE raw data from processed_data.pkl without any added metadata
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

def extract_pure_raw_data(pkl_path, max_frames=3):
    """Extract PURE raw data without any added metadata"""
    print(f"Extracting PURE raw data from: {pkl_path}")
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
        
        # Extract PURE raw data structure (exactly as it exists in the pickle)
        pure_raw_data = {
            first_seq_key: {}
        }
        
        for key, value in first_seq.items():
            print(f"\nProcessing: {key}")
            
            if isinstance(value, dict) and 'timestamps' in value:
                timestamps = value['timestamps']
                poses = value['poses_9d']
                
                print(f"  Timestamps: {type(timestamps)} with shape {timestamps.shape}")
                print(f"  Poses: {type(poses)} with shape {poses.shape}")
                
                # Store EXACTLY as it exists in the pickle
                pure_raw_data[first_seq_key][key] = {
                    'timestamps': timestamps[:max_frames].tolist(),
                    'poses_9d': poses[:max_frames].tolist()
                }
                
            elif isinstance(value, dict):
                # Objects dict
                print(f"  Objects: {len(value)} objects")
                pure_raw_data[first_seq_key][key] = {}
                
                # Sample first few objects
                for i, (obj_id, obj_data) in enumerate(value.items()):
                    if i >= 2:  # Only show first 2 objects
                        break
                        
                    if isinstance(obj_data, dict) and 'timestamps' in obj_data:
                        timestamps = obj_data['timestamps']
                        poses = obj_data['poses_9d']
                        
                        print(f"    Object {obj_id}: {type(poses)} with shape {poses.shape}")
                        
                        # Store EXACTLY as it exists in the pickle
                        pure_raw_data[first_seq_key][key][str(obj_id)] = {
                            'timestamps': timestamps[:max_frames].tolist(),
                            'poses_9d': poses[:max_frames].tolist()
                        }
        
        # Convert to JSON-serializable format
        json_data = numpy_to_json(pure_raw_data)
        
        # Save pure raw JSON
        output_file = os.path.join(os.path.dirname(pkl_path), "pure_raw_data.json")
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nPure raw JSON saved to: {output_file}")
        
        # Display the PURE raw JSON structure
        print("\n" + "="*60)
        print("PURE RAW DATA STRUCTURE (exactly as in pickle)")
        print("="*60)
        print(json.dumps(json_data, indent=2))
        
        # Also show what the original pickle structure looks like
        print("\n" + "="*60)
        print("ORIGINAL PICKLE STRUCTURE")
        print("="*60)
        print("The original pickle contains:")
        print(f"- A dictionary with {len(data)} sequence keys")
        print(f"- Each sequence contains: {list(first_seq.keys())}")
        print(f"- Each component has 'timestamps' and 'poses_9d' keys")
        print(f"- No 'data_structure', 'data_type', or other metadata fields")
        
        return json_data
        
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for processed_data.pkl
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    pkl_path = os.path.join(data_dir, "processed_data.pkl")
    
    extract_pure_raw_data(pkl_path)
