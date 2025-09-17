#!/usr/bin/env python3
"""
Script to inspect the format of data/hand_articulations.pkl
"""
import pickle
import os
from pprint import pprint

def inspect_pkl_data(pkl_path):
    """Inspect the structure and contents of a pickle file"""
    print(f"Inspecting: {pkl_path}")
    print("=" * 50)
    
    if not os.path.exists(pkl_path):
        print(f"ERROR: File {pkl_path} does not exist!")
        return
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Data structure:")
        
        if isinstance(data, dict):
            print("Dictionary keys:")
            for key in data.keys():
                value = data[key]
                print(f"  '{key}': {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
                elif isinstance(value, (list, tuple)):
                    print(f"    Length: {len(value)}")
                    if len(value) > 0:
                        print(f"    First element type: {type(value[0])}")
                        if hasattr(value[0], 'shape'):
                            print(f"    First element shape: {value[0].shape}")
                elif isinstance(value, dict):
                    print(f"    Nested dict with keys: {list(value.keys())[:5]}...")
                print()
        
        elif isinstance(data, (list, tuple)):
            print(f"List/Tuple with {len(data)} elements")
            if len(data) > 0:
                print(f"First element type: {type(data[0])}")
                if hasattr(data[0], 'shape'):
                    print(f"First element shape: {data[0].shape}")
        
        else:
            print(f"Single object of type: {type(data)}")
            if hasattr(data, '__dict__'):
                print("Object attributes:")
                pprint(vars(data))
        
        # Try to print a sample of the data (first few items)
        print("\nSample data (first few items):")
        if isinstance(data, dict):
            for i, (key, value) in enumerate(data.items()):
                if i >= 3:  # Only show first 3 items
                    break
                print(f"  {key}: {str(value)[:100]}...")
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data[:3]):
                print(f"  [{i}]: {str(item)[:100]}...")
        else:
            print(f"  {str(data)[:200]}...")
            
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")

if __name__ == "__main__":
    # Check if data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist!")
        print("Available directories in workspace:")
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        for item in os.listdir(workspace_root):
            item_path = os.path.join(workspace_root, item)
            if os.path.isdir(item_path):
                print(f"  {item}/")
    else:
        pkl_path = os.path.join(data_dir, "hand_articulations.pkl")
        inspect_pkl_data(pkl_path)
        
        # Also check MANO models
        mano_dir = os.path.join(data_dir, "mano_models")
        if os.path.exists(mano_dir):
            print("\n" + "="*50)
            print("MANO Models found:")
            for mano_file in ["MANO_LEFT.pkl", "MANO_RIGHT.pkl"]:
                mano_path = os.path.join(mano_dir, mano_file)
                if os.path.exists(mano_path):
                    print(f"  ✓ {mano_path}")
                else:
                    print(f"  ✗ {mano_path} (missing)")
        else:
            print(f"\nMANO models directory {mano_dir} does not exist!")
