#!/usr/bin/env python3
"""
Script to inspect one sequence in detail from data/generation.pkl
"""
import pickle
import numpy as np
from pprint import pprint
import os

def inspect_sequence_detail(pkl_path):
    """Inspect the detailed structure of one sequence"""
    print(f"Detailed inspection of: {pkl_path}")
    print("=" * 50)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get the first sequence
    first_seq_key = list(data.keys())[0]
    first_seq = data[first_seq_key]
    
    print(f"Inspecting sequence: {first_seq_key}")
    print(f"Keys in sequence: {list(first_seq.keys())}")
    print()
    
    for key in first_seq.keys():
        print(f"=== {key.upper()} ===")
        sequence_data = first_seq[key]
        print(f"Type: {type(sequence_data)}")
        print(f"Length: {len(sequence_data) if hasattr(sequence_data, '__len__') else 'N/A'}")
        
        if isinstance(sequence_data, list) and len(sequence_data) > 0:
            first_item = sequence_data[0]
            print(f"First item type: {type(first_item)}")
            
            if isinstance(first_item, dict):
                print(f"First item keys: {list(first_item.keys())}")
                for item_key, item_value in first_item.items():
                    print(f"  {item_key}: {type(item_value)}")
                    if hasattr(item_value, 'shape'):
                        print(f"    Shape: {item_value.shape}")
                    elif isinstance(item_value, (list, tuple)):
                        print(f"    Length: {len(item_value)}")
                        if len(item_value) > 0 and hasattr(item_value[0], '__len__'):
                            print(f"    Sample values: {item_value[:3] if len(item_value) >= 3 else item_value}")
                    else:
                        print(f"    Value: {item_value}")
                        
                # Show a few more samples
                print(f"Sample of first 3 items:")
                for i, item in enumerate(sequence_data[:3]):
                    print(f"  Item {i}:")
                    for item_key, item_value in item.items():
                        if hasattr(item_value, 'shape'):
                            print(f"    {item_key}: shape {item_value.shape}")
                        else:
                            print(f"    {item_key}: {item_value}")
            else:
                print(f"First item: {first_item}")
        print()

if __name__ == "__main__":
    pkl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "generation.pkl")
    inspect_sequence_detail(pkl_path)
