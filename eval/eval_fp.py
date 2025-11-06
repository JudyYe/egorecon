import os
import os.path as osp
import pickle
import json
import numpy as np
from glob import glob
from eval.eval_hoi import eval_hotclip_pose6d

def aggregate_fp(fp_dir, save_file, split_obj=None):
    """
    Aggregate foundation pose .npz files into a format suitable for eval_hotclip_pose6d.
    
    Input format (from run_foundation_pose.py):
    - .npz files named {seq}.npz (e.g., 001874.npz)
    - Contains: wTo (num_objects, T, 4, 4), valid (num_objects, T), 
                index (array with object IDs), score (num_objects, T)
    - index format: ["left_hand", "right_hand", obj_id1, obj_id2, ...]
    - Objects start at index 2 in the index array
    
    Output format (for eval_hotclip_pose6d):
    - .pkl file with structure: {seq: {obj_{obj_id}_wTo: (T, 4, 4)}}
    
    Args:
        fp_dir: Directory containing .npz files
        save_file: Output .pkl file path
        split_obj: Optional list of {seq}_{objid} strings to filter by (e.g., ["001874_000007", ...])
                  If None, processes all files. If provided, only includes sequences/objects in the list.
    """
    fp_files = glob(osp.join(fp_dir, "*.npz"))
    
    # Convert split_obj to set for faster lookup if provided
    if split_obj is None:
        split_obj_set = None
    else:
        split_obj_set = set(split_obj)
    # Aggregate predictions by sequence
    seq_dict = {}
    included_count = 0
    excluded_count = 0
    
    for fp_file in fp_files:
        # Extract sequence name from filename (e.g., 001874.npz -> 001874)
        seq = osp.basename(fp_file).replace(".npz", "")
        
        # Load foundation pose data
        data = np.load(fp_file, allow_pickle=True)
        wTo_list = data["wTo"]  # (num_objects, T, 4, 4)
        valid_list = data["valid"]  # (num_objects, T)
        index = data["index"]  # array with object IDs
        score_list = data.get("score", None)  # (num_objects, T), optional
        
        # Initialize sequence dict if not exists
        if seq not in seq_dict:
            seq_dict[seq] = {}
        
        # Extract object IDs from index (skip first 2 which are hands)
        # index format: ["left_hand", "right_hand", obj_id1, obj_id2, ...]
        # wTo_list[o] corresponds to index[o+2]
        num_objects = wTo_list.shape[0]
        
        for o in range(num_objects):
            # Get object ID from index (objects start at index 2)
            obj_id = str(index[o + 2])
            
            # If split_obj is provided, only include if {seq}_{obj_id} is in the list
            if split_obj_set is not None:
                seq_obj_key = f"{seq}_{obj_id}"
                if seq_obj_key not in split_obj_set:
                    excluded_count += 1
                    continue
            
            included_count += 1
            # Get wTo for this object: (T, 4, 4)
            wTo = wTo_list[o]
            
            # Store with key format: obj_{obj_id}_wTo
            obj_key = f"obj_{obj_id}_wTo"
            seq_dict[seq][obj_key] = wTo
    
    # Save aggregated predictions
    os.makedirs(osp.dirname(save_file), exist_ok=True)
    with open(save_file, "wb") as f:
        pickle.dump(seq_dict, f)
    
    print(f"Aggregated {len(fp_files)} foundation pose files into {save_file}")
    print(f"Total sequences: {len(seq_dict)}")
    print(f"Included objects: {included_count}, Excluded objects: {excluded_count}")
    
    return seq_dict
    
if __name__ == "__main__":
    data_dir = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP"
    split = "test50obj"
    # load split_obj.json
    
    with open(osp.join(data_dir, "sets", "split_obj.json"), "r") as f:
        split_obj = json.load(f)
    split_obj = split_obj[split]
    new_file = osp.join(data_dir, "eval/foundation_pose", f"split_{split}.pkl")

    
    aggregate_fp(osp.join(data_dir, "foundation_pose"), new_file, split_obj=split_obj)


eval_hotclip_pose6d(pred_file=new_file, split="test50obj", skip_not_there=True, aligned=True)
