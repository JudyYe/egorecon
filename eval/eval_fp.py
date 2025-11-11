from fire import Fire
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
        split_seq_set = None
    else:
        split_obj_set = set()
        split_seq_set = set()
        for item in split_obj:
            if "_" in item:
                split_obj_set.add(item)
            else:
                split_seq_set.add(item)
    # Aggregate predictions by sequence
    seq_dict = {}
    included_count = 0
    excluded_count = 0
    
    for fp_file in fp_files:
        basename = osp.basename(fp_file).replace(".npz", "")
        if "_" in basename:
            seq, obj_suffix = basename.split("_", 1)
            file_objects = {obj_suffix}
        else:
            seq = basename
            file_objects = None
        
        data = np.load(fp_file, allow_pickle=True)
        wTo_list = data["wTo"]
        _valid_list = data["valid"]
        print("Mean valid", np.mean(_valid_list))
        index = data["index"]
        _score_list = data.get("score", None)
        
        if seq not in seq_dict:
            seq_dict[seq] = {}
        
        num_objects = wTo_list.shape[0]
        for o in range(num_objects):
            obj_id = str(index[o + 2])
            if file_objects is not None and obj_id not in file_objects:
                continue
            if split_obj_set is not None:
                seq_obj_key = f"{seq}_{obj_id}"
                if (
                    seq_obj_key not in split_obj_set
                    and (split_seq_set is None or seq not in split_seq_set)
                ):
                    excluded_count += 1
                    continue
            included_count += 1
            wTo = wTo_list[o]
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


def eval_fp(fp_dir, split="test50obj"):
    
    with open(osp.join(data_dir, "sets", "split.json"), "r") as f:
        split_obj = json.load(f)
    split_obj = split_obj[split]
    # new_file = osp.join(data_dir, "eval/foundation_pose", "post_objects.pkl")
    new_file = osp.join(fp_dir, "eval/post_objects.pkl")
    
    # aggregate_fp(osp.join(data_dir, "foundation_pose"), new_file, split_obj=split_obj)
    aggregate_fp(fp_dir, new_file, split_obj=split_obj)
    eval_hotclip_pose6d(pred_file=new_file, split=split, skip_not_there=True, aligned=True)    

if __name__ == "__main__":
    data_dir = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP"
    fp_dir = osp.join(data_dir, "foundation_pose_unidepth")
    Fire(eval_fp)