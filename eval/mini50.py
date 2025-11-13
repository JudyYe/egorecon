from fire import Fire
import os
import os.path as osp
import json
import numpy as np
import pickle
from egorecon.manip.data.utils import load_pickle
# Extract sequence names from filenames
# Format: 001874_000007.pkl --> 001874

# Original filenames data
filenames_data = """001874_000007.pkl  001909_000011.pkl  001999_000018.pkl  002451_000017.pkl  002743_000016.pkl  002977_000016.pkl  003253_000023.pkl
001878_000002.pkl  001911_000017.pkl  002007_000011.pkl  002483_000012.pkl  002850_000017.pkl  003030_000007.pkl  003255_000023.pkl
001880_000004.pkl  001917_000017.pkl  002031_000012.pkl  002488_000002.pkl  002858_000003.pkl  003034_000019.pkl
001881_000024.pkl  001921_000017.pkl  002040_000012.pkl  002517_000005.pkl  002858_000011.pkl  003076_000018.pkl
001882_000030.pkl  001923_000018.pkl  002040_000015.pkl  002602_000006.pkl  002862_000006.pkl  003078_000017.pkl
001890_000030.pkl  001974_000005.pkl  002043_000014.pkl  002604_000006.pkl  002895_000001.pkl  003125_000004.pkl
001894_000023.pkl  001981_000006.pkl  002200_000027.pkl  002625_000028.pkl  002897_000031.pkl  003132_000020.pkl
001896_000001.pkl  001997_000012.pkl  002444_000018.pkl  002635_000028.pkl  002900_000028.pkl  003150_000006.pkl
"""

def get_mini50_sequence_names():

    # Extract all filenames and get unique sequence names
    sequence_names = set()
    for line in filenames_data.strip().split('\n'):
        if line.strip():
            # Split by whitespace and extract filenames
            filenames = line.strip().split()
            for filename in filenames:
                # Extract sequence name (before underscore)
                if '_' in filename:
                    seq_name = filename.split('_')[0]
                    sequence_names.add(seq_name)

    # Sort and create list
    sequence_names = sorted(sequence_names)
    for seq_name in sequence_names:
        print(seq_name)
    
    # Also print as a list/array format
    print(f"\n# Total: {len(sequence_names)} sequences")
    print(f"sequence_names = {sequence_names}")
    
    split_file = osp.join("data/HOT3D-CLIP/sets", "split.json")
    with open(split_file, "r") as f:
        split_dict = json.load(f)
    split_dict["test50"] = sequence_names
    with open(split_file, "w") as f:
        json.dump(split_dict, f, indent=4)


def get_mini50_seq_obj():
    """
    Extract {seq}_{obj} pairs from filenames_data and return split dictionary.
    
    Returns:
        Dictionary with split "test50obj" containing list of seq_obj pairs
    """
    # Extract all filenames and get unique sequence_objid pairs
    sequence_objid_pairs = set()
    for line in filenames_data.strip().split('\n'):
        if line.strip():
            # Split by whitespace and extract filenames
            filenames = line.strip().split()
            for filename in filenames:
                # Extract sequence_objid (remove .pkl extension, keep seq_objid format)
                if '_' in filename:
                    # Format: 001874_000007.pkl --> 001874_000007
                    seq_objid = filename.replace('.pkl', '')
                    sequence_objid_pairs.add(seq_objid)
    
    # Sort and create list
    sequence_objid_pairs = sorted(sequence_objid_pairs)
    
    # Load existing split_obj.json if it exists, otherwise create new dict
    split_file = osp.join("data/HOT3D-CLIP/sets", "split_obj.json")
    if osp.exists(split_file):
        with open(split_file, "r") as f:
            split2seq_obj = json.load(f)
    else:
        split2seq_obj = {}
    
    # Update with test50obj split
    split2seq_obj["test50obj"] = sequence_objid_pairs
    
    # Save to split_obj.json
    os.makedirs(osp.dirname(split_file), exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split2seq_obj, f, indent=4)

    print("Sequence-Object ID pairs:")
    for seq_objid in split2seq_obj["test50obj"]:
        print(seq_objid)
    print(f"\n# Total: {len(split2seq_obj['test50obj'])} sequence-object pairs")    
    return split2seq_obj


def get_mini5_seq_obj():
    "valid sequences are in "
    # 001874/
    # 002043/
    # 003034/
    split_file = osp.join("data/HOT3D-CLIP/sets", "split.json")

    with open(split_file, "r") as f:
        split_dict = json.load(f)
    split_dict["testmini3"] = ["001874", "002043", "003034"]
    with open(split_file, "w") as f:
        json.dump(split_dict, f, indent=4)



def mark_trunc_contact_frames(split="test50obj"):
    data_root = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP"
    split_file = osp.join(data_root, "sets", "split.json")
    gt_data_file = osp.join(data_root, "preprocess", "dataset_contact.pkl")
    mask_dir = osp.join(data_root, "gt_mask")

    frame_subsplit = {}

    with open(split_file, "r") as f:
        split_dict = json.load(f)
    seq_obj_list = split_dict[split]

    gt_data = load_pickle(gt_data_file)

    processed_keys = set()
    for seq_obj in seq_obj_list:
        if "_" in seq_obj:
            seq, obj = seq_obj.split("_", 1)
            candidates = [(seq, obj)]
        else:
            seq = seq_obj
            seq_meta = gt_data.get(seq)
            if seq_meta is None:
                raise KeyError(f"Sequence '{seq}' missing in GT metadata.")
            candidates = [(seq, obj_key) for obj_key in seq_meta["objects"].keys()]

        for seq, obj in candidates:
            seq_meta = gt_data[seq]
            obj_meta = seq_meta["objects"]
            obj_key = obj
            frame_key = f"{seq}_{obj_key}"
            if frame_key in processed_keys:
                continue

            contact_lr = np.asarray(obj_meta[obj_key]["contact_lr"])

            contact = (contact_lr > 0.5).any(axis=-1).astype(np.int32)
            T = contact.shape[0]

            truncate = np.zeros(T, dtype=np.int32)
            mask_file = osp.join(mask_dir, f"{seq}.npz")

            if osp.exists(mask_file):
                mask_data = np.load(mask_file, allow_pickle=True)
                hand_obj_mask = mask_data["hand_obj_mask"]  # (2+O, T, H, W)
                index = mask_data["index"].tolist()

                obj_idx = index.index(obj_key)
                obj_mask = hand_obj_mask[obj_idx]  # (T, H, W)
                obj_mask = obj_mask.astype(bool)

                edge_slices = (
                    obj_mask[:, 0, :],
                    obj_mask[:, -1, :],
                    obj_mask[:, :, 0],
                    obj_mask[:, :, -1],
                )

                has_pixels = obj_mask.reshape(T, -1).any(axis=1)
                touches_edge = np.logical_or.reduce([edge.any(axis=1) for edge in edge_slices])

                truncate = np.where(~has_pixels, 2, truncate)
                truncate = np.where(has_pixels & touches_edge & (truncate == 0), 1, truncate)
            frame_subsplit[frame_key] = {
                "contact": contact.astype(int).tolist(),
                "truncate": truncate.astype(int).tolist(),
            }
            processed_keys.add(frame_key)

    output_file = osp.join(data_root, "sets", f"frame_subsplit_{split}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(frame_subsplit, f)

    print(f"Saved frame subsplit to {output_file}")
    return frame_subsplit


def make_teaser_split():
    with open(osp.join("data/HOT3D-CLIP/sets", "split.json"), "r") as f:
        split_dict = json.load(f)
    clip_list = list(range(1905, 1924))
    clip_list = [f"{clip:06d}" for clip in clip_list]

    split_dict["teaser"] = clip_list
    with open(osp.join("data/HOT3D-CLIP/sets", "split.json"), "w") as f:
        json.dump(split_dict, f, indent=4)

    return 


if __name__ == "__main__":
    # Get and print sequence-object pairs
    # split2seq_obj = get_mini50_seq_obj()
    # get_mini5_seq_obj()
    # Fire(mark_trunc_contact_frames)

    make_teaser_split()

