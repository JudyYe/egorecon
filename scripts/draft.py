import json
import os
import os.path as osp
import shutil


def merge():
    """
    Merge data/HOT3D-CLIP/sets/split.json and data/HOT3D-CLIP/sets/split_obj.json into data/HOT3D-CLIP/sets/split.json.
    Copy split.json to split_old.json as backup before merging.
    Asserts that there are NO overlapping keys to avoid overwriting existing splits.
    """
    base_dir = "data/HOT3D-CLIP/sets"
    split_file = osp.join(base_dir, "split.json")
    split_obj_file = osp.join(base_dir, "split_obj.json")
    split_old_file = osp.join(base_dir, "split_old.json")
    
    # Load both files
    if not osp.exists(split_file):
        raise FileNotFoundError(f"split.json not found at {split_file}")
    if not osp.exists(split_obj_file):
        raise FileNotFoundError(f"split_obj.json not found at {split_obj_file}")
    
    with open(split_file, "r") as f:
        split_data = json.load(f)
    
    with open(split_obj_file, "r") as f:
        split_obj_data = json.load(f)
    
    # Assert that there are NO overlapping keys (to avoid overwriting existing splits)
    split_keys = set(split_data.keys())
    split_obj_keys = set(split_obj_data.keys())
    overlapping_keys = split_keys & split_obj_keys  # Intersection
    
    if overlapping_keys:
        raise ValueError(
            f"Overlapping keys found! Both files have keys: {overlapping_keys}. "
            f"split.json has keys: {split_keys}, "
            f"split_obj.json has keys: {split_obj_keys}. "
            f"Cannot merge without overwriting existing splits."
        )
    
    print(f"✓ No overlapping keys. split.json has: {split_keys}, split_obj.json has: {split_obj_keys}")
    
    # Backup split.json to split_old.json
    shutil.copy2(split_file, split_old_file)
    print("✓ Backed up split.json to split_old.json")
    
    # Merge: combine both dictionaries (no overlapping keys, so no overwriting)
    merged_data = {}
    # Add all entries from split.json
    for key in split_keys:
        merged_data[key] = split_data[key]
        print(f"✓ Added {key}: {len(split_data[key])} entries from split.json")
    # Add all entries from split_obj.json (which have seq_objid format)
    for key in split_obj_keys:
        merged_data[key] = split_obj_data[key]
        print(f"✓ Added {key}: {len(split_obj_data[key])} entries from split_obj.json")
    
    # Save merged data to split.json
    os.makedirs(base_dir, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"✓ Saved merged data to {split_file}")
    print("\nMerged splits:")
    for key, values in merged_data.items():
        print(f"  {key}: {len(values)} entries")
        if len(values) > 0:
            print(f"    Example: {values[0]}")


if __name__ == "__main__":
    merge()