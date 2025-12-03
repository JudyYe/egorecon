import os.path as osp
import torch
import pickle
import numpy as np


def get_norm_stats(metafile, opt, field='target'):
    if isinstance(metafile, str):
        with open(metafile, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = metafile

    mean = [] 
    std = []

    obj_mean = metadata["object_mean"]
    obj_std = metadata["object_std"]
    mean.append(obj_mean) # target
    std.append(obj_std)

    hand = opt.get('hand', 'cond')
    hand_rep = opt.get('hand_rep', 'joint')
    
    if hand_rep == 'theta':
        left_mean = metadata["left_hand_theta_mean"]
        left_std = metadata["left_hand_theta_std"]
        right_mean = metadata["right_hand_theta_mean"]
        right_std = metadata["right_hand_theta_std"]
    elif hand_rep == 'joint':
        left_mean = metadata["left_hand_mean"]
        left_std = metadata["left_hand_std"]
        right_mean = metadata["right_hand_mean"]
        right_std = metadata["right_hand_std"]
    elif hand_rep == 'joint_theta':
        left_mean = np.concatenate([metadata["left_hand_mean"], metadata["left_hand_theta_mean"]], axis=-1)
        left_std = np.concatenate([metadata["left_hand_std"], metadata["left_hand_theta_std"]], axis=-1)
        right_mean = np.concatenate([metadata["right_hand_mean"], metadata["right_hand_theta_mean"]], axis=-1)  
        right_std = np.concatenate([metadata["right_hand_std"], metadata["right_hand_theta_std"]], axis=-1)
    elif hand_rep == 'joint_theta_dot':
        left_mean = np.concatenate([metadata["left_hand_mean"], metadata["left_hand_theta_mean"], metadata["left_hand_mean"]], axis=-1)
        right_mean = np.concatenate([metadata["right_hand_mean"], metadata["right_hand_theta_mean"], metadata["right_hand_mean"]], axis=-1)
        left_std = np.concatenate([metadata["left_hand_std"], metadata["left_hand_theta_std"], metadata["left_hand_std"]], axis=-1)
        right_std = np.concatenate([metadata["right_hand_std"], metadata["right_hand_theta_std"], metadata["right_hand_std"]], axis=-1)
    else:
        raise NotImplementedError(f"Invalid hand representation: {hand_rep}")
    hand_mean = np.concatenate([left_mean, right_mean], axis=-1)
    hand_std = np.concatenate([left_std, right_std], axis=-1)


    # put in mean for target: hand is part of the output 
    if hand in ['out', 'cond_out']:
        mean.append(hand_mean)
        std.append(hand_std)

    contact = False
    if 'output' in opt:
        contact = opt.output.contact

    if contact:
        contact_mean = np.array([[0, 0]])
        contact_std = np.array([[1, 1]])
        mean.append(contact_mean)
        std.append(contact_std)

    if field == 'target':
        mean = np.concatenate(mean, axis=-1)
        std = np.concatenate(std, axis=-1)
    elif field == 'condition':
        if hand == 'out':  # no condition
            mean = np.zeros([1, 0, ])
            std = np.zeros([1, 0, ])
        elif hand in  ['cond', 'cond_out']:
            mean = hand_mean
            std = hand_std
        else:
            raise NotImplementedError(f"Invalid hand: {hand}")

    return mean, std


# def default_collate_fn(batch):  # 
#     from jutils import mesh_utils
#     mesh_utils.collate_meshes
#     from torch.utils.data import default_collate
#     # write a patch when type(batch) is pytroch3d.structures.meshes.Meshes. behavior: Meshes(verts_list=[verts1, verts2, ...], faces_list=[faces1, faces2, ...]) -> Meshes(verts=[verts1, verts2, ...], faces=[faces1, faces2, ...])
#     # where verts_list and faces_list from 
#     return 


def load_pickle(path, num=-1):
    """Load and return the object stored in a pickle file."""
    # with open(path, "rb") as f:
    #     return pickle.load(f)
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            all_seq = pickle.load(f)

        all_processed_data = {}
        if "right_hand" in all_seq:
            # don't need to process
            all_processed_data = all_seq
        else:
            for i, (seq, data) in enumerate(all_seq.items()):
                if num > 0 and i >= num:
                    break
                processed_data = decode_npz(data)
                all_processed_data[seq] = processed_data

    elif path.endswith(".npz"):
        # Legacy support for mini dataset
        data = np.load(path, allow_pickle=True)
        seq_index = osp.basename(path).split(".")[0]

        processed_data = decode_npz(data)

        all_processed_data = {seq_index: processed_data}

    return all_processed_data


def decode_npz(data):
    # data = np.load(path, allow_pickle=True)
    data = dict(data)
    uid_list = data.get("objects", [])
    if "objects" in data:
        data.pop("objects")
    processed_data = {}
    objects = {}
    for uid in uid_list:
        uid = str(uid)
        objects[uid] = {}
        for k in data.keys():
            if k.startswith(f"obj_{uid}"):
                new_key = k.replace(f"obj_{uid}_", "")
                objects[uid][new_key] = data[k]
        # get wTo_shelf
        if "cTo_shelf" in objects[uid]:
            cTo_shelf = objects[uid]["cTo_shelf"]
            wTc = data["wTc"]
            # cTw = np.linalg.inv(wTc)
            wTo_shelf = wTc @ cTo_shelf
            objects[uid]["wTo_shelf"] = wTo_shelf

    processed_data["left_hand"] = {}
    processed_data["right_hand"] = {}
    for k in data:
        if not k.startswith("obj_"):
            if k.startswith("left_hand"):
                processed_data["left_hand"][k.replace("left_hand_", "")] = data[k]
            elif k.startswith("right_hand"):
                processed_data["right_hand"][k.replace("right_hand_", "")] = data[k]
            else:
                processed_data[k] = data[k]

    processed_data["objects"] = objects
    return processed_data


def encode_npz(processed_data):
    """
    Reverse of decode_npz: converts processed_data dict back to npz-compatible format.
    
    Args:
        processed_data: Dictionary with structure:
            - "objects": dict of object data by UID
            - "left_hand": dict of left hand data
            - "right_hand": dict of right hand data  
            - other keys: global data
    
    Returns:
        dict: Dictionary ready to be saved as npz file
    """
    # Start with a copy of the processed data
    npz_data = {}
    
    # Add all non-nested keys directly
    for k, v in processed_data.items():
        if k not in ["objects", "left_hand", "right_hand"]:
            npz_data[k] = v
    
    # Add object UIDs list
    if "objects" in processed_data:
        npz_data["objects"] = list(processed_data["objects"].keys())
        
        # Flatten object data with obj_{uid}_ prefix
        for uid, obj_data in processed_data["objects"].items():
            uid = str(uid)
            for key, value in obj_data.items():
                npz_data[f"obj_{uid}_{key}"] = value
    
    # Flatten left hand data with left_hand_ prefix
    if "left_hand" in processed_data:
        for key, value in processed_data["left_hand"].items():
            npz_data[f"left_hand_{key}"] = value
    
    # Flatten right hand data with right_hand_ prefix  
    if "right_hand" in processed_data:
        for key, value in processed_data["right_hand"].items():
            npz_data[f"right_hand_{key}"] = value
    
    return npz_data

# in rohm:
# init(): create windows -> canonical --> add noise --> get rep


def to_tensor(array, dtype=torch.float32):
    """Convert array to tensor with specified dtype."""
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

