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
    mean.append(obj_mean)
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
    else:
        raise NotImplementedError(f"Invalid hand representation: {hand_rep}")
    hand_mean = np.concatenate([left_mean, right_mean], axis=-1)
    hand_std = np.concatenate([left_std, right_std], axis=-1)

    if hand == 'out':
        mean.append(hand_mean)
        std.append(hand_std)


    contact = opt.output.contact

    if contact:
        contact_mean = np.array([[0, 0]])
        contact_std = np.array([[1, 1]])
        mean.append(contact_mean)
        std.append(contact_std)

    if field == 'target':
        mean = np.concatenate(mean, axis=-1)
        std = np.concatenate(std, axis=-1)
    else:
        if hand == 'out':
            mean = np.zeros([1, 0, ])
            std = np.zeros([1, 0, ])
        elif hand == 'cond':
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