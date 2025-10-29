import pickle
import os
from .eval_joints import get_hotclip_split, eval_hotclip
import numpy as np
import os.path as osp


def post_process_hawor(save_file="", new_file="", split="test", valid_only=False, ):
    seqs = get_hotclip_split(split)


    pred_data = {}
    with open(save_file, "rb") as f:
        hawor_data = pickle.load(f)

    for seq in seqs:
        hawor_pred = hawor_data[seq]

        if valid_only:
            left_valid = hawor_pred["left_hand_valid"]
            right_valid = hawor_pred["right_hand_valid"]
            if not np.all(left_valid) or not np.all(right_valid):
                print(f"Seq {seq} has invalid frames")
                continue
        else:
            infill(hawor_pred, "left")
            infill(hawor_pred, "right")

        pred_data[seq] = hawor_pred

    os.makedirs(osp.dirname(new_file), exist_ok=True)
    with open(new_file, "wb") as f:
        pickle.dump(pred_data, f)
    print(f"Saved to {new_file}" , len(pred_data))
    
    return 

def infill(hawor_pred, side):
    # start or end of the trajectory, there may be invalid frames. use previous valid frame to infill
    T = len(hawor_pred[f"{side}_hand_theta"])
    valid = hawor_pred[f"{side}_hand_valid"]

    # infill the ending frames
    for t in range(1, T):
        if not valid[t] and valid[t-1]:
            hawor_pred[f"{side}_hand_theta"][t] = hawor_pred[f"{side}_hand_theta"][t-1]
            hawor_pred[f"{side}_hand_shape"][t] = hawor_pred[f"{side}_hand_shape"][t-1]
            valid[t] = True

    # infill the starting frames
    for t in range(T-2, -1, -1):
        if not valid[t] and valid[t+1]:
            hawor_pred[f"{side}_hand_theta"][t] = hawor_pred[f"{side}_hand_theta"][t+1]
            hawor_pred[f"{side}_hand_shape"][t] = hawor_pred[f"{side}_hand_shape"][t+1]
            valid[t] = True

    if np.all(hawor_pred[f"{side}_hand_theta"][0] == 0):
        print("??? the whole trajectory is invalid???")


if __name__ == "__main__":
    data_dir = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP"
    save_file = osp.join(data_dir, "preprocess/dataset_contact_patched_hawor_gtcamTrue.pkl")
    new_file = osp.join(data_dir, "eval/hawor_test_camTrue.pkl")
    post_process_hawor(save_file, new_file, valid_only=True)

    gt_file = osp.join(data_dir, "preprocess/dataset_contact.pkl")
    
    eval_hotclip(new_file, gt_file, split="test", skip_not_there=True, side="both", chunk_length=-1)
