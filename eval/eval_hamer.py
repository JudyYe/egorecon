"""Usage: 
python -m eval.eval_hamer \
    --pred_dir /move/u/yufeiy2/hamer/out_demo/"""
import glob
import os
import os.path as osp
import pickle
from typing import Optional

import numpy as np
import torch
from fire import Fire

from eval.eval_hoi import eval_hotclip_joints


gt2hamer = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
# hamer2gt = {j: i for i, j in enumerate(gt2hamer)}
hamer2gt = [0] * 21
for i, j in enumerate(gt2hamer):
    hamer2gt[j] = i



def _load_hamer_predictions(pred_dir: str) -> dict:
    pred_dir = osp.abspath(pred_dir)
    if not osp.isdir(pred_dir):
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    npz_files = sorted(glob.glob(osp.join(pred_dir, "*.npy")))
    if not npz_files:
        raise FileNotFoundError(f"No '.npz' files found under {pred_dir}")

    pred_data = {}
    for npz_path in npz_files:
        seq_name = osp.splitext(osp.basename(npz_path))[0]
        seq_name = seq_name.split('.')[0]
        seq_name = seq_name.replace("clip-", "")
        joints = np.load(npz_path)
        joints = joints.reshape(-1, 2, 21, 3)

        joints = joints[:, :, hamer2gt, :]
        print("joints", joints.shape)
        # with np.load(npz_path) as data:
        #     if "wJoints" not in data:
        #         raise KeyError(f"wJoints not found in {npz_path}")
        #     joints = data["wJoints"]
        if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
            raise ValueError(
                f"Expected wJoints with shape (T, 2, 21, 3); got {joints.shape} in {npz_path}"
            )
        joints = joints.astype(np.float32)
        left = torch.from_numpy(joints[:, 0])  # (T, 21, 3)
        right = torch.from_numpy(joints[:, 1])
        if joints.shape[0] != 150:
            continue
        pred_data[seq_name] = {
            "left_joints": left,
            "right_joints": right,
        }
    return pred_data


def eval_hamer(
    pred_dir: str,
    gt_file: str = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
    split: str = "test50obj",
    side: str = "both",
    save_dir: Optional[str] = None,
    chunk_length: int = -1,
    output_pickle: Optional[str] = None,
):
    """Evaluate HAMER predictions (seq.npz files containing wJoints) using eval_hotclip_joints."""
    pred_data = _load_hamer_predictions(pred_dir)

    if save_dir is None:
        save_dir = osp.join(osp.abspath(pred_dir), "eval")
    os.makedirs(save_dir, exist_ok=True)

    if output_pickle is not None:
        output_pickle = osp.abspath(output_pickle)
        os.makedirs(osp.dirname(output_pickle), exist_ok=True)
        with open(output_pickle, "wb") as f:
            pickle.dump(pred_data, f)
        print(f"Saved reformatted joints to {output_pickle}")

    print("pred_data", )
    return eval_hotclip_joints(
        pred_file=pred_data,
        gt_file=gt_file,
        side=side,
        save_dir=save_dir,
        split=split,
        skip_not_there=True,
        chunk_length=chunk_length,
        force_fk=False,
    )


def main(**kwargs):
    return eval_hamer(**kwargs)


if __name__ == "__main__":
    Fire(main)
