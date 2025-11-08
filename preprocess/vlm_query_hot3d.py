
# 001874  001917  002043  002635  003034
"""Utilities to run VLM contact queries over HOT3D videos one-by-one."""

from __future__ import annotations
from tqdm import tqdm
import argparse
import os
import os.path as osp
import json
from glob import glob
from typing import Dict, Iterable, List, Optional

import numpy as np

from preprocess.eval_vlm import (
    GPT5_MODEL,
    DEFAULT_VLM_MODEL,
    SUPPORTED_VLM_MODELS,
    create_one_folder,
    create_overlay,
    evaluate_vlm_predictions,
    query_image_list,
    save_dir,
    train_index_list,
)
from egorecon.manip.data.utils import load_pickle


video_list = ["001917", "001874", "002043", "002635", "003034"]


def _load_dataset_cache() -> Dict[str, dict]:
    """Load the HOT3D contact dataset once for reuse."""

    from preprocess.eval_vlm import data_dir  # local import to avoid circular at module load

    data_file = osp.join(data_dir, "preprocess", "dataset_contact.pkl")
    dataset = load_pickle(data_file)
    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dataset_contact.pkl to load as dict, got {type(dataset)!r}")
    return dataset


def _ensure_sequence_assets(
    seq: str,
    dataset: Dict[str, dict],
    *,
    force_prepare: bool = False,
    force_overlay: bool = False,
) -> List[str]:
    """Make sure annotation, GT, and overlay frames exist for a sequence.

    Returns sorted absolute overlay image paths for every frame.
    """

    if seq not in dataset:
        raise KeyError(f"Sequence '{seq}' not found in dataset_contact.pkl")

    seq_dir_rel = osp.join(save_dir, seq)
    annotation_path = osp.join(seq_dir_rel, "annotation.npz")
    contact_path = osp.join(seq_dir_rel, "gt_contact.npz")

    if force_prepare or not osp.exists(annotation_path) or not osp.exists(contact_path):
        create_one_folder(seq, t_index=0, save_dir=save_dir, all_contact_data=dataset, all_data=dataset)

    if not osp.exists(annotation_path):
        raise FileNotFoundError(f"annotation.npz missing after preparation for {seq}: {annotation_path}")
    if not osp.exists(contact_path):
        raise FileNotFoundError(f"gt_contact.npz missing after preparation for {seq}: {contact_path}")

    with np.load(annotation_path, allow_pickle=True) as annotation_data:
        total_frames = int(annotation_data["mask"].shape[0])

    overlay_dir = osp.join(seq_dir_rel, "overlay")
    overlay_paths = sorted(glob(osp.join(overlay_dir, "*.jpg")))

    if force_overlay or len(overlay_paths) != total_frames:
        create_overlay(seq)
        overlay_paths = sorted(glob(osp.join(overlay_dir, "*.jpg")))

    if len(overlay_paths) != total_frames:
        raise RuntimeError(
            f"Overlay generation mismatch for {seq}: expected {total_frames} frames, got {len(overlay_paths)}"
        )

    return [osp.abspath(path) for path in overlay_paths]


def _subsample_overlay_paths(paths: List[str], every_n: int) -> List[str]:
    if every_n <= 1:
        return paths
    subset = paths[::every_n]
    if paths and subset and subset[-1] != paths[-1]:
        subset.append(paths[-1])
    return subset


def _has_full_predictions(predictions_root: str, seq: str, expected_frames: int) -> bool:
    pattern = osp.join(predictions_root, seq, "*_prediction.json")
    files = glob(pattern)
    return len(files) == expected_frames


def process_video(
    seq: str,
    dataset: Dict[str, dict],
    *,
    output_root: str,
    example_dir: str,
    train_indices: Optional[Iterable[str]] = None,
    model: str = DEFAULT_VLM_MODEL,
    temperature: float = 0.0,
    force_prepare: bool = False,
    force_overlay: bool = False,
    overwrite_predictions: bool = False,
    query_every_n_frames: int = 10,
):
    """Run the full pipeline (prepare → query → evaluate) for a single video."""

    print(f"\n=== Processing sequence {seq} ===")

    overlay_paths = _ensure_sequence_assets(
        seq,
        dataset,
        force_prepare=force_prepare,
        force_overlay=force_overlay,
    )
    overlay_paths = _subsample_overlay_paths(overlay_paths, query_every_n_frames)
    expected_frames = len(overlay_paths)

    if expected_frames == 0:
        print(f"No overlay frames found for {seq}; skipping.")
        return None

    predictions_root = osp.abspath(osp.join(output_root, seq))
    os.makedirs(predictions_root, exist_ok=True)

    already_done = _has_full_predictions(predictions_root, seq, expected_frames)
    if already_done and not overwrite_predictions:
        print(f"Skipping VLM query for {seq} (found {expected_frames} predictions).")
    else:
        print(f"Querying VLM for {seq}: {expected_frames} frames using model '{model}' …")
        query_image_list(
            overlay_paths,
            vlm_output_dir=predictions_root,
            model=model,
            temperature=temperature,
            train_indices=list(train_indices) if train_indices else None,
            example_dir=example_dir,
        )

    if not _has_full_predictions(predictions_root, seq, expected_frames):
        print(f"No complete predictions found for {seq}; skipping evaluation.")
        return None

    print(f"Evaluating predictions for {seq} …")
    summary = evaluate_vlm_predictions(predictions_root)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query HOT3D videos with VLM contact detector.")
    # parser.add_argument(
    #     "--videos",
    #     nargs="+",
    #     default=video_list,
    #     help="Sequences to process (default: predefined video_list).",
    # )
    parser.add_argument(
        "--model",
        default=GPT5_MODEL,
        choices=sorted(SUPPORTED_VLM_MODELS),
        help="VLM model to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for query_image_list (will be clamped if unsupported).",
    )
    parser.add_argument(
        "--output-root",
        default=osp.join(save_dir, "vlm_output_hot3d"),
        help="Directory to store VLM predictions and metrics.",
    )
    parser.add_argument(
        "--example-dir",
        default=osp.join(save_dir + "_trainset", "examples"),
        help="Directory containing ICL examples (JSON + overlay).",
    )
    parser.add_argument(
        "--train-indices",
        nargs="*",
        default=None,
        help="Override train_index_list with custom seq_frame indices.",
    )
    parser.add_argument(
        "--query-every-n-frames",
        type=int,
        default=10,
        help="Stride for selecting frames to query (1 = all frames).",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Force re-creating structured data folders (images, annotation, GT).",
    )
    parser.add_argument(
        "--force-overlay",
        action="store_true",
        help="Force re-generating overlay images even if they exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run VLM queries even if predictions already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # use test50 
    split_file = osp.join("data/HOT3D-CLIP/sets", "split.json")
    with open(split_file, "r") as f:
        split_dict = json.load(f)
    args.videos = split_dict["test50"]

    videos = [str(v) for v in args.videos]
    train_indices = args.train_indices if args.train_indices is not None else train_index_list

    example_dir = osp.abspath(args.example_dir)
    if not osp.isdir(example_dir):
        print(f"Warning: example_dir '{example_dir}' does not exist. Few-shot ICL will be skipped.")
        example_dir = example_dir  # allow query_image_list to handle missing directory

    dataset = _load_dataset_cache()

    summaries = {}
    
    for seq in tqdm(videos):
        # done_file = osp.join(args.output_root, "done", f"{seq}.done")
        done_file = osp.join(args.output_root, seq, "metrics_summary.json")
        lock_file = osp.join(args.output_root, "lock", f"{seq}.lock")

        if osp.exists(done_file):
            continue
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            continue

        try:
            summary = process_video(
                seq,
                dataset,
                output_root=osp.abspath(args.output_root),
                example_dir=example_dir,
                train_indices=train_indices,
                model=args.model,
                temperature=args.temperature,
                force_prepare=args.force_prepare,
                force_overlay=args.force_overlay,
                overwrite_predictions=args.overwrite,
                query_every_n_frames=max(1, args.query_every_n_frames),
            )
            if summary is not None:
                summaries[seq] = summary
        except Exception as exc:  # pragma: no cover - CLI convenience
            print(f"[ERROR] Failed to process {seq}: {exc}")
        
        # os.makedirs(done_file)
        os.rmdir(lock_file)

    if summaries:
        print("\n=== Summary ===")
        for seq, summary in summaries.items():
            metrics = summary.get("overall", {}).get("metrics", {})
            f1 = metrics.get("f1")
            bal_acc = metrics.get("balanced_accuracy")
            print(
                f"{seq}: F1={f1:.4f} BalancedAcc={bal_acc:.4f}" if (f1 is not None and bal_acc is not None)
                else f"{seq}: metrics summary written to metrics_summary.json"
            )


def load_vlm_contact(seq, obj_id, T, vlm_output_dir=osp.join(save_dir, "vlm_output_hot3d")):
    predictions_root = osp.abspath(osp.join(vlm_output_dir, seq, seq))
    contact = np.zeros((T, 2)) - 1

    glob_pattern = osp.join(predictions_root, "*_prediction.json")
    pred_files = sorted(glob(glob_pattern))
    obj_id_str = str(obj_id)

    for pred_file in pred_files:
        with open(pred_file, "r") as f:
            prediction_data = json.load(f)
        frame_idx = prediction_data.get("frame_idx")
        frame_idx = int(frame_idx)

        predictions = prediction_data.get("predictions", [])

        for item in predictions:
            if str(item.get("object_name")) != obj_id_str:
                continue
            left_val = item.get("left_hand_contact")
            right_val = item.get("right_hand_contact")
            if left_val in (0, 1):
                contact[frame_idx, 0] = int(left_val)
            if right_val in (0, 1):
                contact[frame_idx, 1] = int(right_val)
            break

    return contact


if __name__ == "__main__":
    main()