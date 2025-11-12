import argparse
import json
import os
import os.path as osp
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import cv2
import imageio
import numpy as np

CLIP_DEFINITIONS_PATH = "data/HOT3D-CLIP/clip_definitions.json"
DEFAULT_SPLIT_FILE = "data/HOT3D-CLIP/sets/split.json"
DEFAULT_DATASET_CONTACT = "data/HOT3D-CLIP/preprocess/dataset_contact.pkl"
DEFAULT_IMAGES_ROOT = "data/HOT3D-CLIP/extract_images-rot90"
DEFAULT_OUTPUT_DIR = "outputs/debug_teaser"


def _load_split_list(split_path: str, split_name: str | None) -> List[str]:
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    if isinstance(split_data, list):
        return [f"{int(item):06d}" for item in split_data]

    if not isinstance(split_data, dict):
        raise ValueError(f"Unexpected split file format: {type(split_data)}")

    if split_name is None:
        raise ValueError(
            "split_name must be provided when the split file contains multiple entries"
        )

    if "/" in split_name:
        split_key, device_key = split_name.split("/", 1)
    else:
        split_key, device_key = split_name, None

    if split_key not in split_data:
        raise KeyError(f"Split '{split_key}' not found in {split_path}")

    entry = split_data[split_key]
    clip_ids: List[int] = []

    if isinstance(entry, dict):
        if device_key is None:
            for clips in entry.values():
                clip_ids.extend(int(c) for c in clips)
        else:
            if device_key not in entry:
                raise KeyError(
                    f"Device '{device_key}' not found under split '{split_key}'"
                )
            clip_ids.extend(int(c) for c in entry[device_key])
    elif isinstance(entry, list):
        clip_ids.extend(int(c) for c in entry)
    else:
        raise ValueError(
            f"Unrecognised entry type under split '{split_key}': {type(entry)}"
        )

    return [f"{clip_id:06d}" for clip_id in clip_ids]


def _format_frames(num_frames: int) -> str:
    return f"{num_frames:d} frames"


def load_seq2video(definitions_path: str = CLIP_DEFINITIONS_PATH) -> Dict[str, Dict]:
    with open(definitions_path, "r", encoding="utf-8") as f:
        clip_defs = json.load(f)

    mapping: Dict[str, Dict] = {}
    for clip_id_str, info in clip_defs.items():
        try:
            clip_num = int(clip_id_str)
        except ValueError:
            continue
        seq_key = f"{clip_num:06d}"
        mapping[seq_key] = {
            "clip_id": clip_id_str,
            "sequence_id": info["sequence_id"],
            "device": info.get("device"),
            "per_frame_timestamps_ns": info.get("per_frame_timestamps_ns", []),
        }
    return mapping


def find_dynamic_objects(seq_meta: Dict, motion_threshold: float = 0.05) -> Set[str]:
    dynamic_objects: Set[str] = set()
    if seq_meta is None:
        return dynamic_objects

    object_ids = seq_meta.get("objects", [])
    if object_ids is None:
        return dynamic_objects

    keypoints_local = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
        ],
        dtype=np.float32,
    )

    for obj_id in object_ids:
        wto_key = f"obj_{obj_id}_wTo"
        if wto_key not in seq_meta:
            continue
        wto = np.asarray(seq_meta[wto_key], dtype=np.float32)
        if wto.ndim != 3 or wto.shape[1:] != (4, 4) or len(wto) < 2:
            continue

        positions = wto[:, :3, 3]
        rotations = wto[:, :3, :3]

        keypoints_world = (rotations @ keypoints_local.T).transpose(0, 2, 1)
        keypoints_world = keypoints_world + positions[:, np.newaxis, :]

        displacements = np.linalg.norm(np.diff(keypoints_world, axis=0), axis=2)
        total_motion = displacements.mean(axis=1).sum()

        if total_motion >= motion_threshold:
            dynamic_objects.add(obj_id)

    return dynamic_objects


def aggregate_sequences(
    split_file: str,
    dataset_contact_path: str,
    split_name: str,
) -> Dict[str, Dict]:
    clip_ids = _load_split_list(split_file, split_name)
    with open(dataset_contact_path, "rb") as f:
        dataset_contact: Dict[str, Dict] = pickle.load(f)

    seq2video = load_seq2video()

    video_map: Dict[str, Dict] = defaultdict(
        lambda: {
            "clip_ids": [],
            "clip_lengths": [],
            "object_sets": [],
            "dynamic_objects": set(),
            "clip_metadata": {},
        }
    )

    for clip_seq in clip_ids:
        seq_meta = dataset_contact.get(clip_seq)
        if seq_meta is None:
            continue

        video_info = seq2video.get(clip_seq)
        if video_info is None:
            continue

        video_name = video_info["sequence_id"]
        dynamic_objects = find_dynamic_objects(seq_meta)

        video_entry = video_map[video_name]
        video_entry["clip_ids"].append(clip_seq)
        frame_count = len(seq_meta.get("wTc", []))
        video_entry["clip_lengths"].append(frame_count)
        video_entry["object_sets"].append(set(seq_meta.get("objects", [])))
        video_entry["dynamic_objects"].update(dynamic_objects)

        video_entry["clip_metadata"][clip_seq] = {
            "clip_id": video_info["clip_id"],
            "length": frame_count,
            "per_frame_timestamps_ns": video_info["per_frame_timestamps_ns"],
        }

    return video_map


def rank_videos(video_map: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    scored_videos = [
        (
            video_name,
            {
                "dynamic_count": len(entry["dynamic_objects"]),
                "total_objects": len(set().union(*entry["object_sets"]))
                if entry["object_sets"]
                else 0,
                "entry": entry,
            },
        )
        for video_name, entry in video_map.items()
    ]

    scored_videos.sort(
        key=lambda item: (
            item[1]["dynamic_count"],
            item[1]["total_objects"],
            len(item[1]["entry"]["clip_ids"]),
        ),
        reverse=True,
    )
    return scored_videos


def summarise_videos(ranked_videos: List[Tuple[str, Dict]], top: int | None = None) -> None:
    subset = ranked_videos if top is None else ranked_videos[:top]
    for video_name, info in subset:
        entry = info["entry"]
        print(
            f"Sequence {video_name}: {info['dynamic_count']} dynamic objects (total objects: {info['total_objects']})"
        )
        for clip_seq_id in sorted(entry["clip_metadata"].keys()):
            clip_meta = entry["clip_metadata"][clip_seq_id]
            clip_id = int(clip_meta["clip_id"])
            length = clip_meta["length"]
            timestamps = clip_meta["per_frame_timestamps_ns"]
            sensor_key = "214-1"
            sensor_times = [
                frame.get(sensor_key)
                for frame in timestamps
                if isinstance(frame, dict) and sensor_key in frame
            ]
            preview = (
                f" | 214-1 timestamps: {sensor_times[:5]}..."
                if sensor_times
                else ""
            )
            print(
                f"  Clip {clip_id:06d} (seq {clip_seq_id}): {_format_frames(length)}{preview}"
            )
        print()


def _prepare_frame(image_path: str, base_shape: Tuple[int, int] | None, label: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = cv2.imread(image_path)
    if img is None:
        if base_shape is None:
            base_shape = (480, 640)
        h, w = base_shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
    shape = img.shape[:2]
    if base_shape is None:
        base_shape = shape
    cv2.putText(
        img,
        label,
        (10, img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return img, base_shape


def create_teaser_videos(
    ranked_videos: List[Tuple[str, Dict]],
    images_root: str,
    output_dir: str,
    top: int | None,
    fps: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    subset = ranked_videos if top is None else ranked_videos[:top]

    for video_name, info in subset:
        entry = info["entry"]
        clip_seqs = sorted(entry["clip_ids"])
        if not clip_seqs:
            continue

        clip_info_list = []
        for clip_seq in clip_seqs:
            clip_meta = entry["clip_metadata"].get(clip_seq)
            if clip_meta is None:
                continue
            clip_id = int(clip_meta["clip_id"])
            clip_folder = osp.join(images_root, f"clip-{clip_id:06d}")
            clip_length = clip_meta["length"] or len(clip_meta.get("per_frame_timestamps_ns", []))
            if clip_length <= 0:
                continue
            clip_info_list.append((clip_seq, clip_id, clip_folder, clip_length))

        if not clip_info_list:
            continue

        base_shape = None
        sample_img = None
        for _, clip_id, clip_folder, _ in clip_info_list:
            first_path = osp.join(clip_folder, "0000.jpg")
            sample_img = cv2.imread(first_path)
            if sample_img is not None:
                base_shape = sample_img.shape[:2]
                break
        if base_shape is None:
            base_shape = (480, 640)

        output_path = osp.join(output_dir, f"{video_name}.mp4")
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
        )

        for clip_seq, clip_id, clip_folder, clip_length in clip_info_list:
            for frame_idx in range(clip_length):
                image_path = osp.join(clip_folder, f"{frame_idx:04d}.jpg")
                label = f"{clip_id:06d}-{frame_idx:04d}"
                frame, base_shape = _prepare_frame(image_path, base_shape, label)
                if frame.shape[:2] != base_shape:
                    frame = cv2.resize(frame, (base_shape[1], base_shape[0]))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame_rgb)
        writer.close()
        print(f"Saved teaser video to {output_path}")


def main(
    split_file: str,
    dataset_contact_path: str,
    split_name: str,
    top: int | None,
    images_root: str,
    output_dir: str,
    fps: int,
) -> None:
    video_map = aggregate_sequences(split_file, dataset_contact_path, split_name)
    ranked = rank_videos(video_map)
    summarise_videos(ranked, top=top)
    create_teaser_videos(ranked, images_root, output_dir, top, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find sequences with multiple dynamic objects")
    parser.add_argument("--split_file", default=DEFAULT_SPLIT_FILE, help="Path to JSON clip split definition")
    parser.add_argument(
        "--dataset_contact_path",
        default="data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
        help="Path to dataset_contact.pkl",
    )
    parser.add_argument(
        "--split-name",
        dest="split_name",
        default="test",
        help="Split/device identifier, e.g. 'train/Aria'",
    )
    parser.add_argument(
        "--images-root",
        default=DEFAULT_IMAGES_ROOT,
        help="Root directory containing clip image folders",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save teaser videos",
    )
    parser.add_argument("--top", type=int, help="Only display/render the top-N sequences")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for teaser videos")
    args = parser.parse_args()

    main(
        args.split_file,
        args.dataset_contact_path,
        args.split_name,
        args.top,
        args.images_root,
        args.output_dir,
        args.fps,
    )
