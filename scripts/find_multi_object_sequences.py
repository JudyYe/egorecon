"""
Find multi-object sequences and group clips into continuous segments.

Use Cases:
----------

1. Command-line usage to find and display continuous segments:
   $ python scripts/find_multi_object_sequences.py --split-name test
   
   This will:
   - Load clips from the test split
   - Group them by sequence
   - Find continuous segments within each sequence (no frame gaps)
   - Print segments sorted from longest to shortest

2. Programmatic usage to get segments:
   from scripts.find_multi_object_sequences import group_clips_into_segments
   
   segments = group_clips_into_segments(
       split_file="data/HOT3D-CLIP/sets/split.json",
       dataset_contact_path="data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
       split_name="test",
       max_gap_ns=1000000000  # 1 second gap threshold
   )
   
   # segments is a list of segments, each segment is a list of tuples:
   # (clip_name, involved_objects, frame_number)
   for segment in segments:
       for clip_name, objects, frame_num in segment:
           print(f"Clip {clip_name}, frame {frame_num}, objects: {objects}")

3. Filter for long continuous sequences:
   segments = group_clips_into_segments(...)
   long_segments = [seg for seg in segments if len(seg) > 100]  # > 100 frames
   
4. Get unique clips and objects per segment:
   for seg_idx, segment in enumerate(segments):
       clips = set(clip_name for clip_name, _, _ in segment)
       objects = set()
       for _, obj_list, _ in segment:
           objects.update(obj_list)
       print(f"Segment {seg_idx}: {len(clips)} clips, {len(objects)} objects")
"""

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


def _extract_timestamps_from_frame_data(timestamps_list: List) -> List[int]:
    """
    Extract timestamps from per_frame_timestamps_ns structure.
    Returns a list of timestamps in nanoseconds.
    """
    extracted_timestamps = []
    for frame_data in timestamps_list:
        if isinstance(frame_data, dict):
            # Try to get a timestamp from any sensor
            # Prefer "214-1" if available, otherwise use first available
            timestamp = None
            if "214-1" in frame_data:
                timestamp = frame_data["214-1"]
            else:
                # Get first available timestamp value
                for key, value in frame_data.items():
                    if isinstance(value, (int, float)):
                        timestamp = int(value)
                        break
            if timestamp is not None:
                extracted_timestamps.append(int(timestamp))
        elif isinstance(frame_data, (int, float)):
            extracted_timestamps.append(int(frame_data))
    return extracted_timestamps


def _check_timestamps_continuous(timestamps: List[int], max_gap_ns: int) -> bool:
    """
    Check if timestamps are continuous (no large gaps) and monotonically increasing.
    """
    if len(timestamps) <= 1:
        return True
    
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > max_gap_ns or gap < 0:  # gap < 0 means not monotonically increasing
            return False
    return True


def _split_clip_by_timestamps(
    clip_name: str,
    objects: List[str],
    timestamps: List[int],
    max_gap_ns: int,
) -> List[Dict]:
    """
    Split a clip into sub-clips where timestamps are continuous and monotonically increasing.
    Returns a list of clip parts, each with its own timestamp range.
    """
    if len(timestamps) == 0:
        return []
    
    parts = []
    current_part_start = 0
    
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        
        # If gap is too large or negative (not monotonically increasing), split here
        if gap > max_gap_ns or gap < 0:
            # Save current part
            part_timestamps = timestamps[current_part_start:i]
            if len(part_timestamps) > 0:
                parts.append({
                    "clip_name": clip_name,
                    "objects": objects,
                    "timestamps": part_timestamps,
                    "frame_start": current_part_start,
                    "frame_end": i - 1,
                    "num_frames": len(part_timestamps),
                })
            current_part_start = i
    
    # Add the last part
    if current_part_start < len(timestamps):
        part_timestamps = timestamps[current_part_start:]
        parts.append({
            "clip_name": clip_name,
            "objects": objects,
            "timestamps": part_timestamps,
            "frame_start": current_part_start,
            "frame_end": len(timestamps) - 1,
            "num_frames": len(part_timestamps),
        })
    
    return parts


def group_clips_into_segments(
    split_file: str,
    dataset_contact_path: str,
    split_name: str,
    max_gap_ns: int = 1000000000,  # 1 second default gap threshold
) -> List[List[Tuple[str, List[str], int]]]:
    """
    Group clips from test split into continuous segments based on timestamps.
    
    This function:
    1. Maps clip names to sequence names (using aggregate_sequences)
    2. Groups clips by sequence
    3. Within each sequence, sorts clips by timestamp
    4. Groups clips into segments where timestamps are continuous (no gaps > max_gap_ns)
    
    Args:
        split_file: Path to split JSON file (e.g., "data/HOT3D-CLIP/sets/split.json")
        dataset_contact_path: Path to dataset_contact.pkl
        split_name: Split name (e.g., "test", "train", "train/Aria")
        max_gap_ns: Maximum gap in nanoseconds to consider clips continuous (default: 1 second)
    
    Returns:
        List of segments, where each segment is a list of tuples:
        (clip_name, involved_objects, frame_number)
        
        Example:
        [
            [
                ("000123", ["000007", "000008"], 0),
                ("000123", ["000007", "000008"], 1),
                ("000124", ["000007", "000008"], 0),
                ...
            ],
            [
                ("000125", ["000009"], 0),
                ...
            ]
        ]
        
    Use Case:
        # Get all continuous segments from test split
        segments = group_clips_into_segments(
            split_file="data/HOT3D-CLIP/sets/split.json",
            dataset_contact_path="data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
            split_name="test"
        )
        
        # Find longest segments
        longest = max(segments, key=len)
        print(f"Longest segment has {len(longest)} frames")
    """
    # Get clip name -> sequence name mapping
    video_map = aggregate_sequences(split_file, dataset_contact_path, split_name)
    
    # Build clip -> sequence mapping
    clip_to_sequence: Dict[str, str] = {}
    for sequence_name, entry in video_map.items():
        for clip_seq in entry["clip_ids"]:
            clip_to_sequence[clip_seq] = sequence_name
    
    # Load dataset_contact to get objects per clip
    with open(dataset_contact_path, "rb") as f:
        dataset_contact: Dict[str, Dict] = pickle.load(f)
    
    # Group clips by sequence and process each sequence
    all_segments: List[List[Tuple[str, List[str], int]]] = []
    
    for sequence_name, entry in video_map.items():
        clip_metadata = entry["clip_metadata"]
        clips_with_timestamps = []
        
        # Collect all clips for this sequence with their timestamps and objects
        for clip_seq in entry["clip_ids"]:
            clip_meta = clip_metadata.get(clip_seq)
            if clip_meta is None:
                continue
            
            seq_meta = dataset_contact.get(clip_seq)
            if seq_meta is None:
                continue
            
            # Extract timestamps
            timestamps_list = clip_meta.get("per_frame_timestamps_ns", [])
            extracted_timestamps = _extract_timestamps_from_frame_data(timestamps_list)
            
            if len(extracted_timestamps) == 0:
                continue
            
            # Get objects for this clip
            objects = seq_meta.get("objects", [])
            if isinstance(objects, np.ndarray):
                objects = objects.tolist()
            if not isinstance(objects, list):
                objects = []
            
            # Get clip name (use clip_seq which is the formatted clip ID)
            clip_name = clip_seq
            
            # Split clip into parts if timestamps are not monotonically increasing
            clip_parts = _split_clip_by_timestamps(
                clip_name, objects, extracted_timestamps, max_gap_ns
            )
            
            # Add each part as a separate clip entry
            for part in clip_parts:
                clips_with_timestamps.append(part)
        
        if len(clips_with_timestamps) == 0:
            continue
        
        # Sort clips by first timestamp
        clips_with_timestamps.sort(key=lambda x: x["timestamps"][0] if x["timestamps"] else 0)
        
        # Group into continuous segments
        current_segment: List[Tuple[str, List[str], int]] = []
        last_end_timestamp = None
        
        for clip_info in clips_with_timestamps:
            clip_name = clip_info["clip_name"]
            objects = clip_info["objects"]
            timestamps = clip_info["timestamps"]
            frame_start = clip_info.get("frame_start", 0)
            # frame_end is always provided by _split_clip_by_timestamps
            frame_end = clip_info.get("frame_end", len(timestamps) - 1)
            
            if len(timestamps) == 0:
                continue
            
            first_timestamp = timestamps[0]
            last_timestamp = timestamps[-1]
            
            # Check if this clip is continuous with the previous one
            # Must be: monotonically increasing (first_timestamp >= last_end_timestamp)
            # and gap is within threshold
            is_continuous = False
            if last_end_timestamp is not None:
                gap = first_timestamp - last_end_timestamp
                # Allow small gaps, but timestamps must be monotonically increasing (gap >= 0)
                if gap >= 0 and gap <= max_gap_ns:
                    is_continuous = True
            
            if is_continuous and current_segment:
                # Add to current segment (use actual frame indices from the clip)
                for frame_idx in range(frame_start, frame_end + 1):
                    current_segment.append((clip_name, objects, frame_idx))
            else:
                # Start a new segment (timestamps not monotonically increasing or gap too large)
                if current_segment:
                    all_segments.append(current_segment)
                current_segment = []
                # Add frames using actual frame indices from the clip
                for frame_idx in range(frame_start, frame_end + 1):
                    current_segment.append((clip_name, objects, frame_idx))
            
            last_end_timestamp = last_timestamp
        
        # Add the last segment
        if current_segment:
            all_segments.append(current_segment)
    
    return all_segments

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


def _calculate_object_motion(wTo: np.ndarray, motion_threshold: float = 0.05) -> float:
    """
    Calculate total motion for an object trajectory using wTo matrices.
    Based on _is_motion_window from hand2obj_w_geom_motion.py.
    
    Args:
        wTo: Array of shape [T, 4, 4] - object transformation matrices
        motion_threshold: Motion threshold (not used in calculation, kept for compatibility)
    
    Returns:
        Total motion (scalar)
    """
    if wTo.ndim != 3 or wTo.shape[1:] != (4, 4) or len(wTo) < 2:
        return 0.0
    
    # Extract position and rotation from transformation matrices
    positions = wTo[:, :3, 3]  # [T, 3]
    rotations = wTo[:, :3, :3]  # [T, 3, 3]
    
    # Define object keypoints: origin (0,0,0) and x,y,z axes with 0.01m length
    keypoints_local = np.array(
        [
            [0.0, 0.0, 0.0],  # origin
            [0.01, 0.0, 0.0],  # x-axis
            [0.0, 0.01, 0.0],  # y-axis
            [0.0, 0.0, 0.01],  # z-axis
        ],
        dtype=np.float32,
    )  # [4, 3]
    
    # Transform keypoints to world coordinates for all frames at once
    # keypoints_local: [4, 3], rotations: [T, 3, 3], positions: [T, 3]
    # Use batch matrix multiplication: [T, 3, 3] @ [3, 4] -> [T, 3, 4]
    keypoints_world = np.einsum(
        "tij,jk->tik", rotations, keypoints_local.T
    )  # [T, 3, 4]
    # Transpose to [T, 4, 3] and add translation
    keypoints_world = (
        keypoints_world.transpose(0, 2, 1) + positions[:, np.newaxis, :]
    )  # [T, 4, 3]
    
    # Calculate accumulated motion for all keypoints at once
    # Calculate displacement between consecutive frames for all keypoints
    # keypoints_world: [T, 4, 3] -> diff: [T-1, 4, 3]
    displacements = np.linalg.norm(
        np.diff(keypoints_world, axis=0), axis=2
    )  # [T-1, 4]
    displacements = displacements.mean(axis=-1)  # [T-1]
    
    # Sum displacements across time
    total_motion = np.sum(displacements)  # Scalar
    
    return total_motion


def count_dynamic_objects_in_segment(
    segment: List[Tuple[str, List[str], int]],
    dataset_contact: Dict[str, Dict],
    motion_threshold: float = 0.05,
) -> int:
    """
    Count the number of dynamic objects in a segment.
    
    Args:
        segment: List of tuples (clip_name, objects, frame_num)
        dataset_contact: Dictionary containing wTo data for each clip
        motion_threshold: Motion threshold for considering an object dynamic
    
    Returns:
        Number of dynamic objects
    """
    # Collect all unique objects across the segment
    all_objects = set()
    for _, objects, _ in segment:
        all_objects.update(objects)
    
    # Check each object for motion
    dynamic_objects = set()
    
    for obj_id in all_objects:
        # Collect wTo matrices for this object in temporal order across the segment
        wTo_frames = []
        
        for clip_name, objects, frame_num in segment:
            if obj_id not in objects:
                continue
            
            seq_meta = dataset_contact.get(clip_name)
            if seq_meta is None:
                continue
            
            wto_key = f"obj_{obj_id}_wTo"
            if wto_key not in seq_meta:
                continue
            
            wto = np.asarray(seq_meta[wto_key], dtype=np.float32)
            if wto.ndim != 3 or wto.shape[1:] != (4, 4):
                continue
            
            # Get the wTo matrix for this specific frame
            if frame_num < len(wto):
                wTo_frames.append(wto[frame_num])
        
        if len(wTo_frames) < 2:
            continue
        
        # Stack frames into trajectory array [T, 4, 4]
        full_trajectory = np.stack(wTo_frames, axis=0)
        
        # Calculate motion
        total_motion = _calculate_object_motion(full_trajectory, motion_threshold)
        
        if total_motion >= motion_threshold:
            dynamic_objects.add(obj_id)
    
    return len(dynamic_objects)


def save_segments_to_json(
    segments: List[List[Tuple[str, List[str], int]]],
    output_path: str,
    dataset_contact_path: str,
    min_frames: int = 300,
    min_dynamic_objects: int = 3,
    motion_threshold: float = 0.05,
) -> None:
    """
    Filter segments by minimum frame count and dynamic objects, then save clip lists to JSON.
    
    Args:
        segments: List of segments, each segment is a list of tuples (clip_name, objects, frame_num)
        output_path: Path to save the JSON file
        dataset_contact_path: Path to dataset_contact.pkl for accessing wTo data
        min_frames: Minimum number of frames required to keep a segment (default: 300)
        min_dynamic_objects: Minimum number of dynamic objects required (default: 3)
        motion_threshold: Motion threshold for considering an object dynamic (default: 0.05)
    """
    # Load dataset_contact to get wTo data
    with open(dataset_contact_path, "rb") as f:
        dataset_contact: Dict[str, Dict] = pickle.load(f)
    
    # Filter segments with >= min_frames
    filtered_segments = [seg for seg in segments if len(seg) >= min_frames]
    
    # Filter segments with > min_dynamic_objects (more than min_dynamic_objects)
    print(f"\nFiltering segments by dynamic objects (> {min_dynamic_objects})...")
    dynamic_filtered_segments = []
    for seg_idx, segment in enumerate(filtered_segments, 1):
        num_dynamic = count_dynamic_objects_in_segment(
            segment, dataset_contact, motion_threshold
        )
        if num_dynamic > min_dynamic_objects:
            dynamic_filtered_segments.append(segment)
        if (seg_idx - 1) % 10 == 0 or seg_idx == len(filtered_segments):
            print(f"  Processed {seg_idx}/{len(filtered_segments)} segments...")
    
    filtered_segments = dynamic_filtered_segments
    
    # Extract unique clip names from each segment
    segments_dict = {}
    for seg_idx, segment in enumerate(filtered_segments, 1):
        # Get unique clip names from this segment
        unique_clips = sorted(set(clip_name for clip_name, _, _ in segment))
        segments_dict[f"segment{seg_idx}"] = unique_clips
    
    # Create output directory if it doesn't exist
    output_dir = osp.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments_dict, f, indent=2)
    
    print(f"\nSaved {len(filtered_segments)} segments to {output_path}")
    print(f"  - Total segments before filtering: {len(segments)}")
    print(f"  - After frame filter (>= {min_frames} frames): {len([s for s in segments if len(s) >= min_frames])}")
    print(f"  - After dynamic objects filter (> {min_dynamic_objects} dynamic objects): {len(filtered_segments)}")


def print_segments(segments: List[List[Tuple[str, List[str], int]]]) -> None:
    """
    Print segments sorted from longest to shortest.
    """
    # Sort segments by length (longest first)
    sorted_segments = sorted(segments, key=len, reverse=False)
    
    print(f"\n{'='*80}")
    print(f"Found {len(segments)} continuous segments")
    print(f"{'='*80}\n")
    
    for seg_idx, segment in enumerate(sorted_segments, 1):
        print(f"Segment {seg_idx}: {len(segment)} frames")
        print("  Clips: ", end="")
        
        # Group by clip name to show summary
        clip_summary: Dict[str, Dict] = {}
        for clip_name, objects, frame_num in segment:
            if clip_name not in clip_summary:
                clip_summary[clip_name] = {
                    "objects": set(objects),
                    "frame_range": [frame_num, frame_num],
                }
            else:
                clip_summary[clip_name]["frame_range"][0] = min(
                    clip_summary[clip_name]["frame_range"][0], frame_num
                )
                clip_summary[clip_name]["frame_range"][1] = max(
                    clip_summary[clip_name]["frame_range"][1], frame_num
                )
                clip_summary[clip_name]["objects"].update(objects)
        
        clip_info_strs = []
        for clip_name in sorted(clip_summary.keys()):
            info = clip_summary[clip_name]
            frame_start, frame_end = info["frame_range"]
            objects_str = ",".join(sorted(info["objects"]))
            if frame_start == frame_end:
                frame_str = f"frame {frame_start}"
            else:
                frame_str = f"frames {frame_start}-{frame_end}"
            clip_info_strs.append(f"{clip_name} ({frame_str}, objects: [{objects_str}])")
        
        print(", ".join(clip_info_strs))
        print()


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
    
    # Group clips into continuous segments
    segments = group_clips_into_segments(
        split_file=split_file,
        dataset_contact_path=dataset_contact_path,
        split_name=split_name,
    )
    
    # Print segments from longest to shortest
    print_segments(segments)
    
    # Filter and save segments with >= 300 frames and > 3 dynamic objects to JSON
    segments_output_path = "data/HOT3D-CLIP/sets/segments.json"
    save_segments_to_json(
        segments,
        segments_output_path,
        dataset_contact_path=dataset_contact_path,
        min_frames=300,
        min_dynamic_objects=3,
        motion_threshold=0.05,
    )
    
    # create_teaser_videos(ranked, images_root, output_dir, top, fps)


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
