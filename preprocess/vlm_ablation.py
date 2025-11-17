
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import os.path as osp
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from preprocess.vlm_query_hot3d import _ensure_sequence_assets, _load_dataset_cache
from preprocess.eval_vlm import (
    DEFAULT_VLM_MODEL,
    GPT5_MODEL,
    GPT5_PRO_MODEL,
    SUPPORTED_VLM_MODELS,
    call_vlm_for_contact,
    load_few_shot_examples,
    save_dir,
    train_index_list,
    models_info_path,
)
_MODELS_INFO_CACHE: Optional[Dict[str, object]] = None


def load_models_info() -> Dict[str, object]:
    global _MODELS_INFO_CACHE
    if _MODELS_INFO_CACHE is None:
        with open(models_info_path, "r") as f:
            _MODELS_INFO_CACHE = json.load(f)
    return _MODELS_INFO_CACHE


PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
SPLIT_PATH = osp.join(PROJECT_ROOT, "data", "HOT3D-CLIP", "sets", "split.json")
WORKSPACE_DIR = osp.join(PROJECT_ROOT, "outputs", "vlm_ablation")
DEFAULT_GT_DIR = osp.join(WORKSPACE_DIR, "gt")
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR


def resolve_gt_dir(gt_dir_arg: Optional[str]) -> str:
    if gt_dir_arg:
        return osp.abspath(gt_dir_arg)
    return DEFAULT_GT_DIR


@dataclass
class SampleRecord:
    sample_id: str
    seq: str
    frame_idx: int
    image_path: str
    overlay_path: str
    mask_path: str
    object_names: List[str]
    contact: List[List[int]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "seq": self.seq,
            "frame_idx": self.frame_idx,
            "image_path": self.image_path,
            "overlay_path": self.overlay_path,
            "mask_path": self.mask_path,
            "object_names": self.object_names,
            "contact": self.contact,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SampleRecord":
        return cls(
            sample_id=str(data["sample_id"]),
            seq=str(data["seq"]),
            frame_idx=int(data["frame_idx"]),
            image_path=str(data["image_path"]),
            overlay_path=str(data["overlay_path"]),
            mask_path=str(data["mask_path"]),
            object_names=[str(name) for name in data["object_names"]],
            contact=[[int(x) for x in pair] for pair in data["contact"]],
        )

    @property
    def num_objects(self) -> int:
        return len(self.object_names)

    def contact_array(self) -> np.ndarray:
        return np.asarray(self.contact, dtype=np.int32)


_OPENAI_CLIENT: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


def load_split_ids(split_name: str) -> List[str]:
    with open(SPLIT_PATH, "r") as f:
        split_dict = json.load(f)
    split_list = split_dict[split_name]
    return [str(seq) for seq in split_list]


def safe_div(num: int, denom: int) -> Optional[float]:
    if denom:
        return float(num) / float(denom)
    return None


def update_counts(counts: Dict[str, int], gt_label: int, pred_label: int) -> None:
    if gt_label not in (0, 1) or pred_label not in (0, 1):
        return
    if gt_label == 1:
        if pred_label == 1:
            counts["tp"] += 1
        else:
            counts["fn"] += 1
    else:
        if pred_label == 1:
            counts["fp"] += 1
        else:
            counts["tn"] += 1


def summarize_counts(counts: Dict[str, int]) -> Dict[str, object]:
    total = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    positives = counts["tp"] + counts["fn"]
    negatives = counts["tn"] + counts["fp"]

    precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
    specificity = safe_div(counts["tn"], counts["tn"] + counts["fp"])
    accuracy = safe_div(counts["tp"] + counts["tn"], total)
    if precision is not None and recall is not None and (precision + recall):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    if recall is not None and specificity is not None:
        balanced_accuracy = 0.5 * (recall + specificity)
    elif recall is not None:
        balanced_accuracy = recall
    else:
        balanced_accuracy = specificity

    return {
        "counts": dict(counts),
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
        },
        "support": {
            "total_pairs": total,
            "positive_pairs": positives,
            "negative_pairs": negatives,
        },
    }


def compute_frame_counts(gt_contact: np.ndarray, pred_contact: np.ndarray) -> Dict[str, int]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    num_objects = gt_contact.shape[0]
    for obj_idx in range(num_objects):
        for hand_idx in range(2):
            gt_label = int(gt_contact[obj_idx, hand_idx])
            pred_label = int(pred_contact[obj_idx, hand_idx])
            update_counts(counts, gt_label, pred_label)
    return counts


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def image_file_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def load_first_json_object(payload: str) -> object:
    decoder = json.JSONDecoder()
    stripped = payload.lstrip()
    obj, end_idx = decoder.raw_decode(stripped)
    remainder = stripped[end_idx:].strip()
    if remainder:
        # Keep only the first object; additional duplicates may be appended by the API.
        return obj
    return obj


def invoke_openai_json(
    model: str,
    temperature: float,
    responses_messages: List[Dict[str, object]],
    chat_messages: List[Dict[str, object]],
    schema_name: str,
    schema_body: Dict[str, object],
    reasoning_effort: str = "medium",
) -> str:
    client = get_openai_client()
    if model in {GPT5_MODEL, GPT5_PRO_MODEL}:
        json_payload = {
            "type": "json_schema",
            "name": schema_name,
            "schema": schema_body,
            "strict": True,
        }
        max_output_tokens = 2048
        effort = reasoning_effort if model == GPT5_PRO_MODEL else reasoning_effort
        content = ""
        for _ in range(3):
            response = client.responses.create(
                model=model,
                reasoning={"effort": effort},
                input=responses_messages,
                text={"format": json_payload},
                max_output_tokens=max_output_tokens,
            )
            content_parts: List[str] = []
            if hasattr(response, "output"):
                for output in response.output:
                    for item in getattr(output, "content", []) or []:
                        if getattr(item, "type", "") in {"output_text", "text"}:
                            content_parts.append(getattr(item, "text", ""))
            if hasattr(response, "output_text") and response.output_text:
                content_parts.append(response.output_text)
            content = "".join(content_parts).strip()
            if content:
                break
            incomplete_details = getattr(response, "incomplete_details", None)
            if (
                getattr(response, "status", "") == "incomplete"
                and incomplete_details is not None
                and getattr(incomplete_details, "reason", "") == "max_output_tokens"
            ):
                max_output_tokens = min(max_output_tokens * 2, 1024)
                effort = "low"
                continue
            if effort != "minimal":
                effort = "minimal"
                continue
        if not content:
            raise RuntimeError(f"Empty response content for model {model}")
        return content
    json_schema_meta = {"name": schema_name, "schema": schema_body, "strict": True}
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=chat_messages,
        response_format={"type": "json_schema", "json_schema": json_schema_meta},
    )
    return response.choices[0].message.content


def adjust_temperature(model: str, temperature: float) -> float:
    if model in {GPT5_MODEL, GPT5_PRO_MODEL} and math.isclose(temperature, 0.0):
        return 0.1
    return temperature


def build_single_object_prompts(object_name: str) -> Tuple[str, str]:
    system_instruction = """You are a precise visual classifier for hand-object contact detection in cluttered scenes.

CRITICAL CONSTRAINTS:
1. "In contact" means direct physical touch: grasping, holding, pressing, or any visible contact.
2. If a hand is not clearly touching the object, you must output 0 for that hand.
"""
    user_prompt = f"""Determine whether the hands are in contact with the object named "{object_name}" in this frame.

STRICT DEFINITION OF CONTACT:
- Contact (label = 1) requires visible physical touch between the hand and the object in this single frame.
- Reaching, hovering, occluded, or ambiguous contact must be labeled 0.

OUTPUT FORMAT:
Return only JSON in this exact form (no extra text):
{{
  "left": 0,
  "right": 0
}}
"""
    return system_instruction, user_prompt


def build_object_name_prompt(object_names: Sequence[str]) -> Tuple[str, str]:
    system_instruction = """You are a precise visual classifier for hand-object contact detection in cluttered scenes.

CRITICAL CONSTRAINTS:
1. Each hand (left/right) can be in contact with AT MOST ONE object at a time.
2. "In contact" means direct physical touch: grasping, holding, pressing, or any visible contact.
3. If a hand is not clearly touching any object, you must output 0 for that hand.
"""
    object_lines = "\n".join([f"- {name}" for name in object_names])
    output_lines = []
    for idx, name in enumerate(object_names):
        suffix = "," if idx < len(object_names) - 1 else ""
        output_lines.append(f'  "{name}": {{"left": 0, "right": 0}}{suffix}')
    output_example = "\n".join(output_lines)
    user_prompt = f"""Analyze this image for hand–object contact (actual touching, not just reaching).

CANDIDATE OBJECTS (use exact names):
{object_lines}

STRICT DEFINITION OF CONTACT:
- Contact (label = 1) requires mask-level overlap or clear visual touch between the hand and the object in this single frame.
- Reaching, hovering, depth-aligned without touching, or ambiguous overlaps must be labeled 0.

CONSTRAINTS:
- Each hand can touch at most one object.
- If a hand is not clearly touching any object, mark all objects as 0 for that hand.

OUTPUT FORMAT:
Return only a JSON object in this exact structure (no extra text):
{{
{output_example}
}}

You must include every object from the list above as keys in the JSON.
"""
    return system_instruction, user_prompt


def predictions_to_array(sample: SampleRecord, predictions: Sequence[Dict[str, object]]) -> np.ndarray:
    arr = np.full((sample.num_objects, 2), -1, dtype=np.int32)
    name_to_index = {name: idx for idx, name in enumerate(sample.object_names)}
    for pred in predictions:
        object_name = str(pred["object_name"])
        idx = name_to_index[object_name]
        arr[idx, 0] = int(pred["left_hand_contact"])
        arr[idx, 1] = int(pred["right_hand_contact"])
    if (arr < 0).any():
        raise RuntimeError(f"Incomplete predictions for sample {sample.sample_id}")
    return arr


def prepare_test_set(gt_dir: str, num_samples: int, seed: int, force: bool) -> List[SampleRecord]:
    gt_root = resolve_gt_dir(gt_dir)
    manifest_path = osp.join(gt_root, "manifest.json")
    if osp.exists(manifest_path) and not force:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return [SampleRecord.from_dict(entry) for entry in manifest["samples"]]

    dataset = _load_dataset_cache()
    split_sequences = load_split_ids("test50")
    os.makedirs(gt_root, exist_ok=True)
    rng = random.Random(seed)

    candidate_frames: List[Tuple[str, int]] = []

    for seq in tqdm(split_sequences, desc="Enumerating frames"):
        _ensure_sequence_assets(seq, dataset)
        seq_dir = osp.abspath(osp.join(save_dir, seq))
        annotation_path = osp.join(seq_dir, "annotation.npz")
        contact_path = osp.join(seq_dir, "gt_contact.npz")

        with np.load(annotation_path, allow_pickle=True) as annotation_data:
            names_arr = annotation_data["name"]
            total_frames = int(annotation_data["mask"].shape[0])
            object_ids = annotation_data["object_id"]
        models_info = load_models_info()
        object_names = []
        for obj_id in object_ids:
            uid = str(obj_id)
            info = models_info.get(uid)
            if info is not None and "name" in info:
                object_names.append(str(info["name"]))
            else:
                object_names.append(uid)
        if not object_names:
            continue
        with np.load(contact_path) as contact_data:
            contact_frames = int(contact_data["contact_label"].shape[0])
        if total_frames != contact_frames:
            raise RuntimeError(f"Frame mismatch for sequence {seq}: annotation={total_frames}, contact={contact_frames}")
        for frame_idx in range(total_frames):
            candidate_frames.append((seq, frame_idx))

    if num_samples > len(candidate_frames):
        raise ValueError(f"Requested {num_samples} samples but only {len(candidate_frames)} frames with objects are available.")

    selected = rng.sample(candidate_frames, num_samples)
    sample_records: List[SampleRecord] = []

    for seq, frame_idx in tqdm(selected, desc="Preparing GT samples"):
        seq_dir = osp.abspath(osp.join(save_dir, seq))
        annotation_path = osp.join(seq_dir, "annotation.npz")
        contact_path = osp.join(seq_dir, "gt_contact.npz")

        with np.load(annotation_path, allow_pickle=True) as annotation_data:
            names_arr = annotation_data["name"]
            mask_frame = np.asarray(annotation_data["mask"][frame_idx])
            object_ids = annotation_data["object_id"]
        models_info = load_models_info()
        object_names = []
        for obj_id in object_ids:
            uid = str(obj_id)
            info = models_info.get(uid)
            if info is not None and "name" in info:
                object_names.append(str(info["name"]))
            else:
                object_names.append(uid)

        if not object_names:
            raise RuntimeError(f"No objects found for selected frame {seq}:{frame_idx:04d}")

        with np.load(contact_path) as contact_data:
            contact_frame = np.asarray(contact_data["contact_label"][frame_idx])

        sample_id = f"{seq}_{frame_idx:04d}"
        sample_dir = osp.join(gt_root, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        image_src = osp.join(seq_dir, "images", f"{frame_idx:04d}.jpg")
        overlay_src = osp.join(seq_dir, "overlay", f"{frame_idx:04d}.jpg")

        image_dst = osp.join(sample_dir, "image.jpg")
        overlay_dst = osp.join(sample_dir, "overlay.jpg")
        shutil.copy2(image_src, image_dst)
        shutil.copy2(overlay_src, overlay_dst)

        mask_path = osp.join(sample_dir, "mask.npz")
        np.savez_compressed(mask_path, mask=mask_frame, name=names_arr)

        contact_list = contact_frame.tolist()

        sample_records.append(
            SampleRecord(
                sample_id=sample_id,
                seq=seq,
                frame_idx=frame_idx,
                image_path=osp.abspath(image_dst),
                overlay_path=osp.abspath(overlay_dst),
                mask_path=osp.abspath(mask_path),
                object_names=object_names,
                contact=contact_list,
            )
        )

    manifest = {
        "seed": seed,
        "requested_samples": num_samples,
        "num_samples": len(sample_records),
        "samples": [record.to_dict() for record in sample_records],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return sample_records


def predict_vanilla(sample: SampleRecord, model: str, temperature: float) -> List[Dict[str, object]]:
    temperature_use = adjust_temperature(model, temperature)
    predictions: List[Dict[str, object]] = []
    for object_name in sample.object_names:
        system_instruction, user_prompt = build_single_object_prompts(object_name)
        image_url = image_file_to_data_url(sample.image_path)
        responses_messages = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_instruction}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            },
        ]
        chat_messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "left": {"type": "integer", "enum": [0, 1]},
                "right": {"type": "integer", "enum": [0, 1]},
            },
            "required": ["left", "right"],
        }
        content = invoke_openai_json(
            model=model,
            temperature=temperature_use,
            responses_messages=responses_messages,
            chat_messages=chat_messages,
            schema_name="SingleObjectContact",
            schema_body=schema,
            reasoning_effort="medium",
        )
        parsed = load_first_json_object(content)
        predictions.append(
            {
                "object_name": object_name,
                "left_hand_contact": int(parsed["left"]),
                "right_hand_contact": int(parsed["right"]),
            }
        )
    return predictions


def predict_one_of_k_by_name(sample: SampleRecord, model: str, temperature: float, *, use_overlay: bool) -> List[Dict[str, object]]:
    system_instruction, user_prompt = build_object_name_prompt(sample.object_names)
    image_path = sample.overlay_path if use_overlay else sample.image_path
    image_url = image_file_to_data_url(image_path)
    responses_messages = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_instruction}],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "image_url": image_url},
            ],
        },
    ]
    chat_messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    properties = {
        name: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "left": {"type": "integer", "enum": [0, 1]},
                "right": {"type": "integer", "enum": [0, 1]},
            },
            "required": ["left", "right"],
        }
        for name in sample.object_names
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties.keys()),
    }
    temperature_use = adjust_temperature(model, temperature)
    content = invoke_openai_json(
        model=model,
        temperature=temperature_use,
        responses_messages=responses_messages,
        chat_messages=chat_messages,
        schema_name="MultiObjectContactByName",
        schema_body=schema,
        reasoning_effort="medium",
    )
    parsed = load_first_json_object(content)
    predictions: List[Dict[str, object]] = []
    for name in sample.object_names:
        obj_data = parsed[name]
        predictions.append(
            {
                "object_name": name,
                "left_hand_contact": int(obj_data["left"]),
                "right_hand_contact": int(obj_data["right"]),
            }
        )
    return predictions


def predict_one_of_k_overlay(sample: SampleRecord, model: str, temperature: float) -> List[Dict[str, object]]:
    return predict_one_of_k_by_name(sample, model, temperature, use_overlay=True)


def predict_full_model(
    sample: SampleRecord,
    model: str,
    temperature: float,
    few_shot_examples: Optional[List[Dict[str, object]]],
) -> List[Dict[str, object]]:
    temperature_use = adjust_temperature(model, temperature)
    return call_vlm_for_contact(
        image_path=sample.image_path,
        overlay_image_path=sample.overlay_path,
        object_names=sample.object_names,
        model=model,
        temperature=temperature_use,
        few_shot_examples=few_shot_examples,
    )


def run_variant(
    variant_name: str,
    samples: Sequence[SampleRecord],
    output_dir: str,
    predictor,
) -> Dict[str, object]:
    variant_dir = osp.join(output_dir, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    rows: List[Dict[str, object]] = []
    total_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for sample in tqdm(samples, desc=f"Running {variant_name}"):
        predictions = predictor(sample)
        pred_array = predictions_to_array(sample, predictions)
        gt_array = sample.contact_array()
        frame_counts = compute_frame_counts(gt_array, pred_array)
        frame_summary = summarize_counts(frame_counts)
        for key in total_counts:
            total_counts[key] += frame_counts[key]

        row = {
            "seq": sample.seq,
            "frame_idx": sample.frame_idx,
            "num_objects": sample.num_objects,
            "tp": frame_counts["tp"],
            "tn": frame_counts["tn"],
            "fp": frame_counts["fp"],
            "fn": frame_counts["fn"],
            "precision": frame_summary["metrics"]["precision"],
            "recall": frame_summary["metrics"]["recall"],
            "f1": frame_summary["metrics"]["f1"],
            "balanced_accuracy": frame_summary["metrics"]["balanced_accuracy"],
            "total_pairs": frame_summary["support"]["total_pairs"],
            "positive_pairs": frame_summary["support"]["positive_pairs"],
            "negative_pairs": frame_summary["support"]["negative_pairs"],
        }
        rows.append(row)

        prediction_record = {
            "sample_id": sample.sample_id,
            "seq": sample.seq,
            "frame_idx": sample.frame_idx,
            "predictions": predictions,
            "ground_truth": sample.contact,
            "object_names": sample.object_names,
            "image_path": sample.image_path,
            "overlay_path": sample.overlay_path,
        }
        prediction_path = osp.join(variant_dir, f"{sample.sample_id}.json")
        with open(prediction_path, "w") as f:
            json.dump(prediction_record, f, indent=2)

    mean_summary = summarize_counts(total_counts)
    csv_path = osp.join(variant_dir, f"{variant_name}_metrics.csv")
    write_metrics_csv(csv_path, rows, mean_summary)
    metrics = mean_summary["metrics"]
    print(
        f"[{variant_name}] "
        f"F1={format_metric(metrics['f1'])} "
        f"Precision={format_metric(metrics['precision'])} "
        f"Recall={format_metric(metrics['recall'])} "
        f"BalancedAcc={format_metric(metrics['balanced_accuracy'])}"
    )
    return {"csv_path": csv_path, "summary": mean_summary}


def write_metrics_csv(
    csv_path: str,
    rows: Sequence[Dict[str, object]],
    mean_summary: Dict[str, object],
) -> None:
    fieldnames = [
        "seq",
        "frame_idx",
        "num_objects",
        "tp",
        "tn",
        "fp",
        "fn",
        "total_pairs",
        "positive_pairs",
        "negative_pairs",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]

    mean_counts = mean_summary["counts"]
    mean_metrics = mean_summary["metrics"]
    mean_support = mean_summary["support"]

    mean_row = {
        "seq": "MEAN",
        "frame_idx": "",
        "num_objects": "",
        "tp": mean_counts["tp"],
        "tn": mean_counts["tn"],
        "fp": mean_counts["fp"],
        "fn": mean_counts["fn"],
        "total_pairs": mean_support["total_pairs"],
        "positive_pairs": mean_support["positive_pairs"],
        "negative_pairs": mean_support["negative_pairs"],
        "precision": format_metric(mean_metrics["precision"]),
        "recall": format_metric(mean_metrics["recall"]),
        "f1": format_metric(mean_metrics["f1"]),
        "balanced_accuracy": format_metric(mean_metrics["balanced_accuracy"]),
    }

    formatted_rows: List[Dict[str, object]] = []
    for row in rows:
        formatted_rows.append(
            {
                "seq": row["seq"],
                "frame_idx": row["frame_idx"],
                "num_objects": row["num_objects"],
                "tp": row["tp"],
                "tn": row["tn"],
                "fp": row["fp"],
                "fn": row["fn"],
                "total_pairs": row["total_pairs"],
                "positive_pairs": row["positive_pairs"],
                "negative_pairs": row["negative_pairs"],
                "precision": format_metric(row["precision"]),
                "recall": format_metric(row["recall"]),
                "f1": format_metric(row["f1"]),
                "balanced_accuracy": format_metric(row["balanced_accuracy"]),
            }
        )

    rows_to_write = [mean_row] + formatted_rows
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_to_write:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablations for VLM contact prediction.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of random frames to sample for the test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling frames.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["vanilla", "one_of_k", "one_of_k_visual", "full"],
        choices=["vanilla", "one_of_k", "one_of_k_visual", "full"],
        help="Which ablation variants to run.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_VLM_MODEL,
        choices=sorted(SUPPORTED_VLM_MODELS),
        help="VLM model to query.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for model calls.",
    )
    parser.add_argument(
        "--gt-dir",
        default=DEFAULT_GT_DIR,
        help="Directory under workspace to store GT samples.",
    )
    parser.add_argument(
        "--force-gt",
        action="store_true",
        help="Regenerate GT samples even if manifest exists.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save ablation outputs.",
    )
    parser.add_argument(
        "--example-dir",
        default=None,
        help="Directory containing ICL examples for full model (defaults to save_dir+'_trainset/examples').",
    )
    parser.add_argument(
        "--train-indices",
        nargs="*",
        default=None,
        help="Override train_index_list with custom seq_frame indices for ICL.",
    )
    return parser.parse_args()


def resolve_few_shot_examples(
    example_dir_arg: Optional[str],
    train_indices_arg: Optional[List[str]],
) -> Tuple[str, Optional[List[Dict[str, object]]]]:
    if train_indices_arg is not None:
        train_indices_resolved = train_indices_arg
    else:
        train_indices_resolved = train_index_list
    if example_dir_arg is not None:
        example_dir_resolved = osp.abspath(example_dir_arg)
    else:
        example_dir_resolved = osp.join(osp.abspath(save_dir + "_trainset"), "examples")
    few_shot_examples = load_few_shot_examples(example_dir_resolved, train_indices_resolved)
    return example_dir_resolved, few_shot_examples


def main() -> None:
    args = parse_args()
    gt_dir = args.gt_dir
    output_dir = osp.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    samples = prepare_test_set(gt_dir, args.num_samples, args.seed, args.force_gt)

    # _, few_shot_examples = resolve_few_shot_examples(args.example_dir, args.train_indices)

    # for variant_name in args.variants:
    #     if variant_name == "vanilla":
    #         run_variant(
    #             variant_name,
    #             samples,
    #             output_dir,
    #             predictor=lambda sample: predict_vanilla(sample, args.model, args.temperature),
    #         )
    #     elif variant_name == "one_of_k":
    #         run_variant(
    #             variant_name,
    #             samples,
    #             output_dir,
    #             predictor=lambda sample: predict_one_of_k_by_name(sample, args.model, args.temperature, use_overlay=False),
    #         )
    #     elif variant_name == "one_of_k_visual":
    #         run_variant(
    #             variant_name,
    #             samples,
    #             output_dir,
    #             predictor=lambda sample: predict_one_of_k_overlay(sample, args.model, args.temperature),
    #         )
    #     elif variant_name == "full":
    #         run_variant(
    #             variant_name,
    #             samples,
    #             output_dir,
    #             predictor=lambda sample: predict_full_model(sample, args.model, args.temperature, few_shot_examples),
    #         )


if __name__ == "__main__":
    main()

