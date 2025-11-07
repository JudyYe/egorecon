import os
import sys
import re
import json
import base64
import argparse
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from PIL import Image


# -----------------------
# Filename filter
# -----------------------
# Strict pattern: exactly 6 digits, ".image_", one-or-more digits, "-", one digit, ".jpg"
# Example: 000000.image_214-1.jpg
# NAME_PATTERN = re.compile(r'^\d{6}\.image_\d+-\d\.jpg$', re.IGNORECASE)
# NAME_PATTERN = re.compile(r'^\d{6}\.image_214-1.jpg$', re.IGNORECASE)

# 0000.jpg  0010.jpg  0020.jpg  0030.jpg  0040.jpg  0050.jpg  0060.jpg  0070.jpg  0080.jpg  0090.jpg  0100.jpg  0110.jpg  0120.jpg  0130.jpg  0140.jpg
# 0001.jpg  0011.jpg  0021.jpg  0031.jpg  0041.jpg  0051.jpg  0061.jpg  0071.jpg  0081.jpg  0091.jpg  0101.jpg  0111.jpg  0121.jpg  0131.jpg  0141.jpg
NAME_PATTERN = re.compile(r'^\d{4}\.jpg$', re.IGNORECASE)

def matches_required_name(p: Path) -> bool:
    """Return True iff filename matches image pattern."""
    return bool(NAME_PATTERN.match(p.name))

MODEL_PRICES = {
    "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "cached": 0.50, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.60},
}

def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _obj_to_dict(obj):
    """Return a plain dict from a pydantic model or dict-like, else {}."""
    if obj is None:
        return {}
    # pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    return {}

def compute_cost_from_usage(model_name: str, usage) -> dict:
    prices = MODEL_PRICES[model_name]  # let this KeyError loudly if misconfigured

    # Handle both old/new field names
    input_tokens  = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", 0)
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", 0)

    # Details may live under input_* or prompt_* (SDK/version dependent)
    details = getattr(usage, "input_tokens_details", None)
    if details is None:
        details = getattr(usage, "prompt_tokens_details", None)

    d = _obj_to_dict(details)
    cached_tokens = _to_int(d.get("cached_tokens", 0), 0)

    input_tokens  = _to_int(input_tokens, 0)
    output_tokens = _to_int(output_tokens, 0)
    noncached_input = max(input_tokens - cached_tokens, 0)

    per_tok_input  = prices["input"]  / 1_000_000.0
    per_tok_cached = prices["cached"] / 1_000_000.0
    per_tok_output = prices["output"] / 1_000_000.0

    input_usd  = noncached_input * per_tok_input
    cached_usd = cached_tokens   * per_tok_cached
    output_usd = output_tokens   * per_tok_output
    total_usd  = input_usd + cached_usd + output_usd

    # Try to get total_tokens if present (old/new)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = _to_int(input_tokens + output_tokens, 0)

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_usd": round(input_usd, 6),
        "cached_input_usd": round(cached_usd, 6),
        "output_usd": round(output_usd, 6),
        "total_usd": round(total_usd, 6),
    }


# -----------------------
# Helpers
# -----------------------

def image_file_to_data_url(path: Path) -> str:
    """
    Read a local image file and return a data URL for vision models.
    This script only processes .jpg files per the filename pattern,
    but we still compute MIME for robustness.
    """
    suf = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".jfif": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }.get(suf, "image/jpeg")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# -----------------------
# Prompting + schema
# -----------------------

SYSTEM_INSTRUCTION = (
    "You are a precise visual classifier. "
    "For each specified object name, decide if the LEFT and RIGHT human hands are physically touching that object in the image. "
    "Rules: (1) 'In contact' = visible touch, including grasp/press/hold. "
    "Output MUST strictly match the JSON schema."
)

def build_json_schema() -> Dict[str, Any]:
    """
    Strict JSON schema:
    {
      "predictions": [
        {"object_name": str, "left_hand_contact": 0|1, "right_hand_contact": 0|1},
        ...
      ]
    }
    """
    return {
        "name": "HandObjectContactBatch",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "object_name": {"type": "string"},
                            "left_hand_contact": {"type": "integer", "enum": [0, 1]},
                            "right_hand_contact": {"type": "integer", "enum": [0, 1]},
                        },
                        "required": [
                            "object_name",
                            "left_hand_contact",
                            "right_hand_contact",
                        ],
                    },
                }
            },
            "required": ["predictions"],
        },
    }

def classify_image(
    client: OpenAI,
    model: str,
    image_path: Path,
    user_prompt: str,
    object_names: List[str],
    temperature: float = 0.0,
) -> List[Dict[str, int]]:
    """
    Returns a list of:
      [{"object_name": str, "left_hand_contact": 0|1, "right_hand_contact": 0|1}, ...]
    aligned to the provided object_names.
    """
    tmp_path = image_path.with_suffix(".resized.jpg")
    Image.open(image_path).resize((512,512)).save(tmp_path, quality=90)

    full_image_url = image_file_to_data_url(tmp_path)

    user_content = [
        {"type": "text", "text": user_prompt.strip()},
        {"type": "image_url", "image_url": {"url": full_image_url}},
        {"type": "text", "text": "Objects to evaluate (in this order):"},
        {"type": "text", "text": json.dumps(object_names, ensure_ascii=False)},
    ]

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_schema", "json_schema": build_json_schema()},
    )

    usage = resp.usage  # has prompt_tokens, completion_tokens, total_tokens

    cost = compute_cost_from_usage(model_name=model, usage=usage)
    print(f"Tokens: prompt={cost['input_tokens']}, \
        completion={cost['output_tokens']},\
        total={cost['total_tokens']}  |  Cost: ${cost['total_usd']}")

    content = resp.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to salvage the first JSON object if model wrapped it
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(content[start : end + 1])
        else:
            raise RuntimeError(f"Invalid JSON from model for {image_path.name}: {content!r}")

    preds = parsed.get("predictions", [])
    pred_by_name = {p.get("object_name"): p for p in preds if isinstance(p, dict)}

    out = []
    for name in object_names:
        p = pred_by_name.get(name)
        if not p:
            raise RuntimeError(f"Missing prediction for object_name='{name}' in {image_path.name}. Got: {preds}")
        L, R = p.get("left_hand_contact"), p.get("right_hand_contact")
        if L not in (0, 1) or R not in (0, 1):
            raise RuntimeError(f"Bad labels for '{name}' in {image_path.name}: {p}")
        out.append({"object_name": name, "left_hand_contact": int(L), "right_hand_contact": int(R)})
    return out

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Hand-object contact per object (same list for all images).")
    ap.add_argument("--images_dir", type=str,required=True, help="Folder with images.")
    ap.add_argument("--objects", required=True, help="Comma-separated object names for all images (e.g., 'mug,phone,laptop').")
    ap.add_argument("--prompt", required=True, help="Short task/context prompt shown with each image.")
    ap.add_argument("--model", default="gpt-4o", help="Vision-capable chat model.")
    ap.add_argument("--output", default="contact_by_object.jsonl", help="Output JSONL path.")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Not a directory: {images_dir}", file=sys.stderr)
        sys.exit(1)

    object_names = [s.strip() for s in args.objects.split(",") if s.strip()]
    if not object_names:
        print("Provide at least one object via --objects", file=sys.stderr)
        sys.exit(1)

    # Collect ONLY files that match the strict name pattern
    paths = sorted(p for p in images_dir.iterdir() if p.is_file() and matches_required_name(p))[::10]
    if not paths:
        print(f"No matching files found in {images_dir} (expected like 000000.image_214-1.jpg)", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()  # uses OPENAI_API_KEY from env

    out_path = Path(args.output)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for img in paths:

            try:
                preds = classify_image(
                    client=client,
                    model=args.model,
                    image_path=img,
                    user_prompt=args.prompt,
                    object_names=object_names,
                    temperature=args.temperature,
                )
                rec = {"image": str(img), "predictions": preds}

                f.write(json.dumps(rec) + "\n")
                written += 1
                short = ", ".join([f"{p['object_name']}:L{p['left_hand_contact']} R{p['right_hand_contact']}" for p in preds])
                print(f"[OK] {img.name} -> {short}")
                
            except Exception as e:
                print(f"[FAIL] {img.name}: {e}")

    print(f"\nWrote {written} rows to {out_path.resolve()}")

if __name__ == "__main__":
    main()
