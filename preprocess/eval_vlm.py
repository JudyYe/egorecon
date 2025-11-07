# overlayed mask
# image: 
# mask: (T, 2+O, H, W)
# contact label: (T, O, 2)
# name: 2+O
from tqdm import tqdm

from egorecon.manip.data.utils import load_pickle
import os
import os.path as osp
import json
import shutil
import numpy as np
from glob import glob
import math

data_dir = "data/HOT3D-CLIP/"
models_info_path = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/object_models_eval/models_info.json"
gt_mask_dir = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/gt_mask"
save_dir = "outputs/debug_vlm"
vlm_output_dir = "outputs/debug_vlm/vlm_output"
split = "test50"

# VLM model configuration
DEFAULT_VLM_MODEL = "gpt-4o"
GPT5_MODEL = "gpt-5"
GPT5_PRO_MODEL = "gpt-5-pro"
SUPPORTED_VLM_MODELS = {DEFAULT_VLM_MODEL, GPT5_MODEL, GPT5_PRO_MODEL}


def create_one_folder(seq, t_index, save_dir, all_contact_data, all_data):
    """
    Create a folder structure for a sequence with images, masks, and contact labels.
    
    Args:
        seq: Sequence ID (e.g., "001870")
        t_index: Time index (not used currently, but kept for compatibility)
        save_dir: Directory to save the output folder
        all_contact_data: Dictionary containing contact data from dataset_contact.pkl
        all_data: Dictionary containing sequence data from dataset_contact.pkl
    
    Creates:
        seq/
            images/
                *.jpg
            annotation.npz
                'mask': (T, 2+O, H, W)
                'name': (2+O,) array of names
                'object_id': array of object IDs
                'object_name': array of object names (aligned with object_id)
            gt_contact.npz
                'contact_label': (T, O, 2) - left and right hand contact per object
    """
    # Create output directory structure
    seq_dir = osp.join(save_dir, seq)
    images_dir = osp.join(seq_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Load models_info.json to map UIDs to object names
    with open(models_info_path, "r") as f:
        models_info = json.load(f)
    uid_to_name = {uid: models_info[uid]["name"] for uid in models_info.keys()}
    
    # Get sequence data
    seq_data = all_data[seq]
    object_ids = list(seq_data["objects"].keys())
    T = len(seq_data["wTc"])
    
    # Copy images from extract_images-rot90/clip-{seq}/ to seq/images/
    source_image_dir = osp.join(data_dir, "extract_images-rot90", f"clip-{seq}")
    image_list = sorted(glob(osp.join(source_image_dir, "*.jpg")))
    assert len(image_list) == T, f"Image count mismatch: {len(image_list)} != {T}"
    
    for i, img_path in enumerate(image_list):
        # Copy with 4-digit zero-padded name
        dst_path = osp.join(images_dir, f"{i:04d}.jpg")
        shutil.copy2(img_path, dst_path)
    
    # Load mask from existing file
    # Mask is saved in /move/u/yufeiy2/egorecon/data/HOT3D-CLIP/gt_mask/{seq}.npz
    mask_file = osp.join(gt_mask_dir, f"{seq}.npz")
    if not osp.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    
    # Load mask data
    mask_data = np.load(mask_file)
    hand_obj_mask = mask_data["hand_obj_mask"]  # (2+O, T, H, W)
    mask_index = mask_data["index"]  # (2+O,) array with ["left_hand", "right_hand", uid1, uid2, ...]
    
    # Transpose to (T, 2+O, H, W)
    mask = hand_obj_mask.transpose(1, 0, 2, 3)  # (T, 2+O, H, W)
    
    # Map UIDs to names
    names = []
    for idx in mask_index:
        if idx == "left_hand":
            names.append("left_hand")
        elif idx == "right_hand":
            names.append("right_hand")
        else:
            # idx is a UID, map to name
            uid = str(idx)
            if uid in uid_to_name:
                names.append(uid_to_name[uid])
            else:
                # Fallback to UID if not found
                names.append(uid)
    
    names = np.array(names)
    
    # Create object_name mapping: object_id -> object_name
    # Save as arrays for better numpy compatibility
    object_id_array = []
    object_name_array = []
    for obj_id in object_ids:
        uid = str(int(obj_id))
        object_id_array.append(uid)
        object_name_array.append(uid_to_name[uid])

    # Convert to numpy arrays
    object_id_array = np.array(object_id_array, dtype=object)
    object_name_array = np.array(object_name_array, dtype=object)
    
    # Save annotation.npz
    annotation_file = osp.join(seq_dir, "annotation.npz")
    np.savez_compressed(
        annotation_file, 
        mask=mask, 
        name=names, 
        object_id=object_id_array,
        object_name=object_name_array
    )
    
    # Get contact labels from per-object records inside all_data[seq]["objects"]
    contact_labels = []
    for obj_id in object_ids:
        obj_meta = all_contact_data[seq]["objects"].get(obj_id)
        contact = obj_meta["contact_lr"]
        contact_labels.append(contact)
    
    # Stack to get (T, O, 2)
    contact_label = np.stack(contact_labels, axis=1)  # (T, O, 2)
    
    # Save gt_contact.npz
    contact_file = osp.join(seq_dir, "gt_contact.npz")
    np.savez_compressed(contact_file, contact_label=contact_label)
    
    print(f"Created folder structure for {seq}: {seq_dir}")
    print(f"  - Images: {len(image_list)} frames")
    print(f"  - Mask shape: {mask.shape}")
    print(f"  - Contact label shape: {contact_label.shape}")
    print(f"  - Names: {names}")
    print(f"  - Object IDs: {object_id_array}")
    print(f"  - Object names: {object_name_array}")
    
    return seq_dir

def create_all_overlay(save_dir, split):
    # Load split
    split_file = osp.join(data_dir, "sets", "split.json")
    with open(split_file, "r") as f:
        split_dict = json.load(f)
    seqs = split_dict[split]
    for seq in tqdm(seqs[::10]):
        create_overlay(seq)

def create_all_folders(save_dir, split, max_num=None):

    # Load data
    data_file = osp.join(data_dir, "preprocess", "dataset_contact.pkl")
    all_data = load_pickle(data_file)
    all_contact_data = all_data  # Contact data is in the same file
    
    # Load split
    split_file = osp.join(data_dir, "sets", "split.json")
    with open(split_file, "r") as f:
        split_dict = json.load(f)
    seqs = split_dict[split]
    if split != "test50":
        # first need to substract test50 from split
        test50_seqs = split_dict["test50"]
        seqs = [seq for seq in seqs if seq not in test50_seqs]

    
    for seq in tqdm(seqs[::10][:max_num]):
        create_one_folder(seq, t_index=0, save_dir=save_dir, all_contact_data=all_contact_data, all_data=all_data)
        

def create_mask_overlay_image(image_path, mask_data, frame_idx, draw_numbers=True):
    """
    Create a mask-overlaid image for VLM prompting with numbered labels.
    
    Args:
        image_path: Path to the original image
        mask_data: Dictionary with 'mask' (T, 2+O, H, W) and 'name' (2+O,) arrays
        frame_idx: Which frame to use (default: 0)
        draw_numbers: Whether to draw numbers on masks for better grounding (default: True)
    
    Returns:
        PIL Image with masks overlaid and numbered labels
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Load original image
    img = np.array(Image.open(image_path))
    H, W = img.shape[:2]
    
    # Get masks for this frame
    mask = mask_data['mask'][frame_idx]  # (2+O, H, W)
    names = mask_data['name']  # (2+O,)
    
    # Create overlay image
    overlay = img.copy().astype(np.float32)
    
    # Color scheme: Left hand (green), Right hand (blue), Objects (distinct colors)
    colors = {
        'left_hand': np.array([0, 255, 0]),      # Green
        'right_hand': np.array([255, 0, 0]),     # Red 
    }
    
    # Generate distinct colors for objects (up to 33 objects)
    # Use HSV color space with better distribution for maximum distinguishability
    def generate_distinct_colors(n):
        """Generate n visually distinct colors using HSV color space."""
        import colorsys
        colors_list = []
        # Use different strategies for better color separation
        for i in range(n):
            # Distribute hues evenly around the color wheel
            hue = (i / n) % 1.0
            
            # Vary saturation and brightness in patterns to maximize contrast
            # Alternate between high/low saturation and brightness
            sat_pattern = [0.8, 0.9, 0.7, 0.85, 0.75, 0.95]
            val_pattern = [0.9, 0.8, 0.95, 0.85, 0.9, 0.75]
            
            saturation = sat_pattern[i % len(sat_pattern)]
            value = val_pattern[i % len(val_pattern)]
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to BGR format (0-255 range)
            bgr = np.array([int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)])
            colors_list.append(bgr)
        return colors_list
    
    object_colors = generate_distinct_colors(33)  # Max 33 objects
    
    # Store label positions for drawing numbers
    label_positions = []
    hand_markers = []
 
    for i, name in enumerate(names):
        mask_i = mask[i].astype(np.float32)  # (H, W)
        
        if name == 'left_hand':
            color = colors['left_hand']
            label = "L"  # Left hand
            is_hand = True
        elif name == 'right_hand':
            color = colors['right_hand']
            label = "R"  # Right hand
            is_hand = True
        else:
            # Object: use distinct color from generated palette
            obj_idx = i - 2  # Subtract 2 for left/right hand
            color = object_colors[obj_idx % len(object_colors)]
            label = str(obj_idx + 1)  # Object number (1-indexed)
            is_hand = False
        
        mask_binary = mask_i > 0.5

        if not is_hand:
            # Overlay object mask with transparency
            alpha = 0.4  # Transparency
            mask_3d = np.stack([mask_i, mask_i, mask_i], axis=-1)  # (H, W, 3)
            overlay = overlay * (1 - alpha * mask_3d) + np.array(color) * (alpha * mask_3d)

            if draw_numbers and np.any(mask_binary):
                y_coords, x_coords = np.where(mask_binary)
                if len(y_coords) > 0:
                    centroid_y = int(np.mean(y_coords))
                    centroid_x = int(np.mean(x_coords))
                    label_positions.append((centroid_x, centroid_y, label, name))
        else:
            if draw_numbers and np.any(mask_binary):
                y_coords, x_coords = np.where(mask_binary)
                if len(y_coords) > 0:
                    centroid_y = int(np.mean(y_coords))
                    centroid_x = int(np.mean(x_coords))
                    hand_markers.append((centroid_x, centroid_y, label, color))
 
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(overlay)
 
    # Draw numbers/labels on the image
    if draw_numbers:
        draw = ImageDraw.Draw(pil_image)
 
        # Use default font
        font = ImageFont.load_default()
 
        # Draw hand markers first (filled circles with outline)
        for x, y, label, color in hand_markers:
            radius = 5
            bbox = [
                (x - radius, y - radius),
                (x + radius, y + radius),
            ]
            draw.ellipse(bbox, fill=tuple(color.tolist()))
 
            # Center text
            if hasattr(draw, "textbbox"):
                tb = draw.textbbox((0, 0), label, font=font)
                text_w = tb[2] - tb[0]
                text_h = tb[3] - tb[1]
            else:
                text_w, text_h = font.getsize(label)
            text_x = x - text_w // 2
            text_y = y - text_h // 2
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

        for x, y, label, name in label_positions:
            # Draw text with outline for better visibility
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), label, fill=(0, 0, 0), font=font)

            draw.text((x, y), label, fill=(255, 255, 255), font=font)
    
    return pil_image

def create_overlay(seq):
    seq_folder = osp.join(save_dir, seq)
    mask_data = np.load(osp.join(seq_folder, "annotation.npz"))
    for i in tqdm(range(mask_data['mask'].shape[0])):
        image_path = osp.join(seq_folder, "images", f"{i:04d}.jpg")
        overlay_image = create_mask_overlay_image(image_path, mask_data, frame_idx=i)
        os.makedirs(osp.join(seq_folder, "overlay"), exist_ok=True)
        overlay_image.save(osp.join(seq_folder, "overlay", f"{i:04d}.jpg"))


def get_optimal_vlm_prompt(object_names):
    """
    Generate the optimal prompt for VLM hand-object contact detection.
    
    Args:
        object_names: List of object names to evaluate
    
    Returns:
        Tuple of (system_instruction, user_prompt)
    """
    # system_instruction = """You are a precise visual classifier for hand-object contact detection in cluttered scenes. Analyze images to determine if hands are actually touching objects (not just reaching)."""
    system_instruction = """You are a precise visual classifier for hand-object contact detection in cluttered scenes.

CRITICAL CONSTRAINTS:
1. Each hand (left/right) can be in contact with AT MOST ONE object at a time.
2. "In contact" means direct physical touch: grasping, holding, pressing, or any visible contact.
3. If a hand is not clearly touching any object, you must mark all objects as 0 for that hand.
"""
    # Format object names list
    object_names_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(object_names)])
    
    user_prompt = f"""Analyze this image for hand–object contact (actual touching, not just reaching).

VISUAL GUIDANCE:
The image has been annotated with colored masks:
- GREEN mask = Left hand
- RED mask = Right hand
- Other COLORED masks = Candidate objects (each object has a unique color)

CANDIDATE OBJECTS (in order):
{object_names_list}

STRICT DEFINITION OF CONTACT:
For this task, contact means clear physical touching in this frame only.

Contact (label = 1) requires BOTH:
1. Mask intersection:
   - The hand mask and the object mask share some pixels or directly overlap at the boundary (no visible gap).
2. Touching region:
   - The overlap is at a plausible touching area (finger tips, fingers, palm, side of hand) on the visible surface of the object.

NO Contact (label = 0) in all of these cases:
- The hand is reaching toward, hovering above, or very close to an object with a visible gap between masks.
- The hand is aligned in depth (e.g., above or behind the object) but the masks do not intersect.
- The hand is in a pose that suggests future contact, but there is no current touching in this single frame.
- There is only a tiny, ambiguous intersection (1–2 pixels) that could be noise or occlusion. In such uncertain cases, choose 0 (no contact).

IMPORTANT:
- Reaching or hovering is NOT contact.
- If you are unsure whether contact is happening, choose 0 (no contact).

SPATIAL REASONING STEPS:
1. For each object:
   - Check if its mask intersects with the Left hand (GREEN) mask.
   - Check if its mask intersects with the Right hand (RED) mask.
2. If there is no mask intersection, label that hand–object pair as 0 (no contact).
3. If there is mask intersection, zoom in mentally on fingertip regions to confirm whether the contact is real. Assess if it looks like true touching (fingers/palm on the object) versus:
   - accidental overlap from occlusion,
   - motion blur,
   - or ambiguous edge contact.
   Only if it clearly looks like touching, assign 1 (contact).
4. If a hand appears to touch multiple objects, choose the object with the largest, clearest overlap as contact = 1, and set contact = 0 for the others.

CONSTRAINTS (VALIDATION CHECK):
- Each hand can touch AT MOST ONE object.
  - Sum of left across all objects must be ≤ 1.
  - Sum of right across all objects must be ≤ 1.
- If a hand is not clearly touching any object, it should have 0 for all objects.

OUTPUT FORMAT:
Return only a JSON object in this exact format (no extra text):

{{
  "obj1": {{"left": 0, "right": 1}},
  "obj2": {{"left": 0, "right": 0}},
  "obj3": {{"left": 1, "right": 0}}
}}

Where:
- 1 = the specified hand is clearly touching that object in this frame.
- 0 = the specified hand is not touching that object in this frame."""

    return system_instruction, user_prompt


def call_vlm_for_contact(
    image_path,
    overlay_image_path,
    object_names,
    model=DEFAULT_VLM_MODEL,
    temperature=0.0,
    few_shot_examples=None,
):
    """
    Call VLM API to detect hand-object contact.
    
    Args:
        image_path: Path to original image (for reference)
        overlay_image_path: Path to mask-overlaid image (for VLM input)
        object_names: List of object names to evaluate
        model: VLM model name (default: DEFAULT_VLM_MODEL). Supported: "gpt-4o", "gpt-5", "gpt-5-pro"
        temperature: Temperature parameter for API call (default: 0.0)
                     - 0.0 = deterministic output (not supported by GPT-5 family)
                     - Higher values = more creative/varied output
        few_shot_examples: Optional list of example dicts, each containing
                            keys like 'seq', 'frame_idx', 'overlay_image',
                            'object_names', and 'ground_truth' to be used as
                            in-context learning examples.
    
    Returns:
        List of contact predictions: [{"object_name": str, "prompt_id": int, "left_hand_contact": 0|1, "right_hand_contact": 0|1}, ...]
    """
    from openai import OpenAI
    import base64

    if model not in SUPPORTED_VLM_MODELS:
        raise ValueError(
            f"Unsupported VLM model '{model}'. Supported models: {sorted(SUPPORTED_VLM_MODELS)}"
        )
    
    # Load the prompt generator
    system_instruction, user_prompt = get_optimal_vlm_prompt(object_names)
    
    # Convert overlay image to data URL
    def image_file_to_data_url(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
 
    overlay_url = image_file_to_data_url(overlay_image_path)
 
    # Build JSON schema matching the output format
    def build_json_schema_for_objects():
        num_objects = len(object_names)
        properties = {}
        for i in range(1, num_objects + 1):
            obj_key = f"obj{i}"
            properties[obj_key] = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "left": {"type": "integer", "enum": [0, 1]},
                    "right": {"type": "integer", "enum": [0, 1]},
                },
                "required": ["left", "right"],
            }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties.keys()),
        }

        return {
            "name": "HandObjectContact",
            "strict": True,
            "schema": schema,
        }

    json_schema_meta = build_json_schema_for_objects()

    few_shot_examples = few_shot_examples or []

    responses_messages = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_instruction}],
        }
    ]

    chat_messages = [
        {"role": "system", "content": system_instruction}
    ]

    if few_shot_examples:
        for idx, example in enumerate(few_shot_examples, start=1):
            example_overlay_path = example.get("overlay_image")
            example_ground_truth = example.get("ground_truth", {})
            example_object_names = example.get("object_names", [])
            example_seq = example.get("seq")
            example_frame = example.get("frame_idx")

            if example_overlay_path is None or not osp.exists(example_overlay_path):
                print(f"[VLM] Example overlay missing: {example_overlay_path}")
                continue

            overlay_url_example = image_file_to_data_url(example_overlay_path)
            object_names_list = "\n".join(
                [f"{i + 1}. {name}" for i, name in enumerate(example_object_names)]
            )
            gt_json_str = json.dumps(example_ground_truth, indent=2)

            header_lines = [
                f"Example {idx} (Ground Truth)",
            ]
            if example_seq is not None and example_frame is not None:
                try:
                    frame_disp = int(example_frame)
                    header_lines.append(f"Sequence {example_seq}, frame {frame_disp:04d}")
                except Exception:
                    header_lines.append(f"Sequence {example_seq}, frame {example_frame}")
            if object_names_list:
                header_lines.append("Candidate objects:")
                header_lines.append(object_names_list)
            header_lines.append("Respond only with the JSON contact labels for this frame.")
            example_user_text = "\n".join(header_lines)

            responses_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": example_user_text},
                        {"type": "input_image", "image_url": overlay_url_example},
                    ],
                }
            )
            responses_messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": gt_json_str},
                    ],
                }
            )

            chat_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example_user_text},
                        {"type": "image_url", "image_url": {"url": overlay_url_example}},
                    ],
                }
            )
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": gt_json_str},
                    ],
                }
            )

    responses_messages.append(
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt.strip()},
                {"type": "input_image", "image_url": overlay_url},
            ],
        }
    )

    chat_messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt.strip()},
                {"type": "image_url", "image_url": {"url": overlay_url}},
            ],
        }
    )
 
    # Handle temperature constraints for newer models
    if model in {GPT5_MODEL, GPT5_PRO_MODEL}:
        effective_temperature = max(0.0, min(temperature, 1.0))
        if not math.isclose(effective_temperature, temperature):
            print(
                f"[VLM] Clamping temperature from {temperature} to {effective_temperature} for model '{model}'."
            )
    else:
        effective_temperature = temperature
    print(f"Effective temperature: {effective_temperature}")
 
    # Call VLM API with temperature parameter
    client = OpenAI()  # uses OPENAI_API_KEY from env
 
    if model in {GPT5_MODEL, GPT5_PRO_MODEL}:
        # GPT-5 family currently requires the Responses API with reasoning enabled
        max_output_tokens = 2048
        reasoning_effort = "high" if model == GPT5_PRO_MODEL else "medium"
        content = ""
        last_error_resp = None

        for attempt in range(3):
            response_kwargs = dict(
                model=model,
                reasoning={"effort": reasoning_effort},
                input=responses_messages,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": json_schema_meta["name"],
                        "schema": json_schema_meta["schema"],
                        "strict": json_schema_meta["strict"],
                    }
                },
                max_output_tokens=max_output_tokens,
            )

            # temperature is not currently supported for GPT-5 Responses API calls
            resp = client.responses.create(**response_kwargs)

            # Collect text output from responses API
            content_parts = []
            if getattr(resp, "output", None):
                for output in resp.output:
                    for item in getattr(output, "content", []) or []:
                        if getattr(item, "type", "") in {"output_text", "text"}:
                            content_parts.append(getattr(item, "text", ""))
            if not content_parts and getattr(resp, "output_text", None):
                content_parts.append(resp.output_text)

            content = "".join(content_parts).strip()
            if content:
                break

            last_error_resp = resp

            # If the response was truncated due to max_output_tokens, bump the limit and retry
            incomplete_reason = getattr(resp, "incomplete_details", None)
            if (
                getattr(resp, "status", "") == "incomplete"
                and incomplete_reason is not None
                and getattr(incomplete_reason, "reason", "") == "max_output_tokens"
            ):
                print(
                    f"[VLM][{model}] output truncated at {max_output_tokens} tokens; "
                    f"retrying with budget {min(max_output_tokens * 2, 1024)} and reasoning_effort='low'."
                )
                max_output_tokens = min(max_output_tokens * 2, 1024)
                reasoning_effort = "low"
                continue

            # Otherwise try once more with reduced reasoning effort if possible
            if reasoning_effort != "minimal":
                print(f"[VLM][{model}] empty output; retrying with reasoning_effort='minimal'.")
                reasoning_effort = "minimal"
                continue

        if not content:
            raise RuntimeError(
                f"Empty response content from model {model}: {last_error_resp or resp}"
            )
    else:
        resp = client.chat.completions.create(
            model=model,
            temperature=effective_temperature,
            messages=chat_messages,
            response_format={"type": "json_schema", "json_schema": json_schema_meta},
        )
        content = resp.choices[0].message.content
    
    # Parse response - new format: {"obj1": {"left": 0, "right": 1}, "obj2": {"left": 0, "right": 0}, ...}
    parsed = json.loads(content)
 
    # Align with object_names order
    out = []
    for i, name in enumerate(object_names):
        obj_key = f"obj{i+1}"  # obj1, obj2, obj3, ...
        obj_data = parsed.get(obj_key)
        if obj_data is None:
            raise RuntimeError(f"Missing prediction for {obj_key} (object: '{name}'). Got keys: {list(parsed.keys())}")
        L = obj_data.get("left")
        R = obj_data.get("right")
        if L not in (0, 1) or R not in (0, 1):
            raise RuntimeError(f"Bad labels for '{name}' ({obj_key}): {obj_data}")
        out.append(
            {
                "object_name": name,
                "prompt_id": i + 1,
                "left_hand_contact": int(L),
                "right_hand_contact": int(R),
            }
        )
    
    return out


def query_image_list(
    image_path_list,
    vlm_output_dir=vlm_output_dir,
    model=DEFAULT_VLM_MODEL,
    temperature=0.0,
    train_indices=None,
    example_dir=None,
):
    """
    Query a list of overlay images using VLM and save predictions.
    
    Args:
        image_path_list: List of overlay image paths (e.g., ["outputs/debug_vlm/001917/overlay/0000.jpg", ...])
        vlm_output_dir: Directory to save VLM predictions (default: vlm_output_dir)
        model: VLM model name (default: DEFAULT_VLM_MODEL). Supported: "gpt-4o", "gpt-5", "gpt-5-pro"
        temperature: Temperature parameter for API calls (default: 0.0)
                     - 0.0 = deterministic output (not supported by GPT-5 family)
                     - Higher values = more creative/varied output
        train_indices: Optional list of strings formatted as "seq_frame" indicating
                        which precomputed GT examples to use for in-context learning.
        example_dir: Directory containing example JSON/overlay pairs created via
                      creat_examples(). Defaults to '{save_dir}_trainset/examples'.
    
    Saves:
        Predictions as JSON files under vlm_output_dir/
        Format: {seq}/{frame_idx:04d}_prediction.json
    """
    os.makedirs(vlm_output_dir, exist_ok=True)

    if train_indices is None and 'train_index_list' in globals():
        train_indices = train_index_list
    if example_dir is None:
        example_dir = osp.join(save_dir + '_trainset', 'examples')

    few_shot_examples = load_few_shot_examples(example_dir, train_indices)

    for overlay_image_path in tqdm(image_path_list, desc="Querying VLM"):
        # Extract sequence and frame index from overlay image path
        # Expected format: {save_dir}/{seq}/overlay/{frame_idx:04d}.jpg
        overlay_image_path = osp.abspath(overlay_image_path)
        overlay_basename = osp.basename(overlay_image_path)  # e.g., "0000.jpg"
        overlay_dir = osp.dirname(overlay_image_path)  # e.g., "outputs/debug_vlm/001917/overlay"
        seq_dir = osp.dirname(overlay_dir)  # e.g., "outputs/debug_vlm/001917"
        seq = osp.basename(seq_dir)  # e.g., "001917"
        
        # Extract frame index from filename (e.g., "0000.jpg" -> 0)
        frame_idx_str = osp.splitext(overlay_basename)[0]  # e.g., "0000"
        try:
            frame_idx = int(frame_idx_str)
        except ValueError:
            raise ValueError(f"Could not parse frame index from filename: {overlay_basename}")
        
        # Load annotation.npz to get object names
        annotation_file = osp.join(seq_dir, "annotation.npz")
        if not osp.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        annotation_data = np.load(annotation_file, allow_pickle=True)
        names = annotation_data['name']  # (2+O,) array: ["left_hand", "right_hand", obj1_name, obj2_name, ...]
        
        # Extract object names (skip "left_hand" and "right_hand")
        object_names = []
        for name in names:
            if name not in ["left_hand", "right_hand"]:
                object_names.append(str(name))
        
        if len(object_names) == 0:
            print(f"Warning: No objects found for {seq} frame {frame_idx}, skipping...")
            continue
        
        # Get original image path (for reference, though not used in call_vlm_for_contact)
        original_image_path = osp.join(seq_dir, "images", f"{frame_idx:04d}.jpg")
        if not osp.exists(original_image_path):
            print(f"Warning: Original image not found: {original_image_path}, using overlay path as reference")
            original_image_path = overlay_image_path
        
        # Query VLM
        try:
            # GPT-5 family does not support temperature=0.0
            if model in {GPT5_MODEL, GPT5_PRO_MODEL} and temperature == 0:
                temperature = 0.1
            predictions = call_vlm_for_contact(
                image_path=original_image_path,
                overlay_image_path=overlay_image_path,
                object_names=object_names,
                model=model,
                temperature=temperature,
                few_shot_examples=few_shot_examples,
            )
        except Exception as e:
            print(f"Error querying VLM for {overlay_image_path}: {e}")
            continue
        
        # Save prediction
        seq_output_dir = osp.join(vlm_output_dir, seq)
        os.makedirs(seq_output_dir, exist_ok=True)
        
        prediction_file = osp.join(seq_output_dir, f"{frame_idx:04d}_prediction.json")
        
        # Format prediction for saving
        prediction_data = {
            "seq": seq,
            "frame_idx": frame_idx,
            "overlay_image_path": overlay_image_path,
            "object_names": object_names,
            "predictions": predictions,
        }
        
        with open(prediction_file, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        print(f"Saved prediction: {prediction_file}")


def evaluate_vlm_predictions(
    predictions_dir,
    output_path=None,
):
    """
    Evaluate VLM predictions stored as JSON files against ground-truth contact labels.

    Args:
        predictions_dir: Directory containing per-sequence prediction JSON files
                         (layout: {predictions_dir}/{seq}/{frame_idx:04d}_prediction.json)
        output_path: Optional path to save evaluation summary as JSON. Defaults to
                     {predictions_dir}/metrics_summary.json.

    Returns:
        Dictionary with metrics for left hand, right hand, and combined pairs.
    """

    def safe_div(num, denom):
        return float(num) / float(denom) if denom else None

    def update_counts(counts, gt_label, pred_label):
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

    def summarize_counts(counts):
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
            "counts": {
                "tp": counts["tp"],
                "tn": counts["tn"],
                "fp": counts["fp"],
                "fn": counts["fn"],
            },
            "metrics": {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "positive_rate": safe_div(positives, total),
                "negative_rate": safe_div(negatives, total),
            },
            "support": {
                "total_pairs": total,
                "positive_pairs": positives,
                "negative_pairs": negatives,
            },
        }

    prediction_files = sorted(glob(osp.join(predictions_dir, "*", "*_prediction.json")))
    if len(prediction_files) == 0:
        print(f"No prediction files found under {predictions_dir}.")
        return {}

    left_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    right_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    skipped = 0

    for prediction_file in tqdm(prediction_files, desc="Evaluating VLM predictions"):
        with open(prediction_file, "r") as f:
            prediction_data = json.load(f)

        seq = prediction_data.get("seq")
        frame_idx = prediction_data.get("frame_idx")
        overlay_image_path = prediction_data.get("overlay_image_path")
        predictions = prediction_data.get("predictions", [])

        if overlay_image_path is None:
            print(f"Skipping {prediction_file}: missing overlay_image_path")
            skipped += 1
            continue

        overlay_dir = osp.dirname(overlay_image_path)
        seq_dir = osp.dirname(overlay_dir)
        gt_contact_file = osp.join(seq_dir, "gt_contact.npz")

        if not osp.exists(gt_contact_file):
            print(f"Ground-truth contact file not found for {seq}: {gt_contact_file}. Skipping.")
            skipped += 1
            continue

        gt_contact = np.load(gt_contact_file)
        if "contact_label" not in gt_contact:
            print(f"contact_label missing in {gt_contact_file}; skipping.")
            skipped += 1
            continue

        contact_label = gt_contact["contact_label"]  # (T, O, 2)

        num_objects = contact_label.shape[1]

        for pred in predictions:
            obj_idx = pred.get("prompt_id", 0) - 1
            if obj_idx < 0 or obj_idx >= num_objects:
                print(
                    f"Prediction object index {obj_idx} out of range for seq {seq} frame {frame_idx}; skipping pair."
                )
                skipped += 1
                continue

            # view 002043/0085 contact gt. print out where gt is 1. 
            gt_left = int(contact_label[frame_idx, obj_idx, 0])
            gt_right = int(contact_label[frame_idx, obj_idx, 1])
            pred_left = int(pred.get("left_hand_contact", 0))
            pred_right = int(pred.get("right_hand_contact", 0))

            update_counts(left_counts, gt_left, pred_left)
            update_counts(right_counts, gt_right, pred_right)

    overall_counts = {
        key: left_counts[key] + right_counts[key]
        for key in ("tp", "tn", "fp", "fn")
    }

    summary = {
        "left_hand": summarize_counts(left_counts),
        "right_hand": summarize_counts(right_counts),
        "overall": summarize_counts(overall_counts),
        "skipped_pairs": skipped,
    }

    if output_path is None:
        output_path = osp.join(predictions_dir, "metrics_summary.json")

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved evaluation summary to {output_path}")
    # print F1
    print(f"F1: {summary['overall']['metrics']['f1']}")
    return summary


def find_key_frames(train_dir):
    """Identify key frames around contact transitions and save to JSON."""

    if not osp.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    seq_dirs = sorted(
        d for d in os.listdir(train_dir) if osp.isdir(osp.join(train_dir, d))
    )

    key_entries = []
    total_transitions = 0

    for seq in tqdm(seq_dirs, desc="Scanning key frames"):
        seq_dir = osp.join(train_dir, seq)
        contact_path = osp.join(seq_dir, "gt_contact.npz")
        if not osp.exists(contact_path):
            print(f"[KeyFrame] Missing contact labels for {seq}: {contact_path}")
            continue

        contact_npz = np.load(contact_path)
        if "contact_label" not in contact_npz:
            print(f"[KeyFrame] contact_label missing in {contact_path}")
            continue

        contact_label = np.asarray(contact_npz["contact_label"], dtype=np.int32)
        if contact_label.ndim != 3 or contact_label.shape[2] != 2:
            print(f"[KeyFrame] Unexpected contact_label shape {contact_label.shape} in {contact_path}")
            continue

        T = contact_label.shape[0]
        if T < 2:
            continue

        flat = contact_label.reshape(T, -1)
        transitions = np.where((flat[1:] != flat[:-1]).any(axis=1))[0] + 1
        if transitions.size == 0:
            continue

        total_transitions += int(transitions.size)
        candidate_frames = set()
        for t0 in transitions:
            for delta in (-3, 3):
                idx = t0 + delta
                if 0 <= idx < T:
                    candidate_frames.add(int(idx))

        for frame_idx in sorted(candidate_frames):
            key_entries.append({"seq": seq, "frame_idx": int(frame_idx)})

    summary = {
        "seq_time": key_entries,
        "metadata": {
            "num_sequences": len(seq_dirs),
            "num_transitions": total_transitions,
            "num_key_frames": len(key_entries),
        },
    }

    output_path = osp.join(train_dir, "keyframe.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Saved {len(key_entries)} key frames from {total_transitions} transitions to {output_path}"
    )

    return summary

def creat_examples(summary, example_dir):
    """Generate overlay images and GT JSON examples for in-context learning."""

    if summary is None:
        raise ValueError("Summary is None; run find_key_frames() first.")

    entries = summary.get("seq_time", [])
    if len(entries) == 0:
        print("[Examples] No key frames provided; skipping example generation.")
        return {}

    example_dir = osp.abspath(example_dir)
    os.makedirs(example_dir, exist_ok=True)

    # Infer train directory (parent of examples folder)
    train_dir = osp.dirname(example_dir)
    generated = []
    skipped = 0

    for item in tqdm(entries, desc="Preparing GT examples"):
        seq = item.get("seq")
        frame_idx = int(item.get("frame_idx", -1))

        if seq is None or frame_idx < 0:
            skipped += 1
            continue

        seq_dir = osp.join(train_dir, seq)
        annotation_path = osp.join(seq_dir, "annotation.npz")
        contact_path = osp.join(seq_dir, "gt_contact.npz")
        image_path = osp.join(seq_dir, "images", f"{frame_idx:04d}.jpg")

        if not (osp.exists(annotation_path) and osp.exists(contact_path) and osp.exists(image_path)):
            print(
                f"[Examples] Missing data for {seq}:{frame_idx:04d}; "
                f"annotation={osp.exists(annotation_path)}, contact={osp.exists(contact_path)}, image={osp.exists(image_path)}"
            )
            skipped += 1
            continue

        mask_data = np.load(annotation_path, allow_pickle=True)
        names = mask_data["name"]
        mask_array = mask_data["mask"]

        if frame_idx >= mask_array.shape[0]:
            print(
                f"[Examples] Frame {frame_idx} out of range for masks (T={mask_array.shape[0]}) in seq {seq}; skipping."
            )
            skipped += 1
            continue

        contact_npz = np.load(contact_path)
        if "contact_label" not in contact_npz:
            print(f"[Examples] contact_label missing in {contact_path}; skipping {seq}:{frame_idx:04d}.")
            skipped += 1
            continue

        contact_label = contact_npz["contact_label"]
        if frame_idx >= contact_label.shape[0]:
            print(
                f"[Examples] Frame {frame_idx} out of range for GT (T={contact_label.shape[0]}) in seq {seq}; skipping."
            )
            skipped += 1
            continue

        overlay_image = create_mask_overlay_image(
            image_path=image_path, mask_data=mask_data, frame_idx=frame_idx, draw_numbers=True
        )
        overlay_filename = f"{seq}_{frame_idx:04d}_overlay.jpg"
        overlay_path = osp.join(example_dir, overlay_filename)
        overlay_image.save(overlay_path)

        object_names = [str(name) for name in names if str(name) not in {"left_hand", "right_hand"}]
        obj_dict = {}
        for obj_idx, obj_name in enumerate(object_names):
            gt_left = int(contact_label[frame_idx, obj_idx, 0])
            gt_right = int(contact_label[frame_idx, obj_idx, 1])
            obj_dict[f"obj{obj_idx + 1}"] = {"left": gt_left, "right": gt_right}

        example_json = {
            "seq": seq,
            "frame_idx": frame_idx,
            "overlay_image": overlay_path,
            "object_names": object_names,
            "ground_truth": obj_dict,
        }

        json_filename = f"{seq}_{frame_idx:04d}.json"
        json_path = osp.join(example_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(example_json, f, indent=2)

        generated.append({"json": json_path, "overlay": overlay_path})

    summary_out = {
        "generated": generated,
        "skipped": skipped,
        "total_requested": len(entries),
    }

    manifest_path = osp.join(example_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(summary_out, f, indent=2)

    print(
        f"Prepared {len(generated)} GT examples (skipped {skipped}) under {example_dir}. "
        f"Manifest: {manifest_path}"
    )

    return summary_out



def load_few_shot_examples(example_dir, index_list):
    """Load few-shot example metadata for in-context learning."""

    index_list = index_list or []
    if not index_list:
        return []

    if example_dir is None or not osp.isdir(example_dir):
        print(f"[VLM] Example directory not found: {example_dir}")
        return []

    loaded = []
    for entry in index_list:
        try:
            seq, frame_str = entry.split("_")
            frame_idx = int(frame_str)
        except ValueError:
            print(f"[VLM] Could not parse train index '{entry}'. Expected format 'seq_frame'.")
            continue

        json_path = osp.join(example_dir, f"{seq}_{frame_idx:04d}.json")
        if not osp.exists(json_path):
            print(f"[VLM] Example JSON not found: {json_path}")
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[VLM] Failed to load example {json_path}: {exc}")
            continue

        overlay_path = data.get("overlay_image")
        ground_truth = data.get("ground_truth")
        if overlay_path is None or ground_truth is None:
            print(f"[VLM] Example missing overlay or ground_truth: {json_path}")
            continue

        loaded.append(data)

    if loaded:
        print(
            f"[VLM] Loaded {len(loaded)} few-shot examples (requested {len(index_list)}) from {example_dir}"
        )
    else:
        print(f"[VLM] No few-shot examples loaded from {example_dir}")

    return loaded


img_path_list = [
    "002043/overlay/0085.jpg",
    # "002043/overlay/0006.jpg",
    # "002043/overlay/0008.jpg",
    "002043/overlay/0146.jpg",
    "001874/overlay/0001.jpg",
    "003034/overlay/0096.jpg", 
]
img_path_list = [osp.join(save_dir, img_path) for img_path in img_path_list]


train_index_list = [
    "001885_0070",
    "001885_0076",
    "001910_0011",
    "001910_0084",
    "001971_0131", 
    "002197_0110",
    "002197_0116",
]

# train_index_list = []

if __name__ == "__main__":    
    # Example usage:
    # create_all_folders(save_dir + '_trainset', 'test', max_num=10)
    # summary =find_key_frames(save_dir + '_trainset')
    # creat_examples(summary, save_dir+'_trainset/examples')


    # create_all_folders(save_dir, split)
    # create_all_overlay(save_dir, split)
    # create_overlay(seq="001917")
    # eval_results_from_jsons()  # TODO: Define this function
    
    
    # Choose model: DEFAULT_VLM_MODEL (gpt-4o), GPT5_MODEL (gpt-5), or GPT5_PRO_MODEL
    # query_image_list(img_path_list, model=GPT5_MODEL)
    # query_image_list(img_path_list, model=DEFAULT_VLM_MODEL)
    # query_image_list(img_path_list, model=GPT5_PRO_MODEL)


    vlm_output_dir = f"outputs/debug_vlm/vlm_output-ICLx{len(train_index_list)}"
    example_dir = osp.join(save_dir + '_trainset', 'examples')
    query_image_list(
        img_path_list,
        model=GPT5_MODEL,
        train_indices=train_index_list,
        example_dir=example_dir,
        vlm_output_dir=vlm_output_dir,
    )

    evaluate_vlm_predictions(vlm_output_dir)