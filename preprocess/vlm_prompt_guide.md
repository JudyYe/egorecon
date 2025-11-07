# VLM Prompt Engineering Guide: Hand-Object Contact Detection

## Key Principles

### 1. **Leverage the Constraint Explicitly**
The constraint "one hand can only contact at most one object" is powerful - use it to:
- Reduce ambiguity
- Enable cross-validation
- Improve accuracy

### 2. **Visual Mask Overlay is Critical**
- Overlay masks directly on the image with distinct colors
- Use semi-transparent overlays so the original image is still visible
- Color-code: Left hand (green), Right hand (blue), Objects (different colors per object)

### 3. **Structured Output with Validation**
- Use JSON schema to enforce format
- Include validation logic in the prompt
- Request confidence scores if possible

## Optimal Prompt Structure

### System Instruction
```
You are a precise visual classifier for hand-object contact detection in cluttered scenes.

CRITICAL CONSTRAINTS:
1. Each hand (left/right) can be in contact with AT MOST ONE object at a time
2. "In contact" means: direct physical touch, including grasp, hold, press, or any visible contact
3. If a hand is not touching any object, mark all objects as 0 for that hand

TASK:
For each candidate object, determine if the LEFT hand and/or RIGHT hand are physically touching it.

OUTPUT FORMAT:
Return a JSON object with contact labels for each object.
```

### User Prompt Template
```
Analyze this image for hand-object contact.

VISUAL GUIDANCE:
- The image has been annotated with colored masks:
  * GREEN mask = Left hand
  * BLUE mask = Right hand  
  * COLORED masks = Candidate objects (each object has a unique color)

CANDIDATE OBJECTS (in order):
{object_names_list}

INSTRUCTIONS:
1. For each object, determine if the LEFT hand is touching it (1) or not (0)
2. For each object, determine if the RIGHT hand is touching it (1) or not (0)
3. Remember: Each hand can touch AT MOST ONE object. If you see a hand touching multiple objects, choose the one with the strongest/most visible contact.
4. If a hand is not touching any object, all objects should have 0 for that hand.

VALIDATION CHECK:
- Left hand: Sum of left_hand_contact across all objects should be ≤ 1
- Right hand: Sum of right_hand_contact across all objects should be ≤ 1

Return the contact labels in the exact JSON format specified.
```

## Advanced Techniques

### 1. **Two-Stage Approach**
Stage 1: "Which object is the left hand touching? (answer: object_name or 'none')"
Stage 2: "Which object is the right hand touching? (answer: object_name or 'none')"
Then convert to binary format.

### 2. **Spatial Reasoning Prompt**
```
First, identify the spatial overlap between:
- Left hand mask and each object mask
- Right hand mask and each object mask

Then, determine if the overlap indicates actual physical contact (not just proximity).
Consider:
- Are the masks overlapping at the fingertips/palm?
- Is there visible deformation or pressure?
- Could this be occlusion rather than contact?
```

### 3. **Confidence-Based Approach**
Request confidence scores (0-1) and use thresholding:
- High confidence (>0.8): Direct contact
- Medium confidence (0.5-0.8): Possible contact, needs verification
- Low confidence (<0.5): No contact

## Implementation Recommendations

1. **Create Mask-Overlaid Image**: Overlay all masks on the original image with distinct colors
2. **Use Structured Output**: Enforce JSON schema with strict validation
3. **Temperature = 0.0**: Use deterministic output for consistency
4. **Multi-shot Examples**: Provide 2-3 examples in the prompt if possible
5. **Post-processing Validation**: Check constraint after VLM response

## Example Output Format
```json
{
  "predictions": [
    {"object_name": "can_parmesan", "left_hand_contact": 0, "right_hand_contact": 1},
    {"object_name": "keyboard", "left_hand_contact": 1, "right_hand_contact": 0},
    {"object_name": "mug", "left_hand_contact": 0, "right_hand_contact": 0}
  ],
  "validation": {
    "left_hand_total_contacts": 1,
    "right_hand_total_contacts": 1,
    "constraint_satisfied": true
  }
}
```


