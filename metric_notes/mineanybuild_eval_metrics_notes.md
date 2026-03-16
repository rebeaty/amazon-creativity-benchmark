# Custom Metric Requirements: MineAnyBuild

Source: mineanybuild/evaluator.py, mineanybuild/utils.py in https://github.com/MineAnyBuild/MineAnyBuild

## Output Format

### Blueprint tasks (creativity, spatial_planning, spatial_understanding)

Model output is a JSON 3D array wrapped in a markdown code block:
```
Planning Reasons: ...
Selected_block_materials = {"oak_planks": 1, "cobblestone": 2}
Blueprint: '''json [[[2,2,2],[2,2,2]],[[1,-1,1],[1,1,1]]]'''
```

Dimensions: height(y) x depth(z) x width(x). `-1` = air, positive integers
map to block materials.

### Spatial reasoning

Single token: A, B, C, D, True, or False.

### Spatial commonsense

Free-text answer, max 70 words.

## Metrics

### 1. Output Success Rate (OSR)

Fraction of outputs that parse into a valid 3D list. Parser extracts JSON
from markdown code block via regex, cleans comments and trailing commas,
validates `is_3d_list()`.

```python
def parse_blueprint(raw_string):
    """Extract and validate 3D blueprint from model output."""
    import re, json
    match = re.search(r"(?s)(?:```|''')json\n(.*?)(?:```|''')", raw_string)
    if not match:
        return None
    dirty = match.group(1)
    clean = re.sub(r'\s*#.*?$|//.*?$', '', dirty, flags=re.MULTILINE)
    clean = clean.replace('\u2010', '-').replace('\u2212', '-').strip()
    clean = re.sub(r',\s*([}\]])', r'\1', clean)
    compact = re.sub(r'\s+', '', clean)
    data = json.loads(compact)
    # Validate 3D structure
    if not all(isinstance(layer, list) and
               all(isinstance(row, list) and
                   all(not isinstance(e, list) for e in row)
                   for row in layer)
               for layer in data):
        return None
    return data
```

### 2. Block Matching Accuracy (for spatial_planning, spatial_understanding)

Direct comparison of predicted vs ground-truth 3D matrices.

```python
def block_matching(pred_blueprint, gt_blueprint):
    """Compare predicted and ground truth blueprints block-by-block."""
    h = max(len(pred_blueprint), len(gt_blueprint))
    correct, total = 0, 0
    for y in range(h):
        pred_layer = pred_blueprint[y] if y < len(pred_blueprint) else []
        gt_layer = gt_blueprint[y] if y < len(gt_blueprint) else []
        d = max(len(pred_layer), len(gt_layer))
        for z in range(d):
            pred_row = pred_layer[z] if z < len(pred_layer) else []
            gt_row = gt_layer[z] if z < len(gt_layer) else []
            w = max(len(pred_row), len(gt_row))
            for x in range(w):
                pred_val = pred_row[x] if x < len(pred_row) else -1
                gt_val = gt_row[x] if x < len(gt_row) else -1
                total += 1
                if pred_val == gt_val:
                    correct += 1
    return correct / total if total > 0 else 0.0
```

### 3. Dimension Match

Whether the predicted blueprint has the same dimensions as the ground truth.

```python
def dimension_match(pred, gt_3d_info):
    """Check if predicted blueprint matches expected dimensions."""
    pred_h = len(pred)
    pred_d = max(len(layer) for layer in pred) if pred else 0
    pred_w = max(max(len(row) for row in layer) for layer in pred) if pred else 0
    return (pred_h == gt_3d_info["height"] and
            pred_d == gt_3d_info["depth"] and
            pred_w == gt_3d_info["width"])
```

### 4. Spatial Reasoning Accuracy (exact match)

```python
accuracy = sum(1 for pred, gt in pairs if pred.strip() == gt) / len(pairs)
```

Note: Ground truth has a typo — "Ture" instead of "True" for SR_3 true
cases (144 instances). Scenario corrects this.

### 5. Spatial Commonsense Score (LLM-as-judge)

See annotator_notes.md. GPT-4.1 critic scores 0-10, success rate is
fraction with score >= 7.

## Scoring Formulas (from evaluator.py)

### Creativity task
```
score = Creativity * 0.8 + Completeness * 0.05 + Complexity * 0.05
        + Architecture_Structure * 0.05 + Overall_Aesthetic * 0.05
```

### Spatial Planning task
```
score = Completeness * 0.3 + Complexity * 0.3 + Overall_Aesthetic * 0.4
```

### Spatial Understanding task
```
score = Instruction_Following_Completeness (single dimension)
```

## Notes

- Original evaluation requires building in Minecraft via Mineflayer and
  scoring screenshots with GPT-4.1. Block matching provides a text-only
  alternative for blueprint comparison.
- Creativity task has no "correct" blueprint — the ground truth is a
  reference, not an expected answer. LLM-as-judge is the primary eval.
- Each architecture has a `difficulty_factor` computed as
  `log10(block_amount + block_amount * height + width * height * depth) - 0.4`
  for stratified analysis.
- 473 architectures from 3 sources: Grabcraft (102), Minecraft Official
  Wiki (173), Reakon (198).
