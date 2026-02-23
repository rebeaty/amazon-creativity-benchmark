# Custom Metric Requirements: TextLogo3K

Source: test.py, models/util_funcs.py in https://github.com/yizhiwang96/TextLogoLayout
Additional reference: GLDesigner (arXiv 2411.11435), LayoutGPT (NeurIPS 2023)

## Output Format

Model output should be a JSON array of bounding boxes:
```json
[
  {"char": "X", "box": [left, top, right, bottom]},
  {"char": "Y", "box": [left, top, right, bottom]}
]
```
All coordinates normalized to [0, 1]. Canvas is square (128x128 pixels).

## Metrics (computable from JSON coordinates, no rendering needed)

### 1. Overlap (lower is better)
Pairwise IoU between all predicted character bounding boxes. Characters
should not overlap in a well-designed layout.

```python
def compute_overlap(boxes):
    """boxes: list of [left, top, right, bottom] normalized to [0,1]"""
    total_overlap = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            # Compute intersection area
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            if x1 < x2 and y1 < y2:
                total_overlap += (x2 - x1) * (y2 - y1)
    return total_overlap
```

### 2. Visual Balance (lower is better)
Distance of the layout's center of mass from the canvas center (0.5, 0.5).
Measures spatial equilibrium.

```python
def compute_balance(boxes):
    """Euclidean distance of center-of-mass from canvas center"""
    cx_sum, cy_sum, area_sum = 0, 0, 0
    for b in boxes:
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        area = (b[2] - b[0]) * (b[3] - b[1])
        cx_sum += cx * area
        cy_sum += cy * area
        area_sum += area
    if area_sum == 0:
        return 1.0
    com_x = cx_sum / area_sum
    com_y = cy_sum / area_sum
    return ((com_x - 0.5)**2 + (com_y - 0.5)**2)**0.5
```

### 3. Alignment Score (higher is better)
Measures how well characters align along horizontal or vertical axes.
Computes variance of center-x, center-y, left edges, and top edges.
Lower variance in at least one axis indicates intentional alignment.

```python
def compute_alignment(boxes):
    """Min variance across alignment features (lower = better aligned)"""
    import numpy as np
    centers_x = [(b[0] + b[2]) / 2 for b in boxes]
    centers_y = [(b[1] + b[3]) / 2 for b in boxes]
    lefts = [b[0] for b in boxes]
    tops = [b[1] for b in boxes]
    # Best alignment across any axis
    return min(np.var(centers_x), np.var(centers_y),
               np.var(lefts), np.var(tops))
```

### 4. MaxIoU (higher is better)
Average IoU between optimally matched predicted and ground-truth boxes
using Hungarian matching. Primary accuracy metric.

```python
def compute_max_iou(pred_boxes, gt_boxes):
    """Average IoU with optimal matching via Hungarian algorithm"""
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    n = len(gt_boxes)
    if len(pred_boxes) != n:
        return 0.0
    # Build IoU cost matrix
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            iou_matrix[i][j] = compute_iou(pred_boxes[i], gt_boxes[j])
    # Hungarian matching (maximize IoU = minimize negative IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    return iou_matrix[row_ind, col_ind].mean()
```

### 5. Coordinate MSE (lower is better)
Mean squared error between predicted and ground-truth coordinates, assuming
characters are in the same order as the input.

### 6. Canvas Utilization (target: 0.7-0.9)
Fraction of the canvas area covered by character bounding boxes. The paper's
guidelines specify 70-90% spatial utilization.

```python
def compute_utilization(boxes):
    """Union area of all boxes / canvas area (1.0)"""
    # Approximate via grid sampling or exact polygon union
    import numpy as np
    grid = np.zeros((128, 128))
    for b in boxes:
        l, t, r, bot = [int(v * 128) for v in b]
        l, t = max(0, l), max(0, t)
        r, bot = min(128, r), min(128, bot)
        grid[t:bot, l:r] = 1
    return grid.sum() / (128 * 128)
```

## Notes

- Original paper uses FID (requires rendering) and overlap loss (pixel-level,
  requires rendering glyph images onto canvas). The metrics above are
  text-computable approximations.
- GLDesigner (2024) added Glyph Ratio Consistency (aspect ratio preservation)
  as an additional metric.
- Coordinate order matters: characters should be predicted in the same order
  as the input character list. MaxIoU with Hungarian matching handles
  order-invariant evaluation.
- The dataset is primarily Chinese characters (from Tencent Video posters).
  Character shapes affect optimal layout — this is captured by the glyph
  images but not by text alone.
