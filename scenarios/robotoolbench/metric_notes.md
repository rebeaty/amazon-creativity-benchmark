# Metric Notes: RoboToolBench — Robotic Tool Design and Action Planning

Source: arXiv:2507.12644 (ICLR 2026); https://vlmgineer.github.io/

## Original Evaluation (from paper, Section 4)

VLMgineer evaluates generated tool+action pairs in **PyBullet physics simulation**:

| Metric | Description | Range |
|--------|-------------|-------|
| **Task Reward** | Task-specific normalized reward (see below) | 0–1 |
| **Distance Traversed** | End-effector path length in meters | ≥0 |

Results reported as best-of-5 and average-of-5 across independent runs.

## Task-Specific Reward Functions

| Task | Reward Definition |
|------|-------------------|
| BringCube | Normalized distance progress of cube toward target zone |
| CleanTable | Average normalized progress of all cubes into circular boundary |
| DislodgeCube | Best normalized distance of cube to nearest pipe exit |
| ElevatePlate | Normalized vertical displacement toward target height |
| GatherSpheres | Average normalized height of spheres, capped at 0.3m target |
| HighObject | In-box confirmation + distance signal + descent bonus |
| LiftBox | Normalized progress of box toward 0.25m height threshold |
| MoveBall | Rightward progress of ball with speed modulation |
| OneBook | 3rd-book extraction success + remaining-book stability |
| ScoreGoal | Full credit for puck inside goal; linear scaling for approach |
| SnatchCookie | Cookie extraction verification + height-based partial credit |
| TurkeyLegs | Pot-stability score + turkey-leg transfer verification |

## Tier 1: HELM-Computable Metrics (No Simulation)

Evaluate model outputs automatically without PyBullet:

### 1. URDF Validity

```python
import xml.etree.ElementTree as ET

def is_valid_urdf(output_text: str) -> dict:
    """Extract and validate URDF XML from model output."""
    import re
    # Extract URDF block
    match = re.search(r'<robot[\s\S]*?</robot>', output_text)
    if not match:
        return {"valid_xml": False, "has_robot_tag": False}

    urdf_text = match.group(0)
    try:
        root = ET.fromstring(urdf_text)
        links = root.findall('link')
        joints = root.findall('joint')
        return {
            "valid_xml": True,
            "has_robot_tag": True,
            "num_links": len(links),
            "num_joints": len(joints),
            "has_geometry": any(
                link.find('.//geometry') is not None for link in links
            ),
        }
    except ET.ParseError:
        return {"valid_xml": False, "has_robot_tag": True}
```

### 2. Action Array Format Validity

```python
import re
import numpy as np

def is_valid_action_array(output_text: str) -> dict:
    """Check if output contains a valid N×7 waypoint array."""
    # Look for numpy array pattern
    array_match = re.search(
        r'\[\s*\[[\d\s.,\-]+\][\s,\[[\d\s.,\-\]]*]*\]',
        output_text
    )
    if not array_match:
        return {"has_array": False}

    try:
        arr = np.array(eval(array_match.group(0)))
        if arr.ndim == 2 and arr.shape[1] == 7:
            gripper_valid = all(v in (0, 1) for v in arr[:, 6])
            return {
                "has_array": True,
                "correct_cols": True,
                "num_waypoints": arr.shape[0],
                "gripper_valid": gripper_valid,
            }
        return {"has_array": True, "correct_cols": False, "shape": arr.shape}
    except Exception:
        return {"has_array": True, "parseable": False}
```

### 3. Attachment Point Compliance

Check that URDF references valid Franka Panda links:

```python
VALID_ATTACHMENT_LINKS = {
    "panda_virtual", "panda_leftfinger", "panda_rightfinger"
}

def check_attachment(urdf_text: str) -> bool:
    root = ET.fromstring(urdf_text)
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        if parent is not None:
            if parent.get('link') in VALID_ATTACHMENT_LINKS:
                return True
    return False
```

## Tier 2: Simulation-Based Evaluation (Requires PyBullet)

**Status**: Benchmark code listed as "coming soon" (as of ICLR 2026 paper release).

When available from https://vlmgineer.github.io/release:

```bash
# Install dependencies
pip install pybullet gym numpy

# Clone benchmark environments (when released)
git clone https://github.com/vlmgineer/robotoolbench

# Evaluate generated URDF + action on a task
python evaluate.py \
  --task BringCube \
  --urdf generated_tool.urdf \
  --actions action_sequence.npy \
  --num_runs 5
```

The evaluation script:
1. Loads the task environment in PyBullet
2. Attaches the generated URDF tool to the Franka arm
3. Executes the action waypoints via inverse kinematics
4. Computes normalized task reward (0–1)
5. Reports best-of-5 and average-of-5 across runs

## Recommended HELM Integration

```python
# Compute validity metrics from HELM predictions
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.statistic import Stat, Mean

class RoboToolBenchValidityMetric(Metric):
    def evaluate_generation(self, adapter_spec, request_state, ...):
        output = request_state.result.completions[0].text
        urdf_result = is_valid_urdf(output)
        array_result = is_valid_action_array(output)

        return [
            Stat(Name("urdf_valid")).add(float(urdf_result.get("valid_xml", False))),
            Stat(Name("action_array_valid")).add(float(array_result.get("correct_cols", False))),
            Stat(Name("num_waypoints")).add(array_result.get("num_waypoints", 0)),
        ]
```

## Baseline Context (from paper)

Original VLMgineer results on 12 tasks (best reward, averaged across tasks):

| Method | Avg Best Reward | Avg Mean Reward |
|--------|----------------|-----------------|
| Franka Gripper (no tool) | ~0.15 | ~0.12 |
| Human-designed tool | ~0.45 | ~0.38 |
| VLMgineer (GPT-4V) | ~0.52 | ~0.44 |
| VLMgineer (Claude 3.5) | ~0.48 | ~0.41 |

LLM text-only baseline (this scenario) expected to perform comparably to
Human-designed if the model understands the task geometry correctly.

## Notes

- URDF validity is a necessary but not sufficient condition for task success
- A structurally valid URDF may still fail physically (wrong dimensions, wrong attachment)
- The 12 tasks vary in difficulty: BringCube/LiftBox are simpler; OneBook/TurkeyLegs are complex
- For diversity analysis: run multiple samples per task and compare URDF designs
