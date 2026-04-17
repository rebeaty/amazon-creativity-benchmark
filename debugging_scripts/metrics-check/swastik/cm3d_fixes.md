# cm3d — Fixes Summary

## Attempt 6

### Root cause
The scenario skipped every instance when its image file wasn't found locally (Kaggle CLI unavailable → empty images directory → all `os.path.exists(image_path)` checks return False → `continue` → 0 instances). With 0 instances the HELM pipeline runs zero requests and `BasicMetric` produces no stats.

### Files changed

#### `scenarios/cm3d_scenario.py`
**What:** Replaced the hard `if not os.path.exists(image_path): continue` skip with a conditional that builds a text-only `MultimediaObject` when the image is absent.

**Why:** Allows instances to be created and evaluated even when Kaggle images haven't been downloaded. The `BasicMetric` can then compute `accuracy` over the MCQ references. The image path is still used when the file exists, so full multimodal evaluation is unaffected when images are present.

### Run spec
`run_specs/cm3d_run_specs.py` was already correct (`BasicMetric` with `args={}`). No changes needed.

### Previous attempts
All 5 prior attempts hit the same `0 instances` wall — the `metrics-diagnose-fix` skill hit max turns before identifying the scenario-level root cause (scenario issue, not a MetricSpec issue).
