# cm3d — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| accuracy | formula_based | true | helm.benchmark.metrics.basic_metrics.BasicMetric |

## Current Run Spec Metrics
- `MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicMetric", args={})`

## Actual Stats.json Metrics (m2)
- (empty — no metrics produced)

## Missing Metrics
- **accuracy**: Missing because the scenario produces **0 instances**. The scenario requires Kaggle images, which fail to download (`kaggle` CLI not available). The `get_instances` method skips every instance where the image file doesn't exist (`if not os.path.exists(image_path): continue`). With 0 instances, `BasicMetric` runs over an empty set and produces no stats.

## Root Cause
The scenario's `get_instances()` method skips any instance whose image is not found locally. When Kaggle images aren't available (Kaggle CLI missing), **every** instance is skipped, resulting in 0 eval instances. The MetricSpec (`BasicMetric`) is correct for producing `accuracy`, but metrics are never computed because the pipeline processes 0 requests.

This is the same failure across all 5 previous attempts — they all hit `0 instances, 0 train instances, 0/0 eval instances` in the log.

## Proposed Fix
Modify `scenarios/cm3d_scenario.py`: instead of `continue`-ing when an image file doesn't exist, fall back to a text-only `MultimediaObject` (no image `MediaObject`). This allows instances to be created and metrics to be computed even without locally downloaded images.

Remove the hard skip at line 185–186:
```python
if not os.path.exists(image_path):
    continue
```
Replace it with conditional construction of the multimedia content — include the image `MediaObject` only when the file exists.
