# Fixes: scar

## Missing metric: `f1`

**File changed**: `run_specs/scar_run_specs.py`

**What**: Added a second `MetricSpec` for `metrics.f1_metric.F1Metric`.

**Why**: `BasicGenerationMetric` names its token-level F1 stat `f1_score`, but the
registry expects the stat to be named `f1`. The custom `F1Metric` class at
`metrics/f1_metric.py` already exists and emits `Stat(MetricName("f1"))`, so
adding it as a second metric spec produces the required stat name without
removing any existing metrics.

**No other files changed.**
