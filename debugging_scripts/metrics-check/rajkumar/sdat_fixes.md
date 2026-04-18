## sdat — Fix Applied (Attempt 1, 2026-04-17)
- **Root cause**: Pattern D — `semantic_diversity_score` is model_based with `helm_class: null`. Requires external embedding model `ibm-granite/granite-embedding-278m-multilingual` that cannot be wired as a standard MetricSpec.
- **Files changed**: `run_specs/sdat_run_specs.py`
- **Change summary**: Updated TODO comment to document the unimplementable metric and the required model.
- **Registry bug (⚠ tell Vijeta)**: `semantic_diversity_score` needs custom model-based metric pipeline outside HELM's MetricSpec framework
- **Result**: SKIP — unimplementable as MetricSpec
