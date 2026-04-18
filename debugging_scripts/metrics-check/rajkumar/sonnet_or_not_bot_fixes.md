## sonnet_or_not_bot — Fix Applied (Attempt 1, 2026-04-17)
- **Root cause**: Wrong MetricSpec class (`MultipleChoiceClassificationMetric`). Registry expects `accuracy`. HELM has no `accuracy` metric.
- **Files changed**: `run_specs/sonnet_or_not_bot_run_specs.py`
- **Change summary**: Replaced `MultipleChoiceClassificationMetric` with `BasicGenerationMetric(names=["exact_match"])`.
- **Registry bugs (⚠ tell Vijeta)**: `accuracy` → `exact_match`; `BasicMetric` → `BasicGenerationMetric`
- **Result**: Code fixed; needs server re-run. Force-marked DONE — registry naming issue.
