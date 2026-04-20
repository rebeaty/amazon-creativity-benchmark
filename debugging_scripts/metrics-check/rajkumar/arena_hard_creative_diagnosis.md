# arena_hard_creative — Metrics Diagnosis (Attempt 1)

## Expected Metrics (m1)
| Metric | Type | in_helm | helm_class |
|--------|------|---------|------------|
| `win_rate` | llm_judge | false | null |

## Run Spec Currently Has
- `MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "win_rate"})` ✓
- `AnnotatorSpec(class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator", args={...})` ✓
- `annotators=annotators` in RunSpec ✓
- Rubric `_RUBRIC_WIN_RATE` defined ✓
- **ScenarioSpec `class_name="scenarios_new.arena_hard_creative_scenario.ArenaHardCreativeScenario"`** ✗

## Actual Output (m2)
Empty — both stats.json files (Phase 1 and fresh re-run) contain 0 entries total.
No instances were evaluated at all.

## Missing
- `win_rate`

## Root Cause
The `ScenarioSpec.class_name` references `scenarios_new.arena_hard_creative_scenario`, but
the `scenarios_new/` directory no longer exists — it was renamed to `scenarios/` during the
repo refactor. HELM cannot import the scenario class, so zero instances are loaded and
zero metrics are produced. The stats.json is written but empty.

The MetricSpec/AnnotatorSpec wiring is otherwise correct — this is purely a class path bug.

## Proposed Fix
Two bugs:

**Fix 1 (systemic):** Change `class_name` in `ScenarioSpec` from `scenarios_new.*` → `scenarios.*`
Applied via bulk sed across all 313 run_spec files.

**Fix 2 (scenario parser):** In `scenarios/arena_hard_creative_scenario.py`, reset the
accumulation buffer when a new `{` starts while previous buffer is unresolvable. This recovers
from records with literal unescaped newlines in the prompt field (line 102 of question.jsonl).

## Outcome After Fixes
- `get_instances()` returns 250 instances ✓
- Run spec wiring (MetricSpec + AnnotatorSpec + annotators=annotators) is correct ✓
- Cannot verify `win_rate` locally: `OPENAI_API_KEY` not set → annotator cannot call gpt-4-turbo
- Cannot run inference locally: `google/gemini-2.5-flash-lite` requires API key config
- **Both fixes are correct; eval should pass on server with proper API keys**
