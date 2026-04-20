# slang_generation — Metrics Diagnosis

## Expected Metrics (from registry)
| Metric | Type | In HELM | HELM Class |
|--------|------|---------|------------|
| llm_judge_creativity | llm_judge | false | null |
| llm_judge_relevance | llm_judge | false | null |

## Current Run Spec Metrics
- `MetricSpec(GenericLLMJudgeMetric, args={"metric_name": "llm_judge_creativity"})`
- `MetricSpec(GenericLLMJudgeMetric, args={"metric_name": "llm_judge_relevance"})`
- `AnnotatorSpec(GenericLLMJudgeAnnotator, ...)` for both metrics
- `annotators=annotators` is correctly wired (not None)

## Actual Stats.json Metrics (m2)
- `[]` (empty — 0 stats generated, 0 instances processed)

## Missing Metrics
Both `llm_judge_creativity` and `llm_judge_relevance` are missing because **0 instances** were produced by the scenario. With no instances, the annotator and metric are never called, so stats.json is empty.

## Root Cause
The scenario file `scenarios/slang_generation_scenario.py` reads `conv_slang.txt` line-by-line and calls `ast.literal_eval(line)` on each line. However, the file is a **single Python list expression** spanning 665 lines (e.g., `[('101', "a beginner's course."),` on line 1, individual tuple lines in the middle, `('zany', 'comical, wacky.')]` on the last line). None of the individual lines are valid standalone Python literals — they either start with `[`, have trailing commas, or contain no closing `]`. Every `ast.literal_eval(line)` raises `ValueError` or `SyntaxError` and is silently swallowed by the `except` clause, resulting in 0 instances loaded and 0 stats computed.

The run_spec itself is correctly configured (Pattern C: AnnotatorSpec + MetricSpec + rubrics + `annotators=annotators`). The bug is exclusively in the scenario's file parsing logic.

## Proposed Fix
In `scenarios/slang_generation_scenario.py`, replace the line-by-line parsing with a single `ast.literal_eval(f.read())` that parses the entire file as a Python list. Iterate over the resulting list of `(term, definition)` tuples.
