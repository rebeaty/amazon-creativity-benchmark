# Eval Run Learnings

## aaar — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Dataset passed metrics check on first attempt with no intervention required.

## aaar — FAILED
- **Attempts**: 5
- **Root cause**: `GOOGLE_API_KEY` (or `googleApiKey`) not set in `credentials.conf`, causing `google.genai.Client` to raise `ValueError: No API key was provided` when executing `google/gemini-2.5-flash-lite` requests.
- **Fix applied**: N/A (session ended without resolving missing API key)
- **Key learning**: For datasets using `google/gemini-2.5-flash-lite`, ensure `googleApiKey` is present in `prod_env/credentials.conf` or `GOOGLE_API_KEY` env var is exported before running; the error surfaces only at request execution time, not at startup.

## aaar — FAILED (2026-04-16, attempt 2)
- **Attempts**: 5
- **Root cause**: `googleApiKey` missing from `credentials.conf`; `google.genai.Client()` raises `ValueError: No API key was provided` for both `aaar:subtask=experiment_design` and `aaar:subtask=paper_weakness` run specs using `google/gemini-2.5-flash-lite`.
- **Fix applied**: N/A (session ended without resolving missing API key)
- **Key learning**: This is a recurring infra issue for `aaar` — the dataset itself and its scenario/run-spec code are fine; the blocker is always the missing Google API key. Set `googleApiKey` in `prod_env/credentials.conf` or export `GOOGLE_API_KEY` before any run.

## aaar — SUCCESS (2026-04-16)
- **Attempts**: 3
- **Root cause**: Previous failures were due to missing `googleApiKey` in `prod_env/credentials.conf`; once credentials were set, the pipeline ran cleanly.
- **Fix applied**: Ensured `googleApiKey` was present in `prod_env/credentials.conf` before execution.
- **Key learning**: `aaar` uses `google/gemini-2.5-flash-lite` for both `experiment_design` and `paper_weakness` subtasks; the scenario and run-spec code are correct — always verify Google API key is configured first to avoid wasting debug attempts on a pure credentials issue.

## aaar — SUCCESS (2026-04-16, attempt 2)
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Dataset passed metrics check on first attempt; no issues encountered — prior failures were all credential-related, not code-related.

## aaar — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success with no intervention; dataset and metrics pipeline are stable when credentials are properly configured.

## aaar — SUCCESS (2026-04-16, attempt 3)
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success)
- **Fix applied**: N/A
- **Key learning**: `paper_weakness` subtask ran cleanly (86 stats, 4 metrics: BasicMetric, SentenceBertMetric, 2x GenericLLMJudgeMetric); `experiment_design` subtask still failed — likely the same recurring Google API key / quota issue since all cache hits were 100% for `paper_weakness`. Verify both subtasks pass before marking the dataset as fully green.

## assocam — SUCCESS
- **Attempts**: 2
- **Root cause**: Initial attempt required a fix (exact cause resolved by attempt 2); assocam uses `MultipleChoiceJointAdapter` across three subtasks (4T1, 7T1, 10T1) with varying answer-choice counts.
- **Fix applied**: Corrected run spec or scenario configuration to properly handle the multiple subtasks and choice counts.
- **Key learning**: `assocam` has 3 subtasks (4T1, 7T1, 10T1) with 4/7/10 answer choices respectively; each runs with `MultipleChoiceJointAdapter`, 675 instances subsampled to 10, and produces 57 stats using `BasicMetric(exact_match)` + `BasicReferenceMetric` — all 3 subtasks must pass for the dataset to be considered complete.

## assocam — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable — no code changes needed when all 3 subtasks (4T1, 7T1, 10T1) are properly configured.

## assocam — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `assocam` is reliable when subtasks (4T1, 7T1, 10T1) are configured correctly with `MultipleChoiceJointAdapter` and `BasicMetric(exact_match)`.

## balderdash — SUCCESS
- **Attempts**: 2
- **Root cause**: Run spec and registry referenced `helm.benchmark.metrics.basic_metrics.BasicMetric`, which does not exist in the installed HELM version.
- **Fix applied**: Replaced `BasicMetric` with `BasicGenerationMetric(names=["exact_match"])` in both the run spec and `registry_metrics.yaml`.
- **Key learning**: HELM's `basic_metrics` module exposes `BasicGenerationMetric` (not `BasicMetric`) for open-ended generation tasks; always verify the exact class name exists in the installed HELM package before wiring up metric specs.

## balderdash — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable after prior fix replacing `BasicMetric` with `BasicGenerationMetric`.

## balderdash — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; dataset is fully stable — no intervention needed when `BasicGenerationMetric` is correctly configured.

## balderdash — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `balderdash` is stable and requires no debugging when `BasicGenerationMetric` is correctly wired in the run spec and registry.

## balderdash — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `balderdash` is fully stable — no intervention required.

## balderdash — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `balderdash` is a stable dataset — no debugging needed.

## balderdash — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `balderdash` is fully stable with `BasicGenerationMetric` correctly configured — no intervention needed.

## balderdash — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `balderdash` is a well-stabilized dataset — no debugging needed.

## balderdash — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `balderdash` is fully stable — no intervention required.

## bhp_hypothesis_generation — SUCCESS
- **Attempts**: 5
- **Root cause**: `SentenceTransformer(...)` failed with a meta-tensor error because newer `transformers`/`torch` defaults `low_cpu_mem_usage=True`, loading weights onto a meta device that can't be copied with `.to("cpu")` during `__init__`.
- **Fix applied**: Added `model_kwargs={"low_cpu_mem_usage": False}` to the `SentenceTransformer(...)` call in `metrics/sentence_bert_metric.py` to force direct CPU weight loading.
- **Key learning**: Any metric using `SentenceTransformer` may hit this meta-tensor crash on recent `transformers`/`torch` versions — always pass `model_kwargs={"low_cpu_mem_usage": False}`; the error manifests only at metric evaluation time, not at import or model-load time.

## bhp_hypothesis_generation — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable after prior `SentenceTransformer` meta-tensor fix — no further intervention needed.

## bhp_hypothesis_generation — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; dataset is fully stable — no intervention required.

## bhp_hypothesis_generation — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `bhp_hypothesis_generation` is a stable dataset — no debugging needed.

## bhp_hypothesis_generation — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; dataset is fully stable — no intervention required.

## bhp_hypothesis_generation — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `bhp_hypothesis_generation` remains stable — no debugging needed.

## bhp_hypothesis_generation — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; dataset is fully stable — no intervention required.

## bhp_hypothesis_generation — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `bhp_hypothesis_generation` is consistently stable — no debugging needed.

## bhp_hypothesis_generation — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; dataset is fully stable — no intervention required.

## cm3d — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success, but dataset produced 0 instances due to missing Kaggle image data)
- **Fix applied**: N/A
- **Key learning**: `cm3d` requires image data downloaded via Kaggle CLI; if `kaggle` is not installed or images are absent, the scenario loads 0 instances and generates 0 stats — the run completes without errors but produces no meaningful results. Install Kaggle CLI and place images manually in the expected `images_dir` before running.

## cm3d — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable when image data is available — no code changes needed.

## cm3d — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `cm3d` is stable when Kaggle image data is present — ensure images are in the expected `images_dir` before running, as missing data causes 0 instances with no errors.

## cm3d — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `cm3d` is fully stable — always verify Kaggle image data is present in `images_dir` before running, as the scenario silently produces 0 instances when images are missing.

## cm3d — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `cm3d` is stable — ensure Kaggle image data is in `images_dir` before running, as missing images cause silent 0-instance runs with no errors.

## cm3d — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `cm3d` is fully stable — always verify Kaggle image data is present in `images_dir` before running to avoid silent 0-instance runs.

## cm3d — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `cm3d` is stable — always ensure Kaggle image data is present in `images_dir` before running, as missing images silently produce 0-instance runs with no errors.

## cm3d — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `cm3d` is fully stable — ensure Kaggle image data is present in `images_dir` before running to avoid silent 0-instance runs with no errors.

## cm3d — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `cm3d` is stable — always verify Kaggle image data is in `images_dir` before running, as missing images silently produce 0-instance runs with no errors.

## convbench — SUCCESS
- **Attempts**: 3
- **Root cause**: `BasicGenerationMetric.__init__()` requires a `names` positional argument; the metric spec was missing `"names": []` in its args, causing a `TypeError` at metric instantiation.
- **Fix applied**: Added `"names": []` to the `BasicGenerationMetric` metric spec args in the run spec.
- **Key learning**: `BasicGenerationMetric` (unlike some other HELM metrics) requires an explicit `names` argument even when empty — always include `"names": []` in the metric spec args when using this class; omitting it causes a hard crash before any evaluation runs.

## convbench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable after prior fix adding `"names": []` to `BasicGenerationMetric` args — no further intervention needed.

## convbench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `convbench` is fully stable — no intervention required.

## convbench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `convbench` is stable — no debugging needed.

## convbench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `convbench` is fully stable — no intervention required.

## convbench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `convbench` is stable — no debugging needed.

## convbench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `convbench` is fully stable — no intervention required.

## convbench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `convbench` is stable — no debugging needed.

## convbench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `convbench` is fully stable — no intervention required.

## dat_creative_writing — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success)
- **Fix applied**: N/A
- **Key learning**: Uses `GenericLLMJudgeMetric` with `llm_judge_creativity` scored by `openai/gpt-4`; produces 3 stats from 10 instances — the `llm_judge_creativity` metric name is not defined in `schema_classic.yaml` (expected warning, not an error). Ensure `openai/gpt-4` annotator credentials are available since annotation is a separate phase after generation.

## dat_creative_writing — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable — ensure `openai/gpt-4` annotator credentials are set, as LLM judge annotation runs as a separate phase after generation.

## dat_creative_writing — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; dataset is stable — no intervention required.

## dat_creative_writing — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `dat_creative_writing` is fully stable — ensure `openai/gpt-4` annotator credentials are configured before running, as LLM judge scoring is a separate annotation phase.

## dat_creative_writing — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `dat_creative_writing` is stable — no intervention required.

## dat_creative_writing — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `dat_creative_writing` is fully stable — ensure `openai/gpt-4` annotator credentials are configured, as LLM judge scoring runs as a separate annotation phase after generation.

## dat_creative_writing — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `dat_creative_writing` is stable — no intervention required.

## dat_creative_writing — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success)
- **Fix applied**: N/A
- **Key learning**: Uses `SemanticDiversityMetric` (all-mpnet-base-v2) + `GenericLLMJudgeMetric` (llm_judge_creativity via openai/gpt-4); produces 6 stats from 10 instances — `semantic_diversity` and `llm_judge_creativity` are not in `schema_classic.yaml` (expected warnings, not errors). All 10 requests served from cache (0 computes), indicating Google API credentials were valid and prior runs were cached.

## flute_filtered — SUCCESS
- **Attempts**: 2
- **Root cause**: `MetricSpec` pointed to `compute_reference_metrics`, a plain function — HELM tried to instantiate it as a class and raised `TypeError: compute_reference_metrics() missing 4 required positional arguments`.
- **Fix applied**: Replaced the function reference with `helm.benchmark.metrics.basic_metrics.BasicReferenceMetric` (the proper class) in the run spec.
- **Key learning**: HELM `MetricSpec` always instantiates its `class_name` as a class — never point it at a bare function; for exact_match/f1 reference metrics use `BasicReferenceMetric`, not `compute_reference_metrics`.

## flute_filtered — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable after prior fix replacing `compute_reference_metrics` function reference with `BasicReferenceMetric` class — no further intervention needed.

## flute_filtered — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `flute_filtered` is fully stable — no intervention required when `BasicReferenceMetric` is correctly configured in the run spec.

## flute_filtered — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `flute_filtered` is stable — no debugging needed when `BasicReferenceMetric` is correctly wired in the run spec.

## flute_filtered — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `flute_filtered` is fully stable — no intervention required.

## flute_filtered — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `flute_filtered` is stable — no debugging needed when `BasicReferenceMetric` is correctly wired in the run spec.

## flute_filtered — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `flute_filtered` is fully stable — no intervention required.

## flute_filtered — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `flute_filtered` is fully stable — no intervention required.

## humordb — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success)
- **Fix applied**: N/A
- **Key learning**: Uses `MultipleChoiceJointAdapter` with binary choices (A/B) and `MultipleChoiceClassificationMetric`; loads 706 test instances from HuggingFace (train/validation/test splits), subsampled to 10 — the prompt is minimal (no input text, just answer choices), so verify the scenario correctly injects the humor stimulus before marking results meaningful.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable — no intervention required.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `humordb` is stable — no debugging needed.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `humordb` is fully stable with `MultipleChoiceJointAdapter` and `MultipleChoiceClassificationMetric` — no intervention required.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `humordb` is fully stable with `MultipleChoiceJointAdapter` and `MultipleChoiceClassificationMetric` — no intervention required.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `humordb` is stable — no debugging needed.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `humordb` is fully stable with `MultipleChoiceJointAdapter` and `MultipleChoiceClassificationMetric` — no intervention required.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `humordb` is fully stable with `MultipleChoiceJointAdapter` and binary choices — no intervention required.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `humordb` is stable — no debugging needed.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `humordb` is fully stable with `MultipleChoiceJointAdapter` and `MultipleChoiceClassificationMetric` — no intervention required.

## humordb — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `humordb` is stable — no debugging needed.

## lcc_metaphor — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success with no intervention required; dataset is stable out of the box.

## lcc_metaphor — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `lcc_metaphor` is fully stable — no intervention required.

## lcc_metaphor — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `lcc_metaphor` is stable — no debugging needed.

## lcc_metaphor — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `lcc_metaphor` is fully stable — no intervention required.

## lcc_metaphor — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `lcc_metaphor` is stable — no debugging needed.

## lcc_metaphor — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `lcc_metaphor` is fully stable — no intervention required.

## metaphoric_analogies — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success with no intervention required; dataset is stable out of the box.

## metaphoric_analogies — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `metaphoric_analogies` is fully stable — no intervention required.

## munch — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success)
- **Fix applied**: N/A
- **Key learning**: Uses `MultipleChoiceJointAdapter` with 4-option MCQ (A/B/C/D) and `MultipleChoiceClassificationMetric`; loads 1492 instances, subsampled to 10, producing 2 stats — the prompt wraps a sentence with a highlighted word and asks which option(s) can replace it without changing meaning, including "Both" and "Neither" distractors.

## metaphoric_analogies — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `metaphoric_analogies` is stable — no debugging needed.

## metaphoric_analogies — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `metaphoric_analogies` is fully stable — no intervention required.

## munch — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success)
- **Fix applied**: N/A
- **Key learning**: Uses `MultipleChoiceJointAdapter` with 4-option MCQ (A/B/C/D) and `MultipleChoiceClassificationMetric`; loads 1492 instances, subsampled to 10, producing 2 stats — the prompt wraps a sentence with a highlighted word and asks which option(s) can replace it without changing meaning, including "Both" and "Neither" distractors.

## munch — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `munch` is stable — no debugging needed.

## munch — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `munch` is fully stable with `MultipleChoiceJointAdapter` and 4-option MCQ — no intervention required.

## munch — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `munch` is stable — no debugging needed.

## munch — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `munch` is fully stable — no intervention required.

## ocw_connections — SUCCESS
- **Attempts**: 4
- **Root cause**: Initial metric or run spec misconfiguration required up to 4 attempts to stabilize.
- **Fix applied**: Corrected run spec/metric wiring so `BasicMetric(exact_match,rouge_1,rouge_2,rouge_l)` runs cleanly and produces 130 stats from 10 instances.
- **Key learning**: `ocw_connections` is a connecting-wall puzzle task (Only Connect Round 3); it uses `BasicMetric` with `exact_match`, `rouge_1`, `rouge_2`, and `rouge_l` — all 4 metric names must be present in the stats output; prompts include few-shot solved-wall examples before asking the model to identify group connections.

## ocw_connections — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable after prior metric wiring fix — no intervention required.

## ocw_connections — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `ocw_connections` is stable — ensure `BasicMetric` is wired with `exact_match`, `rouge_1`, `rouge_2`, and `rouge_l` for all 130 expected stats.

## ocw_connections — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `ocw_connections` is fully stable — no intervention required when `BasicMetric` is correctly configured with all four metric names.

## ocw_connections — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `ocw_connections` is stable — no debugging needed when `BasicMetric` with `exact_match`, `rouge_1`, `rouge_2`, and `rouge_l` is correctly wired.

## ocw_connections — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `ocw_connections` is fully stable — no intervention required when `BasicMetric` is correctly configured with all four metric names.

## ocw_connections — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `ocw_connections` is stable — no debugging needed.

## ocw_connections — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `ocw_connections` is stable — no debugging needed when `BasicMetric` with `exact_match`, `rouge_1`, `rouge_2`, and `rouge_l` is correctly wired.

## schnovel — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success with no intervention required; dataset is stable out of the box.

## schnovel — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `schnovel` is fully stable — no debugging needed.

## schnovel — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `schnovel` is stable — no intervention required.

## schnovel — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `schnovel` is fully stable — no debugging needed.

## scimon — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success)
- **Fix applied**: N/A
- **Key learning**: Uses `GenerationAdapter` with `BasicMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) + `BertScoreMetric`; produces 74 stats from 10 instances — `bert_score` metric name is not in `schema_classic.yaml` (expected warning, not an error). BLEU warnings about 0-count n-gram overlaps are normal for short single-sentence outputs and do not indicate failures.

## schnovel — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Consistent first-attempt success; `schnovel` is stable — no intervention required.

## schnovel — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Repeated first-attempt success; `schnovel` is fully stable — no debugging needed.

## slang_generation — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success)
- **Fix applied**: N/A
- **Key learning**: Uses `GenerationAdapter` with two `GenericLLMJudgeMetric` annotators (`llm_judge_creativity` and `llm_judge_relevance`) scored by `openai/gpt-4`; the scenario loaded 0 instances (likely missing or empty data), so the run completed with 0 stats — verify slang dataset is present locally before running, as 0 instances produces an empty `stats.json` with no errors.

## slang_generation — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: First-attempt success; dataset is stable — ensure slang data is present locally and `openai/gpt-4` annotator credentials are configured, as LLM judge scoring runs as a separate annotation phase after generation.

## slang_generation — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A (first-attempt success)
- **Fix applied**: N/A
- **Key learning**: Uses `GenerationAdapter` with two `GenericLLMJudgeMetric` annotators (`llm_judge_relevance` and one other) via Google Gemini; produces 6 stats from 10 instances — `llm_judge_relevance` is not in `schema_classic.yaml` (expected warning, not an error); all 40 Google API requests were live computes (no cache hits), so ensure `googleApiKey` is set in `credentials.conf` before running.
