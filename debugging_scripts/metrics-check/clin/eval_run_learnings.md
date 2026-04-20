# Eval Run Learnings

## data_narrative — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: data_narrative passed on the first attempt with no issues; no special handling required.

## data_narrative — FAILED
- **Attempts**: 5
- **Root cause**: `BERTScorer` is initialized inside each parallel worker thread, causing a meta-tensor error (`Cannot copy out of meta tensor`) when the model is moved to a device in a multi-threaded context.
- **Fix applied**: Replaced manual `AutoModel` loading with `bert_score.BERTScorer`, but the lazy `_load_scorer()` call still happens per-thread, reproducing the same meta-tensor error.
- **Key learning**: `BERTScorer` (and any HuggingFace model) must be initialized once before parallelization and shared across threads, not lazily inside each worker; use a module-level or class-level singleton initialized in `__init__` to avoid the meta-tensor / thread-safety issue.

## data_narrative — SUCCESS
- **Attempts**: 2
- **Root cause**: Stale cached scenario instances referenced image files under `benchmark_output/scenarios/artinsight/images/` that no longer existed on disk, causing `cattrs` deserialization to fail with `AssertionError: Local file does not exist`.
- **Fix applied**: Cleared the stale scenario cache so HELM regenerated instances with valid (text-only) inputs matching the data_narrative scenario.
- **Key learning**: If `cattrs` raises `ClassValidationError` inside `structure_MediaObject` with a missing-file `AssertionError`, the cache is referencing a previous scenario's image paths — delete `benchmark_output/scenarios/<dataset>/` and rerun to regenerate clean instances.

## dialogue_diversity — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: dialogue_diversity passed on the first attempt; note that its metrics (semantic_diversity, distinct_1, distinct_2) are batch-only and require multiple responses per context — ensure the run spec requests multiple generations (num_outputs > 1) or the batch metrics will silently produce trivial scores.

## fann_or_flop — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: fann_or_flop passed on the first attempt with no issues; no special handling required.

## fann_or_flop — SUCCESS
- **Attempts**: 2
- **Root cause**: Stale `benchmark_output/scenarios/artinsight/` cache contained image-path references (bulldog.jpeg, mandala.jpeg, pink_flower.jpeg) that no longer existed, causing `cattrs` to fail during `MediaObject` deserialization.
- **Fix applied**: Cleared the stale scenario cache so HELM regenerated clean instances without dangling image paths.
- **Key learning**: The `artinsight` image-path cache pollution affects any dataset run after an image-based scenario (e.g. artinsight) shares the same benchmark_output directory — always delete `benchmark_output/scenarios/<dataset>/` before rerunning if you see `AssertionError: Local file does not exist at path: benchmark_output/scenarios/artinsight/images/*`.

## fig_qa — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: fig_qa passed on the first attempt with no issues; no special handling required.

## hummus — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: hummus passed on the first attempt with no issues; no special handling required.

## javanese_sundanese_story_cloze — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: javanese_sundanese_story_cloze passed on the first attempt with no issues; no special handling required.

## mars — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: mars passed on the first attempt with no issues; no special handling required.

## met_meme — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: met_meme passed on the first attempt with no issues; no special handling required.

## newyorker_caption — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: newyorker_caption passed on the first attempt; it is an image-to-text MC accuracy task — ensure the run spec uses low temperature and the scenario correctly loads the cartoon images as MediaObjects.

## pace — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: pace passed on the first attempt with no issues; no special handling required.

## protein_bench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: protein_bench passed on the first attempt with no issues; no special handling required.

## recombination_extraction — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: recombination_extraction passed on the first attempt with no issues; no special handling required.

## recombination_extraction — SUCCESS
- **Attempts**: 2
- **Root cause**: Stale `benchmark_output/scenarios/artinsight/` cache contained dangling image-path references (bulldog.jpeg, mandala.jpeg, pink_flower.jpeg) that no longer existed, causing `cattrs` `MediaObject` deserialization to fail with `AssertionError: Local file does not exist`.
- **Fix applied**: Cleared the stale scenario cache so HELM regenerated clean, image-free instances for recombination_extraction.
- **Key learning**: The artinsight image-path cache pollution is a recurring cross-dataset hazard — any dataset run after an image scenario that shares `benchmark_output/` will hit this; always purge `benchmark_output/scenarios/<dataset>/` before rerunning if you see missing-path errors under `benchmark_output/scenarios/artinsight/images/`.

## scar — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: scar passed on the first attempt with no issues; no special handling required.

## scar — SUCCESS
- **Attempts**: 2
- **Root cause**: Stale `benchmark_output/scenarios/artinsight/` cache contained dangling image-path references (bulldog.jpeg, mandala.jpeg, pink_flower.jpeg) that no longer existed, causing `cattrs` `MediaObject` deserialization to fail with `AssertionError: Local file does not exist`.
- **Fix applied**: Cleared the stale scenario cache so HELM regenerated clean, image-free instances for scar.
- **Key learning**: The artinsight image-path cache pollution is a persistent cross-dataset hazard — purge `benchmark_output/scenarios/<dataset>/` before rerunning whenever you see `AssertionError: Local file does not exist at path: benchmark_output/scenarios/artinsight/images/*`.

## dialogue_diversity — SUCCESS
- **Attempts**: 2
- **Root cause**: Stale `benchmark_output/scenarios/artinsight/` cache contained dangling image-path references (bulldog.jpeg, mandala.jpeg, pink_flower.jpeg) that no longer existed, causing `cattrs` `MediaObject` deserialization to fail with `AssertionError: Local file does not exist`.
- **Fix applied**: Cleared the stale scenario cache so HELM regenerated clean, text-only instances for dialogue_diversity.
- **Key learning**: The artinsight image-path cache pollution recurs across datasets — purge `benchmark_output/scenarios/<dataset>/` on any `MediaObject` deserialization failure; also note dialogue_diversity's batch-only metrics (semantic_diversity, distinct_1/2) require `num_outputs > 1` in the run spec or they produce trivial scores.
