# HELM Evaluation Pipeline - Debugging Context

## Project Structure
- **Evaluation bash scripts**: `eval_scripts/`
- **Scenario files**: `scenarios/`
- **Run specifications**: `run_specs/`
- **Dataset-to-metric map**: `data/registry/registry_metrics.yaml`
- **Dataset-to-inference config map**: `data/registry/registry_inference.yaml`
- **Dataset-to-master info (links, modality, etc.)**: `data/registry/registry_master.yaml`
- **Dataset list**: `scenarios/subsampled_list.json` ← ONLY use this list

## Debugging Protocol for Each Dataset

When debugging a dataset, follow this exact checklist IN ORDER:

1. **Data availability**: Check if evaluation data exists locally. If not, download it.
2. **Prompt sanity check**: Verify the input query (test instance wrapped in prompt template) makes sense. Cross-reference with any examples from the paper (find paper link in `data/registry/registry_master.yaml`).
3. **Generation config check**: Verify inference config is logically sound:
   - MCQ/classification → low temperature (0.0–0.3)
   - Open-ended generation → higher temperature (0.5–1.0)
   - Check max tokens, stop sequences, etc.
4. **Raw generation check**: Run a small sample and verify model outputs look reasonable.
5. **Metric selection check**: Verify the correct metric is mapped for this dataset in `registry_metrics.yaml`.
6. **Evaluation execution**: Run the evaluation metric on the generated responses. Debug any errors.
7. **Aggregation check**: Verify metric aggregation is computed correctly.
8. **Results saving**: Verify results are saved correctly to the expected output location.

## Rules
- Maximum 10 debugging attempts per dataset before giving up.
- On success: save results and mark dataset as PASSED.
- On failure after 10 attempts: log the last error encountered and mark as FAILED.
- Always read `scenarios/subsampled_list.json` for the dataset list — no other source.
- When fixing bugs, ONLY fix issues specific to the current dataset — do not make breaking changes to shared code without noting it.

## Model Configuration
- The evaluation model may be an API model (e.g. `claude-haiku-4-5-20251001`, `gpt-4o`) or a local HuggingFace model.
- For API models: verify that `data/registry/registry_inference.yaml` uses the correct API backend, endpoint URL, and model name. Ensure the appropriate API key env var is set (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
- For HuggingFace models: verify the model path is correct and the model can be loaded.
- The model to evaluate will be specified in the prompt. If not specified, use whatever is already configured in the inference registry.