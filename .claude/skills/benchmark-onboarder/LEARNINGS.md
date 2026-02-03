# Benchmark Onboarding Learnings

Add issues and patterns here as you discover them. Everyone on the team benefits.

## Dataset Quirks

| Benchmark | Issue | Solution |
|-----------|-------|----------|
| RiddleSense | Test split has no labels (empty `answerKey`) | Use validation split instead |
| HumorDB | Multimodal (images + text) | Include text component; note `multimodal` in tags |
| ANALOBENCH | Field is `Sentence` not `Story` | Check actual dataset keys before coding |
| BRAINTEASER | No official HF dataset; many unofficial versions | Use `tasksource/brainteasers` with SP/WP configs; use `choice_order` and `label` fields for correct shuffling |
| HUMMUS | Multimodal benchmark (New Yorker cartoons + captions) | Use text-only ablation: `image_descriptions.csv` provides scene descriptions. Data in GitHub repo, not HuggingFace. |
| HUMMUS | Label "WIDLII" means "While In Doubt, Leave It In" | Treat as positive class (metaphorical) along with "Yes"; "Discard" items excluded from test_set.json |
| HUMMUS | Paper has 6 tasks; 4 are text-compatible | Tasks 3-4 (ImageBbox, ImageLabel) require visual grounding. Implemented: classification (940), naming (568), caption_highlight (568), explanation (568) |
| MACGYVER | Data in xlsx format on GitHub, not HuggingFace | Download xlsx via `?raw=true` URL suffix; load with pandas. File: `data/MacGyver/problem_solution_pair.xlsx` |
| Meta4XNLI | Multiple configs with language-specific splits | Use `int_eval` config with `xnli_test_met` split; filter by `language` field for EN/ES. Detection uses `det_en_finetune` config. |
| Meta4XNLI | Exact prompts in Appendix Table 29 | Zero-shot: "Say which is the inference relationship...answer only with one word between 'entailment', 'neutral' or 'contradiction'." Format: `{Premise} -> {Hypothesis}:` |
| Simile Generation | Data in CSV on GitHub, not HuggingFace | Download SimileEMNLP.csv from repo. Input: literal sentence, Output: simile. Human1/Human2 columns are references. |
| MACGYVER | Includes both solvable and unsolvable problems | For unsolvable problems (377 of 1683), expected response is to identify infeasibility. Include all in eval. |
| MACGYVER | Human-annotated evaluation | Paper uses fine-grained categories (efficient, inefficient, infeasible, etc.). See `annotator_notes.md` for judge setup. |

## Common Patterns

- **Datasets requiring `trust_remote_code`**: Add `trust_remote_code=True` to `load_dataset()`
- **Suspected model output fields**: Check the paper to confirm field purpose before skipping

## Style Conventions

- Scenario class names: `{Benchmark}Scenario` (e.g., `RiddlesenseScenario`)
- `name` field: lowercase with underscores (e.g., `riddlesense`)
- `description` field: data source reference, not task description
- Always include `tags = ["creativity", ...]`

## Evaluation Types Encountered

| Benchmark | Eval Type | HELM Pattern | Notes |
|-----------|-----------|--------------|-------|
| HUMMUS (classification) | exact_match | `get_exact_match_metric_specs()` | Binary Yes/No metaphor detection |
| HUMMUS (naming) | open_ended | `get_open_ended_generation_metric_specs()` | Conceptual metaphor ID; paper uses sentence similarity (LaBSE) |
| HUMMUS (caption_highlight) | exact_match | `get_exact_match_metric_specs()` | Tag metaphor text; paper uses Jaccard index |
| HUMMUS (explanation) | open_ended | `get_open_ended_generation_metric_specs()` | <=30 word explanation; paper uses ROUGE |
| MACGYVER (all) | open_ended | `get_open_ended_generation_metric_specs()` | Creative problem-solving; paper uses human annotation with fine-grained categories |
| Meta4XNLI (interpretation) | exact_match | `get_exact_match_metric_specs()` | 3-way NLI classification (entailment/contradiction/neutral) on metaphorical sentences |
| Meta4XNLI (detection) | open_ended | `get_open_ended_generation_metric_specs()` | Token-level metaphor identification; paper uses sequence labeling F1 |

**HELM RunSpec patterns:**
- `exact_match` → `get_exact_match_metric_specs()`
- `open_ended` → `get_open_ended_generation_metric_specs()` (BLEU, ROUGE, F1)
- `summarization` → `get_summarization_metric_specs()`
- `llm_judge` → Custom Annotator required
- `custom` → New metric implementation needed

## LLM-as-Judge Benchmarks

| Benchmark | Judge Model | Rubric Location | Dimensions | Annotator Notes |
|-----------|-------------|-----------------|------------|-----------------|
| MACGYVER | GPT-4 (paper) | Paper Section 4.2, benchmark_results.json | correctness, feasibility, efficiency | `scenarios/macgyver/annotator_notes.md` |

**Workflow:**
1. Create Scenario as normal (Scenario stays pure—no eval info)
2. Extract judge config per Step 3b → `scenarios/benchmark_name/annotator_notes.md`
3. Set `eval_type: llm_judge` in benchmarks.json
4. Common dimensions: novelty, usefulness, fluency, coherence, surprise

## Benchmarks Reviewed but Not Onboarded

| Benchmark | Reason | Notes |
|-----------|--------|-------|
| Open-ended Data-Driven Discovery (DiscoveryBench) | Scientific reasoning, not creativity | Task is discovering patterns in tabular data; evaluation is hypothesis correctness, not creative merit |
