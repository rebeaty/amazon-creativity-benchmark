# RunSpec Generation Report

Generated: 2026-03-25

## Summary

- Total datasets: 159
- Run spec files generated: 158 (all 159 datasets covered; aaar has 2 subtasks in 1 file)
- Total `@run_spec_function` entries: 161
- Datasets with missing scenario files: 0 (all 159 have scenario files)
- Datasets with LLM-judge metrics: 62
- Multi-subtask datasets: 3 (aaar, assocam — but single file each)
- Rubrics generated: 62 datasets × ~1.8 avg = ~113 rubric entries

## Multi-Subtask Datasets

| Dataset | Subtasks | Functions in file |
|---------|----------|------------------|
| aaar | experiment_design, paper_weakness | 2 |
| assocam | 4T1, 7T1, 10T1 | 3 |

Note: All other datasets with `subtask` / `subset` / `domain` init params default to a single run spec using `args={}`. Per instructions, subtask breakdown is only done when `SUBTASKS = [...]` is explicitly defined in the scenario.

## LLM-Judge Datasets (62 total)

aaar, aidanbench, alpaca_eval_2, arena_hard_creative, arena_hard_v01, artinsight,
arastories, banner_request_400, calligrapher, conceptual_design, cpers, creai_cps,
creation_mmbench, creative_process, creativemath, creatset, critics_story, crowd_vote,
cs4, dat_creative_writing, dialogue_diversity, discovery_bench, dpt,
eqbench_creative_writing_v3, fscg8, future_ideas, fuxibench, gauss, geo_story,
grapheval_ai_researcher, grapheval_iclr, grapheval_review_advisor, historical_analogy,
hummus, hypobench, hypogen, layoutsam_eval, litbench, liveideabench, llm_review_focus,
llm_srbench, macgyver, matdesign, mineanybuild, mops, permpst, poetmt, pollux_creativity,
pron_vs_prompt, pun_eval, rebus_puzzle, research_idea_execution, rpgbench, showerthoughts,
slang_generation, speak_to_structure, ss_gen, tinyfabulist, tinystories, webnovelbench,
writingbench

## Custom Formula Metrics Used (non-HELM, non-LLM-judge)

| Metric class | Datasets |
|---|---|
| `metrics.jsd_metric.JSDMetric` | amuse_chord_generation |
| `metrics.distinct_n_metric.DistinctNMetric` | dialogue_diversity, diverse_not_short, llm_discussion |
| `metrics.creativity_score_metric.CreativityScoreMetric` | cdat |
| `metrics.mean_absolute_error_metric.MeanAbsoluteErrorMetric` | ava |
| `metrics.correlation_metric.CorrelationMetric` | ava, muse_perception, story_quality, storyer |
| `metrics.group_match_score_metric.GroupMatchScoreMetric` | nyt_connections, ocw |
| `metrics.type_token_ratio_metric.TypeTokenRatioMetric` | pace |
| `metrics.validity_score_metric.ValidityScoreMetric` | mineanybuild |
| `metrics.validity_metric.ValidityMetric` | protein_bench, speak_to_structure |
| `metrics.pass_at_1_metric.PassAt1Metric` | neocoder |
| `metrics.constraint_satisfaction_metric.ConstraintSatisfactionMetric` | neocoder |
| `metrics.xml_validity_metric.XmlValidityMetric` | robotoolbench |
| `metrics.array_dimensions_metric.ArrayDimensionsMetric` | robotoolbench |
| `metrics.json_validity_metric.JsonValidityMetric` | rpgbench |
| `metrics.vietnamese_poem_metric.VietnamesePoemMetric` | vietnamese_poem |
| `metrics.percentile_rank_metric.PercentileRankMetric` | webnovelbench |
| `metrics.iou_score_metric.IoUScoreMetric` | textlogo3k |
| `metrics.layout_quality_metric.LayoutQualityMetric` | textlogo3k |

## Model-Based Metrics Skipped (no helm_class)

The following datasets have `model_based` metrics with `helm_class: null` — these
require external models and cannot be wired as standard MetricSpec. A `# TODO` comment
is generated in those metric positions.

| Dataset | Metric | Required Model |
|---|---|---|
| cdat | appropriateness | crawl-300d-2M-subword (FastText) |
| cdat | novelty | crawl-300d-2M-subword (FastText) |
| dat | semantic_diversity | all-mpnet-base-v2 (SentenceTransformers) |
| dat_creative_writing | semantic_diversity | all-mpnet-base-v2 (SentenceTransformers) |
| dialogue_diversity | semantic_diversity | all-mpnet-base-v2 (SentenceTransformers) |
| noveltybench | diversity_score | deberta-v3-large (fine-tuned) |
| noveltybench | quality_score | Skywork/Skywork-Reward-Gemma-2-27B-v0.2 |
| pace | association_distance | GloVe 6B 300d |
| protein_bench | plddt_score | ESMFold |
| protein_bench | sctm_score | ESMFold |
| protein_bench | novelty_tmscore | ESMFold + PDB lookup |
| sdat | semantic_diversity_score | ibm-granite/granite-embedding-278m-multilingual |
| vflute | f1_at_bertscore_threshold | bert-base-uncased + BLEURT |
| aidanbench | novelty_embedding_similarity | text-embedding-ada-002 |

## Inference Configuration Notes

| Dataset | Temperature | max_tokens | num_outputs | Notes |
|---|---|---|---|---|
| cpers | 1.0 | 512 | 1 | High temperature for creative generation |
| eqbench_creative_writing_v3 | 0.7 | 2048 | 3 | 3 outputs per prompt, longer context |
| aidanbench | 0.7 | 512 | 30 | 30 responses for diversity metrics |
| amuse_chord_generation | 0.7 | 512 | 30 | 30 chord progressions per keyword |
| noveltybench | 0.7 | 512 | 10 | 10 outputs for novelty measurement |
| writingbench | 0.7 | 16000 | 1 | Very long context, top_p=0.8, top_k=20 |
| All others | 0.7 | 512 | 1 | Default settings |

## Rubric Sources

- **[PAPER]**: 6 rubrics extracted from paper/documentation (rpgbench interestingness, tinyfabulist grammar+creativity, aidanbench coherence_score)
- **[GENERATED]**: ~107 rubrics auto-generated based on dataset task description and metric name

## Assumptions Made

1. **MCQA detection**: Datasets with multiple `Reference` objects where one gets `CORRECT_TAG` are classified as MCQA using `ADAPT_MULTIPLE_CHOICE_JOINT`. Datasets with only open-ended `CORRECT_TAG` references use `ADAPT_GENERATION`.

2. **stop_sequences**: Set to `["\n"]` for MCQA datasets, `[]` for all generation tasks.

3. **max_train_instances**: Set to 5 for datasets that explicitly load `TRAIN_SPLIT`, 0 for all others (zero-shot).

4. **assocam subtasks**: The 3 subtasks (4T1, 7T1, 10T1) are MCQA-style with `CORRECT_TAG`.

5. **graphragbench-wrongone**: Mapped to file `graphragbench_wrongone_run_specs.py` with function name `get_graphragbench_wrongone_spec` and `run_spec_function("graphragbench_wrongone")`.

6. **Deduplication**: Multiple metrics pointing to the same helm_class (e.g., sentence_bert_f1/precision/recall all → one SummarizationMetric; multiple compute_reference_metrics → one MetricSpec) are deduplicated.

7. **fuxibench**: Single run spec with `args={}` — the scenario has 5 subtasks (ci_gen, couplet_gen, poem_gen, poem_nmt_inv, poem_appre) but no registry-level subtask split.

## TODOs / Flags

1. Model-based metrics with no `helm_class` (listed above) require custom metric implementations before evaluation can proceed.

2. LLM judge rubrics tagged `[GENERATED]` should be reviewed by domain experts before use.

3. `historical_analogy` judge_score_analogy uses a 1-4 scale (not 1-5); rubric written accordingly.

4. `tinyfabulist` and `aidanbench` have paper-defined rubrics embedded in `registry_metrics.yaml` judge_prompt field — these are preserved in `registry_rubrics.yaml`.

5. `eqbench_creative_writing_v3` uses `anthropic/claude-sonnet-4` as judge — ensure API access.

6. `webnovelbench` uses `deepseek/deepseek-v3` as judge — ensure API access.
