# Per-Metric Breakdown

## Formula-Based Metrics

| Metric Name | Metric Type | HELM Class | Datasets Using This Metric |
|-------------|-------------|------------|---------------------------|
| exact_match | formula_based | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics | arn, analobench, brainteaser, chinese_homophonic_puns, crowdcounter, csd100, d_humor, deep_math, fig_qa, flute_filtered, funqa, graphrag_bench, graphragbench-wrongone, idrbench, javanese_sundanese_story_cloze, metaphoric_analogies, miqa, music_theory_bench, nyt_connections, ocw, ocw_connections, puzzleworld, recombination_extraction, rebus_puzzle, riddlesense, scar, sudoku_bench, ttcw, v_flute, vflute |
| accuracy | formula_based | helm.benchmark.metrics.basic_metrics.BasicMetric | assocam, balderdash, cii_bench, cm3d, convbench, creative_pair, d_humor, fann_or_flop, fig_qa, humordb, ii_bench, irfl, javanese_sundanese_story_cloze, kiva, lcc_metaphor, mars, met_meme, meta4xnli, miqa, moh_x, munch, muse_perception, music_theory_bench, newyorker_caption, newyorker_humor, oogiri_go, proparalogy, puntuguese, riddlesense, schnovel, sonnet_or_not_bot, tiger_bench, ttcw, yesbut_v2 |
| bleu_4 | formula_based | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics | analobench, balancecc_prompt_generation, bhp_hypothesis_generation, c3_crosstalk, chinese_homophonic_puns, clef_joker_2025_task2, cue_word_story, data_narrative, futuregen, gauss, humor_transfer, llm4biohypogen, memecap, metaphor_generation, mops, ocw_connections, outline_to_story, poetmt, pun_eval, science_analogies, scimon, simile_generation, splat, ss_gen, story_generation_rocstories, thenextchapter, twistlist, unfun_corpus, vgsg, yesbut, yesbut_v2 |
| bleu_1 | formula_based | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics | clef_joker_2025_task2 |
| rouge_l | formula_based | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics | analobench, assocam, balancecc_prompt_generation, bhp_hypothesis_generation, c3_crosstalk, csd100, cue_word_story, data_narrative, esp_dataset, futuregen, funqa, gauss, geo_story, graphrag_bench, humor_transfer, infochartqa, llm4biohypogen, macgyver, memecap, metaphor_generation, mixassist, mops, newyorker_humor, ocw_connections, outline_to_story, permpst, poetmt, pun_eval, science_analogies, scimon, simile_generation, splat, ss_gen, story_generation_rocstories, thenextchapter, twistlist, unfun_corpus, vgsg, yesbut, yesbut_v2 |
| rouge_1 | formula_based | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics | ocw_connections |
| f1 | formula_based | helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics | fann_or_flop, flute_filtered, lcc_metaphor, memecap, metaphor_generation, metaphoric_analogies, moh_x, outline_to_story, puntuguese, recombination_extraction, scar, ttcw, yesbut |
| self_bleu | formula_based | helm.benchmark.metrics.disinformation_metrics.DisinformationMetric | aidanbench, amuse_chord_generation, llm_discussion |
| distinct_1 | formula_based | metrics.distinct_n_metric.DistinctNMetric | dialogue_diversity, diverse_not_short, llm_discussion |
| distinct_2 | formula_based | metrics.distinct_n_metric.DistinctNMetric | dialogue_diversity, diverse_not_short, llm_discussion |
| jensen_shannon_divergence_unigram | formula_based | metrics.jsd_metric.JSDMetric | amuse_chord_generation |
| jensen_shannon_divergence_bigram | formula_based | metrics.jsd_metric.JSDMetric | amuse_chord_generation |
| mean_absolute_error | formula_based | metrics.mean_absolute_error_metric.MeanAbsoluteErrorMetric | ava |
| pearson_correlation | formula_based | metrics.correlation_metric.CorrelationMetric | ava, muse_perception, story_quality |
| spearman_correlation | formula_based | metrics.correlation_metric.CorrelationMetric | ava, story_quality, storyer |
| group_match_score | formula_based | metrics.group_match_score_metric.GroupMatchScoreMetric | nyt_connections, ocw |
| classification_accuracy | formula_based | metrics.classification_accuracy_metric.ClassificationAccuracyMetric | recombination_extraction |
| validity | formula_based | metrics.validity_metric.ValidityMetric | neocoder, speak_to_structure |
| constraint_satisfaction | formula_based | metrics.constraint_satisfaction_metric.ConstraintSatisfactionMetric | neocoder |
| pass_at_1 | formula_based | metrics.pass_at_1_metric.PassAt1Metric | neocoder |
| xml_validity | formula_based | metrics.xml_validity_metric.XmlValidityMetric | robotoolbench |
| array_dimensions | formula_based | metrics.array_dimensions_metric.ArrayDimensionsMetric | robotoolbench |
| json_validity | formula_based | metrics.json_validity_metric.JsonValidityMetric | rpgbench |
| creativity_score | formula_based | metrics.creativity_score_metric.CreativityScoreMetric | cdat |
| iou_score | formula_based | metrics.iou_score_metric.IoUScoreMetric | textlogo3k |
| layout_quality | formula_based | metrics.layout_quality_metric.LayoutQualityMetric | textlogo3k |
| type_token_ratio | formula_based | metrics.type_token_ratio_metric.TypeTokenRatioMetric | pace |
| poem_score | formula_based | metrics.vietnamese_poem_metric.VietnamesePoemMetric | vietnamese_poem |
| length_score | formula_based | metrics.vietnamese_poem_metric.VietnamesePoemMetric | vietnamese_poem |
| tone_score | formula_based | metrics.vietnamese_poem_metric.VietnamesePoemMetric | vietnamese_poem |
| rhyme_score | formula_based | metrics.vietnamese_poem_metric.VietnamesePoemMetric | vietnamese_poem |
| validity_score | formula_based | metrics.validity_score_metric.ValidityScoreMetric | mineanybuild |
| percentile_rank | formula_based | metrics.percentile_rank_metric.PercentileRankMetric | webnovelbench |

## Model-Based Metrics

| Metric Name | Metric Type | HELM Class (null if not available) | Datasets Using This Metric |
|-------------|-------------|-------------------------------------|---------------------------|
| sentence_bert_f1 | model_based | helm.benchmark.metrics.summarization_metrics.SummarizationMetric | aaar |
| sentence_bert_precision | model_based | helm.benchmark.metrics.summarization_metrics.SummarizationMetric | aaar |
| sentence_bert_recall | model_based | helm.benchmark.metrics.summarization_metrics.SummarizationMetric | aaar |
| novelty_embedding_similarity | model_based | null | aidanbench |
| bert_score | model_based | helm.benchmark.metrics.summarization_metrics.SummarizationMetric | bhp_hypothesis_generation, data_narrative, llm4biohypogen, ocw_connections, scimon, splat, twistlist, v_flute |
| clip_score | model_based | helm.benchmark.metrics.image_generation.clip_score_metrics.CLIPScoreMetric | esp_dataset |
| appropriateness | model_based | null | cdat |
| novelty | model_based | null | cdat |
| semantic_diversity | model_based | null | dat, dat_creative_writing, dialogue_diversity |
| association_distance | model_based | null | pace |
| diversity_score | model_based | null | noveltybench |
| quality_score | model_based | null | noveltybench |
| plddt_score | model_based | null | protein_bench |
| sctm_score | model_based | null | protein_bench |
| novelty_tmscore | model_based | null | protein_bench |
| f1_at_bertscore_threshold | model_based | null | vflute |
| semantic_diversity_score | model_based | null | sdat |

## LLM Judge Metrics

| Metric Name | Metric Type | Judge Model | Datasets Using This Metric |
|-------------|-------------|-------------|---------------------------|
| win_rate | llm_judge | openai/gpt-4-turbo | alpaca_eval_2, arena_hard_creative, arena_hard_v01 |
| coherence_score | llm_judge | openai/o1-mini | aidanbench |
| recall_gt_entail_score | llm_judge | openai/gpt-4-1106-preview | aaar |
| precision_pred_entail_score | llm_judge | openai/gpt-4-1106-preview | aaar |
| fluency | llm_judge | openai/gpt-4-0125-preview | arastories |
| coherence | llm_judge | openai/gpt-4-0125-preview | arastories |
| following_instructions | llm_judge | openai/gpt-4-0125-preview | arastories |
| consistency | llm_judge | openai/gpt-4-0125-preview | arastories |
| variety | llm_judge | openai/gpt-4-0125-preview | arastories |
| rubric_score | llm_judge | openai/gpt-4o / anthropic/claude-sonnet-4 | artinsight, eqbench_creative_writing_v3, gauss |
| elo_rating | llm_judge | anthropic/claude-sonnet-4 | eqbench_creative_writing_v3 |
| llm_judge_quality | llm_judge | openai/gpt-4 / gpt-4o | banner_request_400, calligrapher, conceptual_design, cpers, fscg8, grapheval_ai_researcher, grapheval_iclr, grapheval_review_advisor, hummus, layoutsam_eval, litbench, llm_review_focus, macgyver, matdesign, permpst, rebus_puzzle, research_idea_execution, webnovelbench, writingbench |
| llm_judge_creativity | llm_judge | openai/gpt-4 / gpt-4o | creai_cps, creatset, creative_process, dat_creative_writing, dpt, geo_story, liveideabench, mineanybuild, pollux_creativity, pron_vs_prompt, showerthoughts, slang_generation |
| llm_judge_originality | llm_judge | openai/gpt-4 | creative_process, mops, pollux_creativity, pron_vs_prompt |
| llm_judge_correctness | llm_judge | openai/gpt-4 / gpt-4o | creativemath, discovery_bench, llm_srbench, macgyver, rebus_puzzle, speak_to_structure |
| llm_judge_novelty | llm_judge | openai/gpt-4 | future_ideas, hypobench, hypogen, research_idea_execution, slang_generation |
| llm_judge_relevance | llm_judge | openai/gpt-4 | future_ideas, slang_generation |
| llm_judge_significance | llm_judge | openai/gpt-4 | hypobench |
| llm_judge_verifiability | llm_judge | openai/gpt-4 | hypobench |
| llm_judge_feasibility | llm_judge | openai/gpt-4o | research_idea_execution |
| judge_score_analogy | llm_judge | openai/gpt-4 | historical_analogy |
| llm_judge_fascination | llm_judge | openai/gpt-4-turbo | mops |
| llm_judge_beauty_of_sound | llm_judge | openai/gpt-4 | poetmt |
| llm_judge_beauty_of_form | llm_judge | openai/gpt-4 | poetmt |
| llm_judge_beauty_of_meaning | llm_judge | openai/gpt-4 | poetmt |
| llm_judge_literary_devices | llm_judge | deepseek/deepseek-v3 | webnovelbench |
| llm_judge_character_consistency | llm_judge | deepseek/deepseek-v3 | webnovelbench |
| llm_judge_humor | llm_judge | openai/gpt-4 | showerthoughts |
| llm_judge_cleverness | llm_judge | openai/gpt-4 | showerthoughts |
| llm_judge_attractiveness | llm_judge | openai/gpt-4 | pron_vs_prompt |
| interestingness | llm_judge | openai/gpt-4o | rpgbench |
| grammar_score | llm_judge | openai/gpt-4 / o3-mini-2025-01-31 | tinyfabulist, tinystories |
| creativity_score | llm_judge | openai/o3-mini-2025-01-31 | tinyfabulist |
| consistency_score | llm_judge | openai/gpt-4 | tinystories |
| llm_judge_coherence | llm_judge | openai/gpt-4 | ss_gen |
