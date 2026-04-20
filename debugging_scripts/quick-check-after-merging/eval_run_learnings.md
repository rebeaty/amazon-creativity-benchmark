# Eval Run Learnings — trial_after_merging_April_20

## aaar — SUCCESS
- **Attempts**: 4
- **Root cause**: N/A (succeeded within 4 attempts; no fatal error reached)
- **Fix applied**: N/A
- **Key learning**: aaar runs two subtasks (`experiment_design`, `paper_weakness`); both produce 4 metrics each (BasicMetric, SentenceBertMetric, two GenericLLMJudgeMetrics); `sentence_bert_*` and NLI entailment metrics (`recall_gt_entail_score`, `precision_pred_entail_score`) are expected to appear in stats but are not registered in `schema_classic.yaml` — these warnings are benign and do not block the run.

## alpaca_eval_2 — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Uses Google Gemini backend (`google_gemini-2.5-flash-lite`); transient 503 UNAVAILABLE errors from the Google GenAI API are expected under high demand and are retried automatically — the run succeeded after 2 retries. Produces 3 stats including `win_rate` (via `GenericLLMJudgeMetric`); `win_rate` is not registered in `schema_classic.yaml` but this warning is benign.

## analobench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: MCQ analogical reasoning task (340 instances, 4-option multiple choice); prompt instructs model to output only the letter index. Uses `BasicGenerationMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) — BLEU warnings about 0-gram overlaps are expected since model outputs are single letters. Runs cleanly on first attempt with no code changes required.

## arastories — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Arabic story generation task (2996 instances); prompts are in Arabic and specify age group, setting, tone, dialect, and other constraints. Produces 5 LLM-judge metrics (`fluency`, `coherence`, `following_instructions`, `consistency`, `variety`) via `GenericLLMJudgeMetric` — none are registered in `schema_classic.yaml` but warnings are benign. Annotation step completes near-instantly (0.036s) because judge calls are cached. Runs cleanly on first attempt.

## arena_hard_v01 — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Open-ended instruction-following/coding benchmark (500 instances, 10 evaluated); uses `GenericLLMJudgeMetric` producing 3 stats. Transient 503 UNAVAILABLE from Google GenAI API on first attempt was auto-retried and succeeded. `win_rate` is not in `schema_classic.yaml` but the warning is benign. Runs cleanly on first attempt with no code changes.

## arn — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Analogical Reasoning Network task (1095 instances, 10 evaluated); asks model to pick the better narrative analogy and explain why. Uses `BasicGenerationMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) — BLEU n-gram warnings are expected since free-form justification responses rarely overlap at higher n-gram orders with short reference strings. No annotators. Runs cleanly on first attempt with no code changes.

## artinsight — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Art analysis/interpretation task (30 instances, 10 evaluated); produces a single `rubric_score` metric via `GenericLLMJudgeMetric` with a 5-point rubric judging depth of insight, accuracy, and reasoning. Uses `GenericLLMJudgeAnnotator` (gpt-4o judge, temperature 0.0). `rubric_score` is not registered in `schema_classic.yaml` but the warning is benign. Annotation completes near-instantly (0.018s) due to cached judge calls. Runs cleanly on first attempt with no code changes.

## balancecc_prompt_generation — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Video-editing prompt generation task (400 instances, 10 evaluated); model is given a video description with metadata (category, scene/camera/object motion complexity) and asked to generate a creative "Target Prompt" for a given Editing Type and Fantasy Level. Uses `BasicMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4); BLEU 2–4-gram warnings are expected since generated prompts rarely overlap exactly with references. No annotators. Runs cleanly on first attempt with no code changes.

## banner_request_400 — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Advertisement/banner copy generation task (400 instances, 10 evaluated); uses `GenericLLMJudgeMetric` producing a single `llm_judge_quality` stat scored 1–5 by a gpt-4o judge on clarity, persuasiveness, creativity, and audience suitability. `llm_judge_quality` is not registered in `schema_classic.yaml` but the warning is benign. All 10 Google GenAI inference calls and annotation calls were served from cache (0 computes). Runs cleanly on first attempt with no code changes.

## brainteaser — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Lateral-thinking riddle MCQ task (396 instances, 10 evaluated); uses `multiple_choice_joint` adapter with 4 options and `BasicGenerationMetric` (exact_match only). Produces 56 stats. No annotators. Runs cleanly on first attempt with no code changes required.

## c3_crosstalk — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Chinese crosstalk (相声) dialogue continuation task (10 instances evaluated); prompt is in Chinese and asks the model to continue a 相声 dialogue for ~10 more lines. Uses `BasicMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4); BLEU 2–4-gram warnings are expected since open-ended Chinese dialogue generations rarely share exact n-gram sequences with references. No annotators. All 10 Google GenAI inference calls were live (0 cache hits). Runs cleanly on first attempt with no code changes required.

## calligrapher — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Self-referential calligraphy description task (80 instances, 10 evaluated); uses `GenericLLMJudgeMetric` producing a single `llm_judge_quality` stat scored 1–5 by a gpt-4o judge on poetic quality, self-reference accuracy, aesthetic coherence, and linguistic creativity. All 10 Google GenAI inference calls and annotation calls were served from cache (0 computes). `llm_judge_quality` is not registered in `schema_classic.yaml` but the warning is benign. Runs cleanly on first attempt with no code changes required.

## cdat — SUCCESS
- **Attempts**: 2
- **Root cause**: Initial attempt failed; fixed within 2 attempts (exact error not captured in log tail).
- **Fix applied**: N/A (run succeeded; no code change details in log).
- **Key learning**: Convergent/Divergent Associative Thinking task (582 cue words downloaded from GitHub at runtime); prompt asks model to generate 10 mutually dissimilar nouns semantically associated with a cue word, returned as a comma-separated list. Uses `CreativityScoreMetric` producing 9 stats including `creativity_score`, `novelty`, and `appropriateness` — none are registered in `schema_classic.yaml` but warnings are benign. All 10 Google GenAI inference calls were served from cache (0 computes). Dataset is downloaded live from GitHub on each run; ensure network access is available.

## chinese_homophonic_puns — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Chinese homophonic pun identification task (10 instances evaluated); prompt is in Chinese and asks the model to identify the word that is the source of humor in a joke via homophones. Uses `BasicMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4); BLEU 2–4-gram warnings are expected since answers are typically single Chinese words. All 10 Google GenAI inference calls were live (10 computes, 0 cache hits). Runs cleanly on first attempt with no code changes required.

## cpers — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Creative persona/character task (10 instances evaluated); uses `BasicMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) plus `GenericLLMJudgeMetric`, producing 74 stats total. BLEU 2–4-gram warnings are expected for open-ended generative outputs. 8 of 10 Google GenAI inference calls were live (2 cache hits); annotation step was near-instant (0.002s) due to cached judge calls. Runs cleanly on first attempt with no code changes required.

## creai_cps — SUCCESS
- **Attempts**: 2
- **Root cause**: Minor issue resolved within 2 attempts (exact error not captured in log tail).
- **Fix applied**: N/A (run succeeded; no code change details in log).
- **Key learning**: Creative problem-solving task (10 instances evaluated); uses `BasicMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) plus `GenericLLMJudgeMetric` producing `llm_judge_creativity`, for 74 stats total. `llm_judge_creativity` is not registered in `schema_classic.yaml` but the warning is benign. 8 of 10 Google GenAI inference calls were live (2 cache hits); annotation step was near-instant (0.002s) due to cached judge calls. BLEU n-gram warnings are expected for open-ended generative outputs.

## creative_process — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Creative process evaluation task (5 instances evaluated); uses two `GenericLLMJudgeMetric` instances producing 6 stats total including `llm_judge_originality`. `llm_judge_originality` is not registered in `schema_classic.yaml` but the warning is benign. 4 of 5 Google GenAI inference calls were live (1 cache hit); annotation step was near-instant (0.022s) due to cached judge calls. Runs cleanly on first attempt with no code changes required.

## creativemath — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Novel math solution generation task (10 instances evaluated); prompt provides a math problem with existing solutions and asks the model to produce a distinct new solution. Uses a single `GenericLLMJudgeMetric` producing 3 stats including `llm_judge_correctness`. `llm_judge_correctness` is not registered in `schema_classic.yaml` but the warning is benign. All 10 Google GenAI inference calls were live (0 cache hits); annotation step was near-instant (0.001s) due to cached judge calls. Runs cleanly on first attempt with no code changes required.

## creatset — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Creative text generation task (10 instances evaluated); uses `BasicMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) plus a `GenericLLMJudgeMetric`, producing 74 stats total. 10 Google GenAI inference calls were live (0 cache hits) and 30 HuggingFace calls ran (30 computes, 10 cache hits). BLEU 2–4-gram warnings are expected for open-ended generative outputs. Annotation step was near-instant (0.002s) due to cached judge calls. Runs cleanly on first attempt with no code changes required.

## critics_story — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Story critique/evaluation task (10 instances evaluated); uses a single `GenericLLMJudgeMetric` producing 3 stats. 10 Google GenAI inference calls were live (10 computes, 0 cache hits); 30 HuggingFace tokenizer calls ran (30 computes, 0 cache hits); annotation step was near-instant (0.011s) due to cached judge calls. Runs cleanly on first attempt with no code changes required.

## crowd_vote — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Crowd voting/preference task (10 instances evaluated); uses a single `GenericLLMJudgeMetric` producing 3 stats. 10 Google GenAI inference calls were live (10 computes, 0 cache hits); 30 HuggingFace tokenizer calls ran (30 computes, 0 cache hits); annotation step was near-instant (0.015s) due to cached judge calls. Runs cleanly on first attempt with no code changes required.

## cs4 — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: CS4 task (10 instances evaluated); uses a single `GenericLLMJudgeMetric` producing 3 stats. 11 Google GenAI inference calls ran (11 computes, 0 cache hits) with one transient 503 UNAVAILABLE auto-retried successfully; 30 HuggingFace tokenizer calls ran (30 computes, 0 cache hits); annotation step was near-instant (0.022s) due to cached judge calls. Transient 503 errors from Google GenAI are expected under high demand and are retried automatically — no code changes needed.

## csd100 — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: CSD100 task (10 instances evaluated); uses `BasicMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) producing 71 stats. All 10 Google GenAI inference calls were served from cache (0 computes); 40 HuggingFace tokenizer calls also hit cache (0 computes). No annotators. Runs cleanly on first attempt with no code changes required.

## cue_word_story — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Story generation from cue words task (10 instances evaluated); produces 74 stats via `BasicMetric` (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) and `GenericLLMJudgeMetric`. 4 Google GenAI inference calls were live (4 computes, 0 cache hits); 16 HuggingFace calls ran with 12 computes (4 cache hits); annotation step was near-instant (0.002s) due to cached judge calls. BLEU lower-order n-gram warnings are expected for open-ended story outputs. Runs cleanly on first attempt with no code changes required.

## dat — SUCCESS
- **Attempts**: 2
- **Root cause**: Minor issue resolved within 2 attempts (exact error not captured in log tail).
- **Fix applied**: N/A (run succeeded; no code change details in log).
- **Key learning**: Divergent Association Task (10 instances evaluated); produces 3 stats via `SemanticDiversityMetric` (`semantic_diversity` is the key metric). All 10 Google GenAI inference calls and 30 HuggingFace calls were served from cache (0 computes). `semantic_diversity` is not registered in `schema_classic.yaml` but the warning is benign. This is a batch-only metric requiring multiple responses per context — ensure the run spec requests multiple completions per instance.

## dat_creative_writing — SUCCESS
- **Attempts**: 2
- **Root cause**: Minor issue resolved within 2 attempts (exact error not captured in log tail).
- **Fix applied**: N/A (run succeeded; no code change details in log).
- **Key learning**: DAT creative writing task (10 instances evaluated); produces stats written to `stats.json` (2217 chars) and `per_instance_stats.json` (8250 chars). All 10 Google GenAI inference calls and 30 HuggingFace calls were served from cache (0 computes). Closely related to the `dat` dataset — if one fails, check the other for shared metric/scenario code. Runs complete in ~14s total with cached inference.

## data_narrative — SUCCESS
- **Attempts**: 2
- **Root cause**: Minor issue resolved within 2 attempts (exact error not captured in log tail).
- **Fix applied**: N/A (run succeeded; no code change details in log).
- **Key learning**: Data narrative generation task (10 instances evaluated); uses `BertScoreMetric` producing a `bert_score` stat. `bert_score` is not registered in `schema_classic.yaml` but the warning is benign. All 40 HuggingFace calls and 10 Google GenAI inference calls were served from cache (0 computes); summarization ran in ~16s total. If `bert_score` is missing from stats, check that `BertScoreMetric` is correctly wired in the run spec and that the metric annotator completes without error.

## deep_math — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Deep math task (10 instances evaluated); outputs written to `instances.json` (5520 chars), `display_predictions.json` (23296 chars), and `display_requests.json` (8080 chars). Ran cleanly on first attempt with no code changes required. All inference and summarization completed within the ~17.5s annotation window visible in the log tail.

## dialogue_diversity — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Dialogue diversity task (10 instances evaluated); produces 3 custom metrics — `distinct_1`, `distinct_2`, and `coherence_score` — none of which are registered in `schema_classic.yaml`, but the warnings are benign. Outputs written to `instances.json` (30918 chars), `display_predictions.json` (2883 chars), and `display_requests.json` (13028 chars). `distinct_1`/`distinct_2` are batch-only metrics requiring multiple responses per context — ensure run spec requests multiple completions. Ran cleanly on first attempt with no code changes required.

## discovery_bench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Scientific discovery/hypothesis benchmark (10 instances evaluated); outputs written to `instances.json` (30977 chars), `display_predictions.json` (22256 chars), and `display_requests.json` (31977 chars). Ran cleanly on first attempt with no code changes required. All inference and summarization completed within the final parallel `write_run_display_json` sweep alongside `dialogue_diversity` and `deep_math`.

## diverse_not_short — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Diverse-not-short generation task (10 instances evaluated); outputs written to `instances.json` (5961 chars), `display_predictions.json` (1477 chars), and `display_requests.json` (8471 chars). Ran cleanly on first attempt with no code changes required. Output files are notably small compared to peer datasets, suggesting short prompts and brief reference outputs — the "not short" constraint is on the model's generated responses, not the benchmark instances themselves.

## dpt — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: DPT task (10 instances evaluated); outputs written to `instances.json` (4173 chars), `display_predictions.json` (7306 chars), and `display_requests.json` (7943 chars). Ran cleanly on first attempt with no code changes required. Processed in the final parallel `write_run_display_json` sweep alongside `discovery_bench` and `diverse_not_short`.

## eqbench_creative_writing_v3 — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: EQBench creative writing evaluation task (10 instances evaluated); uses `elo_rating` metric which is not registered in `schema_classic.yaml` — the warning is benign. `display_predictions.json` is notably large (208394 chars) compared to peers, reflecting long per-instance generated text. Outputs written to `instances.json` (9788 chars), `display_predictions.json` (208394 chars), and `display_requests.json` (13568 chars). Ran cleanly on first attempt with no code changes required.

## fann_or_flop — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Fan-fiction fandom prediction task; notably large output files — `instances.json` (327758 chars), `display_predictions.json` (65918 chars), `display_requests.json` (190620 chars) — suggesting long per-instance context (likely full fan-fiction passages). Ran cleanly on first attempt with no code changes required; processed in the final parallel `write_run_display_json` sweep alongside `eqbench_creative_writing_v3`.

## fscg8 — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: FSCG8 task (10 instances evaluated); compact output files — `instances.json` (5572 chars), `display_requests.json` (5276 chars) — but `display_predictions.json` is relatively large (25452 chars), indicating the model produces verbose generated outputs relative to short input prompts. Ran cleanly on first attempt with no code changes required.

## future_ideas — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Future ideas generation task; notably large output files — `instances.json` (143168 chars), `display_predictions.json` (28464 chars), `display_requests.json` (142838 chars) — suggesting long per-instance context (likely detailed scenario descriptions). Ran cleanly on first attempt with no code changes required; processed in the final parallel `write_run_display_json` sweep alongside `fann_or_flop` and `fscg8`.

## futuregen — SKIPPED (data access)
- **Attempts**: 1
- **Root cause**: Eval script exceeded the 120s timeout on the first attempt, treated as a data access error.
- **Fix applied**: N/A
- **Key learning**: Timeout at attempt 1 almost always means the dataset is not available locally or requires a slow remote download; verify local data path and network access before retrying.

## futuregen — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Very large output files — `instances.json` (1067118 chars), `display_predictions.json` (31108 chars), `display_requests.json` (1054035 chars) — largest instances.json in the suite, indicating long per-instance context (likely full research papers or detailed scenario descriptions); allow extra time for file I/O when running this dataset.

## fuxibench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Music/creative benchmark task (10 instances evaluated); compact output files — `instances.json` (10305 chars), `display_predictions.json` (21107 chars), `display_requests.json` (7560 chars). Ran cleanly on first attempt with no code changes required; processed in the final parallel `write_run_display_json` sweep immediately after `future_ideas` and `futuregen`.

## gauss — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: GAUSS creativity evaluation task (10 instances evaluated); compact output files — `instances.json` (4093 chars), `display_predictions.json` (6440 chars), `display_requests.json` (1922 chars). Ran cleanly on first attempt with no code changes required; processed at the very end of the final parallel `write_run_display_json` sweep, one of the last datasets to finish writing in the suite.

## geo_story — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Geographic story generation task (10 instances evaluated); compact output files — `instances.json` (3078 chars), `display_predictions.json` (26510 chars), `display_requests.json` (5638 chars). Large `display_predictions.json` relative to `instances.json` suggests the model produces verbose story outputs for short geographic prompts. Ran cleanly on first attempt with no code changes required; processed in the same parallel sweep as `fuxibench`, `futuregen`, and `gauss`.

## grapheval_ai_researcher — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: AI researcher graph evaluation task (10 instances evaluated); notably large output files — `instances.json` (86944 chars), `display_predictions.json` (2376 chars), `display_requests.json` (89149 chars) — large `instances.json` and `display_requests.json` indicate long per-instance prompts (likely graph-structured research context), while the small `display_predictions.json` suggests short model outputs (e.g., classification or short-form answers). Ran cleanly on first attempt with no code changes required; one of the last datasets written in the final parallel sweep (annotation took 0.855s, the longest in the suite tail).

## grapheval_iclr — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: ICLR paper graph evaluation task (10 instances evaluated); large output files — `instances.json` (94068 chars), `display_predictions.json` (3875 chars), `display_requests.json` (96505 chars) — similar profile to `grapheval_ai_researcher` (large prompts, short outputs), consistent with graph-structured academic paper context feeding into short classification/scoring responses. Written in the same final parallel sweep as `grapheval_ai_researcher` and `geo_story`. Ran cleanly on first attempt with no code changes required.

## grapheval_review_advisor — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Review advisor graph evaluation task (10 instances evaluated); large output files — `instances.json` (94710 chars), `display_predictions.json` (2626 chars), `display_requests.json` (97201 chars) — same large-prompt/short-output profile as `grapheval_ai_researcher` and `grapheval_iclr`, consistent with graph-structured paper review context yielding short advisory responses. Written in the same final parallel sweep as the other `grapheval_*` datasets. Ran cleanly on first attempt with no code changes required.

## graphrag_bench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: GraphRAG benchmark task (10 instances evaluated); compact output files — `instances.json` (9870 chars), `display_predictions.json` (25977 chars), `display_requests.json` (6298 chars) — moderate prompt size with notably larger predictions, suggesting the model produces extended graph-grounded narrative or retrieval-augmented answers. Written in the same final parallel sweep as `grapheval_*` siblings. Ran cleanly on first attempt with no code changes required.

## graphragbench-wrongone — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: GraphRAGBench "wrong-one" variant task (10 instances evaluated); output files — `instances.json` (9891 chars), `display_predictions.json` (14200 chars), `display_requests.json` (7042 chars) — similar scale to `graphrag_bench` but with larger predictions, consistent with a negative/wrong-answer detection framing that elicits longer explanatory responses. Written in the same final parallel sweep as `graphrag_bench` and `grapheval_*`. Ran cleanly on first attempt with no code changes required.

## historical_analogy — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Historical analogy generation task (10 instances evaluated); notably large output files — `instances.json` (49367 chars), `display_predictions.json` (3090 chars), `display_requests.json` (51997 chars) — large prompts with short outputs suggest the model is given rich historical context and asked to produce a concise analogy response. Registry maps this to `judge_score_analogy` (1–4 scale). Written in the same final parallel sweep as `grapheval_*` and `graphrag*` siblings. Ran cleanly on first attempt with no code changes required.

## humor_transfer — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Humor transfer task (10 instances evaluated); compact output files — `instances.json` (6906 chars), `display_predictions.json` (2287 chars), `display_requests.json` (8536 chars) — small predictions relative to requests suggest short humor-rewritten outputs for moderate-length input prompts. Written in the same final parallel sweep as `historical_analogy` and `graphragbench-wrongone`. Ran cleanly on first attempt with no code changes required.

## hypobench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Hypothesis generation benchmark task (10 instances evaluated); compact output files — `instances.json` (13233 chars), `display_predictions.json` (2390 chars), `display_requests.json` (10611 chars) — small predictions relative to requests suggest concise hypothesis outputs for moderately long scientific prompts. Written in the same final parallel sweep as `historical_analogy` and `humor_transfer`. Ran cleanly on first attempt with no code changes required.

## hypogen — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Hypothesis generation task (10 instances evaluated); balanced output files — `instances.json` (24336 chars), `display_predictions.json` (26378 chars), `display_requests.json` (23517 chars) — all three files are similar in size, indicating moderate-length prompts and similarly verbose model outputs (unlike `hypobench` where predictions are much smaller than requests). Written in the same final parallel sweep as `hypobench` and `historical_analogy`. Ran cleanly on first attempt with no code changes required.

## idrbench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: IDR (Interdisciplinary Research) benchmark task (10 instances evaluated); large output files — `instances.json` (36819 chars), `display_predictions.json` (2574 chars), `display_requests.json` (37151 chars) — large prompts with small predictions indicate dense per-instance context (likely full research paper or cross-domain problem descriptions) yielding short model responses. Written in the same final parallel sweep as `hypobench`, `hypogen`, and `historical_analogy`. Ran cleanly on first attempt with no code changes required.

## infochartqa — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: InfoChartQA task (10 instances evaluated); balanced output files — `instances.json` (7831 chars), `display_predictions.json` (2621 chars), `display_requests.json` (7677 chars) — moderate prompt size with small predictions, consistent with a chart-based QA task yielding short factual answers. One of the last datasets written in the final parallel sweep (written alongside `idrbench` and `hypogen`). Ran cleanly on first attempt with no code changes required.

## liveideabench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Live idea benchmark task (10 instances evaluated); balanced output files — `instances.json` (6328 chars), `display_predictions.json` (9877 chars), `display_requests.json` (10098 chars) — moderate prompt and prediction sizes consistent with open-ended idea generation. One of the last datasets written in the final parallel sweep (written after `infochartqa` and `idrbench`). Ran cleanly on first attempt with no code changes required.

## llm_discussion — SUCCESS
- **Attempts**: 2
- **Root cause**: Minor issue resolved within 2 attempts (exact error not captured in log tail).
- **Fix applied**: N/A (run succeeded; no code change details in log).
- **Key learning**: LLM discussion task (10 instances evaluated); compact output files — `instances.json` (3817 chars), `display_predictions.json` (23667 chars), `display_requests.json` (7587 chars) — small instances and requests with large predictions suggest short prompts eliciting verbose open-ended responses. Registry maps this dataset to no-reference metrics (4 subsets: aut, similarities, instances, scientific). Written in the final parallel sweep alongside `idrbench`, `infochartqa`, and `liveideabench`. If attempt 1 fails, retry without code changes — the second attempt often succeeds due to caching or transient API errors.

## macgyver — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: MacGyver creative tool-use task (10 instances evaluated); balanced output files — `instances.json` (12882 chars), `display_predictions.json` (4766 chars), `display_requests.json` (12073 chars) — moderate prompt size with smaller predictions, consistent with asking the model to propose a creative solution using available objects and yielding a concise answer. Written in the final parallel sweep alongside `liveideabench`, `llm_discussion`, and `infochartqa`. Ran cleanly on first attempt with no code changes required.

## matdesign — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Material design generation task (10 instances evaluated); compact output files — `instances.json` (2600 chars), `display_predictions.json` (2516 chars), `display_requests.json` (2308 chars) — among the smallest in the suite, suggesting very short prompts and brief model outputs consistent with a structured material-property generation task with no references. Registry maps this to no-reference metrics; expect stats without BLEU/ROUGE. Written in the final parallel sweep immediately after `macgyver` and `llm_discussion`. Ran cleanly on first attempt with no code changes required.

## metaphor_generation — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Metaphor generation task (10 instances evaluated); compact output files — `instances.json` (3096 chars), `display_predictions.json` (19884 chars), `display_requests.json` (5257 chars) — small prompts with notably large predictions suggest the model produces verbose metaphor outputs for short input stimuli. Registry maps this to BLEU/ROUGE/F1 reference-based metrics; BLEU n-gram warnings are expected for open-ended generative outputs. Written in the final parallel sweep alongside `matdesign` and `macgyver`. Ran cleanly on first attempt with no code changes required.

## mineanybuild — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Minecraft build generation task (10 instances evaluated); large output files — `instances.json` (35320 chars), `display_predictions.json` (24264 chars), `display_requests.json` (29952 chars) — balanced sizes across all three files indicate moderate-length prompts and similarly verbose model outputs, consistent with creative open-ended build descriptions. Written in the final parallel sweep alongside `matdesign`, `macgyver`, and `metaphor_generation`. Ran cleanly on first attempt with no code changes required.

## mixassist — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: MixAssist task (10 instances evaluated); large output files — `instances.json` (57516 chars), `display_predictions.json` (11357 chars), `display_requests.json` (57576 chars) — nearly equal `instances.json` and `display_requests.json` sizes indicate long per-instance prompts, while moderately sized predictions suggest concise model outputs relative to the input context. Written in the final parallel sweep alongside `mineanybuild`, `matdesign`, and `metaphor_generation`. Ran cleanly on first attempt with no code changes required.

## mops — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: MOPS task (10 instances evaluated); balanced output files — `instances.json` (19214 chars), `display_predictions.json` (6722 chars), `display_requests.json` (19141 chars) — large prompts with shorter predictions suggest dense per-instance context yielding concise model outputs. Written in the final parallel sweep alongside `mixassist` and `mineanybuild`. Ran cleanly on first attempt with no code changes required.

## neocoder — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: NeoCoder task (10 instances evaluated); balanced output files — `instances.json` (27867 chars), `display_predictions.json` (14183 chars), `display_requests.json` (23904 chars) — moderate prompt and prediction sizes consistent with a code generation or creative coding task. Written in the final parallel sweep alongside `mops` and `mixassist`. Ran cleanly on first attempt with no code changes required.

## nyt_connections — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: NYT Connections word-grouping task (10 instances evaluated); compact output files — `instances.json` (8023 chars), `display_predictions.json` (2326 chars), `display_requests.json` (9482 chars) — moderate prompts with small predictions consistent with a grouping/classification task yielding short structured answers. Written in the final parallel sweep alongside `mops` and `neocoder`. Ran cleanly on first attempt with no code changes required.

## ocw — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: OCW (OpenCourseWare) task (10 instances evaluated); balanced output files — `instances.json` (11412 chars), `display_predictions.json` (2518 chars), `display_requests.json` (54916 chars) — very large `display_requests.json` relative to `instances.json` indicates long prompt templates or system instructions wrapping short per-instance content, while small predictions suggest concise model outputs. Written in the final parallel sweep as one of the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## permpst — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: PermPST task (10 instances evaluated); large output files — `instances.json` (47265 chars), `display_predictions.json` (9940 chars), `display_requests.json` (49795 chars) — nearly equal `instances.json` and `display_requests.json` sizes indicate long per-instance prompts, while moderately sized predictions suggest concise model outputs; one of the last datasets written in the final parallel sweep alongside `ocw` and `nyt_connections`. Ran cleanly on first attempt with no code changes required.

## poetmt — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Poetry machine translation task (10 instances evaluated); balanced output files — `instances.json` (12200 chars), `display_predictions.json` (21764 chars), `display_requests.json` (10517 chars) — larger predictions than requests suggest the model produces verbose translated poem outputs relative to short source prompts. Written in the final parallel sweep alongside `nyt_connections` and `ocw`. Ran cleanly on first attempt with no code changes required.

## pollux_creativity — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Pollux creativity evaluation task (10 instances evaluated); large output files — `instances.json` (97665 chars), `display_predictions.json` (88691 chars), `display_requests.json` (52702 chars) — all three files are substantial, indicating long per-instance prompts and verbose model outputs consistent with an open-ended creative generation task. Written in the final parallel sweep alongside `permpst`, `ocw`, and `poetmt`. Ran cleanly on first attempt with no code changes required.

## pron_vs_prompt — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Pronoun-vs-prompt evaluation task (10 instances evaluated); balanced output files — `instances.json` (9461 chars), `display_predictions.json` (28023 chars), `display_requests.json` (13231 chars) — larger predictions than requests suggest the model produces verbose outputs for moderate-length prompts; written in the final parallel sweep alongside `pollux_creativity`, `permpst`, and `poetmt`. Ran cleanly on first attempt with no code changes required.

## protein_bench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Protein bench task (10 instances evaluated); compact output files — `instances.json` (6645 chars), `display_predictions.json` (11138 chars), `display_requests.json` (10415 chars) — larger predictions than instances suggest moderate-length prompts eliciting verbose protein-design or biology-focused outputs. Written in the final parallel sweep alongside `pron_vs_prompt` and `pollux_creativity`. Ran cleanly on first attempt with no code changes required.

## pun_eval — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Pun evaluation task (10 instances evaluated); balanced output files — `instances.json` (11747 chars), `display_predictions.json` (3341 chars), `display_requests.json` (13723 chars) — moderate prompts with short predictions consistent with a pun quality judgment or classification task. Written in the final parallel sweep immediately after `protein_bench`, one of the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## puntuguese — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Portuguese pun binary classification task (10 instances evaluated); compact output files — `instances.json` (4237 chars), `display_predictions.json` (2454 chars), `display_requests.json` (6087 chars) — small predictions relative to requests indicate short binary (pun/not-pun) outputs for moderate-length prompts. Registry maps this to binary accuracy. Written in the final parallel sweep immediately after `pun_eval` and `protein_bench`, one of the very last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## rebus_puzzle — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Rebus puzzle task (10 instances evaluated); compact output files — `instances.json` (10877 chars), `display_predictions.json` (6730 chars), `display_requests.json` (4900 chars) — small requests with moderate predictions suggest short image-based puzzle prompts yielding brief model answers; registry maps this to accuracy. Written at the very end of the final parallel sweep, one of the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## recombination_extraction — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Recombination extraction task (10 instances evaluated); large output files — `instances.json` (39773 chars), `display_predictions.json` (19088 chars), `display_requests.json` (224766 chars) — very large `display_requests.json` (largest in the suite tail) relative to `instances.json` indicates long prompt templates or extensive per-instance context wrapping, while moderate predictions suggest concise extraction outputs; written at the very end of the final parallel sweep, one of the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## research_idea_execution — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Research idea execution task (10 instances evaluated); balanced output files — `instances.json` (37892 chars), `display_predictions.json` (24030 chars), `display_requests.json` (41662 chars) — moderate-to-large prompts with substantial predictions consistent with an open-ended research idea generation/execution task; written at the very end of the final parallel sweep, among the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## robotoolbench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Robotic tool-use benchmark task (10 instances evaluated); balanced output files — `instances.json` (16613 chars), `display_predictions.json` (17186 chars), `display_requests.json` (20383 chars) — moderate prompts with similarly sized predictions consistent with a creative tool-selection or planning task; written at the very end of the final parallel sweep, one of the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## layoutsam_eval — SKIPPED (data access)
- **Attempts**: 1
- **Root cause**: Eval script exceeded the 120s timeout on the first attempt, treated as a data access error.
- **Fix applied**: N/A
- **Key learning**: Timeout at attempt 1 almost always means the dataset is not available locally or requires a slow remote download; verify the local data path and network access before retrying.

## rpgbench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: RPG benchmark task (10 instances evaluated); very large output files — `instances.json` (461163 chars), `display_predictions.json` (25640 chars), `display_requests.json` (464933 chars) — among the largest in the suite, indicating long per-instance prompts (likely full RPG scenario/world descriptions) with moderate predictions; written at the very end of the final parallel sweep, one of the last datasets to finish. Ran cleanly on first attempt with no code changes required.

## scar — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: SCAR task (10 instances evaluated); balanced output files — `instances.json` (34192 chars), `display_predictions.json` (2730 chars), `display_requests.json` (33406 chars) — large prompts with small predictions suggest dense per-instance context yielding concise model outputs; written in the final parallel sweep immediately after `rpgbench`, one of the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## science_analogies — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Science analogies task (10 instances evaluated); balanced output files — `instances.json` (7897 chars), `display_predictions.json` (23995 chars), `display_requests.json` (5490 chars) — small prompts with larger predictions suggest short analogy stimuli eliciting verbose model-generated analogies; written in the final parallel sweep alongside `scar` and `rpgbench`, one of the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## scimon — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: SciMon scientific hypothesis/monitoring task (10 instances evaluated); compact output files — `instances.json` (9531 chars), `display_predictions.json` (3474 chars), `display_requests.json` (10294 chars) — moderate prompts with small predictions suggest concise model outputs for scientific monitoring or hypothesis-tracking queries; written in the final parallel sweep immediately alongside `science_analogies`, `scar`, and `rpgbench`. Ran cleanly on first attempt with no code changes required.

## showerthoughts — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Shower thoughts generation task (10 instances evaluated); compact output files — `instances.json` (5605 chars), `display_predictions.json` (4278 chars), `display_requests.json` (7470 chars) — small balanced files suggest short prompts and brief model outputs consistent with a concise creative-thought generation task; written in the final parallel sweep immediately alongside `scimon` and `science_analogies`. Ran cleanly on first attempt with no code changes required.

## simile_generation — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Simile generation task (10 instances evaluated); compact output files — `instances.json` (6437 chars), `display_predictions.json` (7641 chars), `display_requests.json` (6423 chars) — balanced sizes with predictions slightly larger than instances suggest short prompts eliciting brief but slightly more verbose simile outputs; written in the final parallel sweep immediately alongside `showerthoughts` and `scimon`. Ran cleanly on first attempt with no code changes required.

## slang_generation — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Slang generation task (10 instances evaluated); compact output files — `instances.json` (4351 chars), `display_predictions.json` (6911 chars), `display_requests.json` (6853 chars) — small instances with slightly larger predictions suggest short prompts eliciting brief creative slang outputs; written in the final parallel sweep immediately alongside `simile_generation` and `showerthoughts`. Ran cleanly on first attempt with no code changes required.

## speak_to_structure — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Speak-to-structure task (10 instances evaluated); compact output files — `instances.json` (4683 chars), `display_predictions.json` (2644 chars), `display_requests.json` (8453 chars) — small instances and predictions with larger requests suggest moderate prompt templates wrapping short per-instance content, yielding brief structured outputs; written at the very end of the final parallel sweep, one of the last datasets to finish writing in the suite. Ran cleanly on first attempt with no code changes required.

## ss_gen — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: SS-Gen task (10 instances evaluated); large output files — `instances.json` (15427 chars), `display_predictions.json` (15102 chars), `display_requests.json` (75741 chars) — very large `display_requests.json` relative to `instances.json` indicates long prompt templates or extensive per-instance context wrapping, while predictions are moderate in size; written immediately after `speak_to_structure` at the very end of the final parallel sweep. Ran cleanly on first attempt with no code changes required.

## story_generation_rocstories — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: ROCStories-based story generation task (10 instances evaluated); compact output files — `instances.json` (8748 chars), `display_predictions.json` (24959 chars), `display_requests.json` (6361 chars) — large predictions relative to small requests indicate short narrative prompts eliciting verbose multi-sentence story outputs; written at the very end of the final parallel sweep alongside `ss_gen` and `speak_to_structure`. Ran cleanly on first attempt with no code changes required.

## story_quality — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Story quality evaluation task (10 instances evaluated); compact output files — `instances.json` (11863 chars), `display_predictions.json` (1738 chars), `display_requests.json` (10853 chars) — large requests with very small predictions indicate dense per-instance prompt context yielding short quality-score outputs; written in the final parallel sweep immediately alongside `ss_gen` and `story_generation_rocstories`. Ran cleanly on first attempt with no code changes required.

## textlogo3k — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Text logo generation/evaluation task (10 instances evaluated); compact output files — `instances.json` (17545 chars), `display_predictions.json` (5792 chars), `display_requests.json` (4962 chars) — moderate instances with small predictions and very small requests suggest short prompts yielding brief model outputs, consistent with a structured text-logo description or classification task; written in the final parallel sweep immediately alongside `story_quality` and `story_generation_rocstories`. Ran cleanly on first attempt with no code changes required.

## thenextchapter — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: "The Next Chapter" story continuation task (10 instances evaluated); compact output files — `instances.json` (5206 chars), `display_predictions.json` (4472 chars), `display_requests.json` (5793 chars) — balanced small sizes suggest short continuation prompts with brief model outputs; written in the final parallel sweep immediately alongside `story_quality` and `story_generation_rocstories`. Ran cleanly on first attempt with no code changes required.

## tinyfabulist — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Tiny fabulist story generation task (10 instances evaluated); compact output files — `instances.json` (13810 chars), `display_predictions.json` (19433 chars), `display_requests.json` (17580 chars) — larger predictions than instances suggest short fable-style prompts eliciting verbose generated stories; written in the final parallel sweep immediately alongside `thenextchapter`, `story_quality`, and `textlogo3k`. Ran cleanly on first attempt with no code changes required.

## tinystories — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: TinyStories generation task (10 instances evaluated); compact output files — `instances.json` (4958 chars), `display_predictions.json` (24775 chars), `display_requests.json` (8728 chars) — small prompts with large predictions indicate short story-seed inputs eliciting verbose generated story outputs; registry maps this to no-reference metrics. Written in the final parallel sweep immediately alongside `tinyfabulist` and `thenextchapter`. Ran cleanly on first attempt with no code changes required.

## unfun_corpus — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Unfun corpus task (10 instances evaluated); compact output files — `instances.json` (4519 chars), `display_predictions.json` (11126 chars), `display_requests.json` (6512 chars) — small prompts with larger predictions suggest short humor/pun prompts eliciting moderate model outputs; written at the very end of the final parallel sweep, one of the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## vietnamese_poem — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: Vietnamese poem generation task (10 instances evaluated); balanced output files — `instances.json` (34552 chars), `display_predictions.json` (11216 chars), `display_requests.json` (6584 chars) — large instances with moderate predictions and small requests suggest dense per-instance poem context with short prompt wrappers yielding concise model outputs; written at the very end of the final parallel sweep, among the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.

## writingbench — SUCCESS
- **Attempts**: 1
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: WritingBench task (10 instances evaluated); very large output files — `instances.json` (278161 chars), `display_predictions.json` (232513 chars), `display_requests.json` (187400 chars) — among the largest in the suite, indicating long per-instance writing prompts and verbose model outputs consistent with an open-ended long-form writing benchmark; written at the very end of the final parallel sweep, one of the last datasets to finish in the suite. Ran cleanly on first attempt with no code changes required.
