# Faithfulness Review — 2026-04-21

Automated per-benchmark audit of the **106 benchmarks** that produced `stats.json` in any suite. Each review compares the scenario prompt / run_spec metrics / judge config / sample predictions against the source paper's intended evaluation, using `scenarios/`, `run_specs/`, `metric_notes/`, and `benchmark_output/runs/` as evidence, with WebFetch attempted on the paper URL from each scenario docstring. Reviewer was Claude general-purpose subagent, one invocation per benchmark.

**Verdict scale** (normalised; agents used a few synonyms):

- **FAITHFUL** — matches the paper to a reasonable standard
- **MOSTLY_FAITHFUL** — minor drift (prompt wording, extra metrics) but valid to run
- **DIVERGENT** — major drift that would compromise the evaluation; should fix before the full run

## Summary

| Verdict | Count | % of reviewed |
|---|---:|---:|
| FAITHFUL | 0 | 0.0% |
| MOSTLY_FAITHFUL | 51 | 48.1% |
| DIVERGENT | 55 | 51.9% |
| **TOTAL** | **106** | 100.0% |

## Blocking fixes before launch (181 total)

These were flagged as `severity: blocking` by the reviewer — running the full evaluation against these without patching first will produce metrics that don't represent what the source paper measures. Grouped by benchmark:

### `alpaca_eval_2`
- Replace Likert rubric with the official AlpacaEval pairwise prompt (see annotator_notes.md:15-36). Judge must compare model output vs reference baseline and return a binary preference; metric must be fraction of wins, not mean of 1-5 scores.
- Add a length-controlled win rate computation (or explicitly document that only raw WR is reported and rename metric accordingly) so results are not labeled 'alpaca_eval_2' while computing something the paper does not endorse.
- Verify which baseline is in the downloaded alpaca_eval.json; if it is text_davinci_003, switch to the AlpacaEval 2.0 baseline (gpt4_turbo reference_outputs) before running.

### `amuse_chord_generation`
- Raise max_tokens from 512 to at least 2048 (30 progressions x ~4 chords x ~5 tokens/chord + newlines). Current truncation is visible in multiple display_predictions entries and will corrupt diversity metrics.
- Reconcile num_outputs with batch semantics: either set num_outputs=1 (paper's batch prompting, parse 30 progressions from one completion) OR drop num_progressions to 1 in the scenario prompt and keep num_outputs=30 (conventional prompting baseline). Current config does both simultaneously and Self-BLEU will be computed over the wrong unit.

### `arena_hard_creative`
- Reconcile rubric: either adopt the 1-10 four-dimension absolute rubric from annotator_notes.md, or remove the 'head-to-head vs reference' framing from _RUBRIC_WIN_RATE since no reference is supplied.
- Raise max_tokens well above 512 (e.g. 2048-4096) to avoid truncating creative responses.

### `arena_hard_v01`
- Decide: either (a) load GPT-4 baseline answers + switch annotator to pairwise A/B judge to compute true win_rate, or (b) rename metric to 'quality_score_1_5' and accept this as a pointwise variant — current state reports a metric that does not match its name or the paper

### `arn`
- Add a prediction post-processor or custom metric that extracts narrative_1/narrative_2 from the templated output and maps to '1'/'2' before exact_match; otherwise accuracy will be ~0 despite correct answers

### `artinsight`
- Replace generic 1-5 rubric in run_specs/artinsight_run_specs.py with the paper's 0-16 five-criterion rubric and wire a vision-capable judge that receives the artwork image alongside the description.
- Investigate why predicted_text is identical across all 30 instances and why base64_images is empty — multimodal input (MultimediaObject in scenarios/artinsight_scenario.py:133-136) is not being delivered to models; fix before eval run.
- Add the 6 few-shot exemplars from description_scorer.py to the annotator prompt for calibrated scoring.

### `balancecc_prompt_generation`
- replace n-gram overlap metrics with an LLM-judge rubric (creativity, fidelity to editing type, fantasy-level calibration) since reference-based overlap penalizes valid creative variation

### `banner_request_400`
- Investigate why display_predictions contain generic/off-task responses for multiple models; verify MultimediaObject logo image is being sent and TASK_PROMPT is rendered before full run
- Replace single _RUBRIC_LLM_JUDGE_QUALITY with six AnnotatorSpecs (TAA/LPS/AQS/CTAE/CPYQ/BIS) using rubrics in annotator_notes.md, and corresponding MetricSpecs, to match paper
- Increase max_tokens well beyond 512 (paper outputs are multi-element JSON blueprints, likely 1500-3000+ tokens)

### `brainteaser`
- Set temperature=0 and either (a) remove stop_sequences / cap max_tokens=5 to force letter-only output, or (b) add a regex answer extractor before exact_match; current config will produce near-zero scores unrelated to model ability
- Add WP subset (second scenario or subset arg) or document SP-only scope explicitly; current benchmark covers half the paper
- Verify dataset split — confirm 'train' is the intended eval target on tasksource/brainteasers (HF mirror may collapse splits); switch to test split if available

### `calligrapher`
- Remove calligrapher from the text-only creativity sweep, or gate it behind an image-capable multimodal model list. Text LLMs cannot perform this task.
- Replace the LLM-judge rubric and metric with the paper's FID/CLIP/DINO/OCR pipeline on generated images; current rubric evaluates a task the scenario does not prompt for.

### `cap_aut`
- run scripts/score_cap_novelty.py (or equivalent) post-HELM before reporting cap_aut scores — otherwise only quality, not divergent-thinking novelty, is measured and the benchmark undersells AUT

### `cap_story`
- Recover the canonical Study 3 triads from the UVA pilot source (script is referenced but missing) and replace the placeholder list in scenarios/cap_story_scenario.py, or explicitly flag the items as pilot stand-ins in the scenario docstring.

### `chinese_homophonic_puns`
- Implement custom fuzzy-similarity metric (SequenceMatcher + fuzzywuzzy.fuzz.ratio, max) per evaluatePunchline.py; replace BasicGenerationMetric

### `conceptual_design`
- Replace lexical-overlap metrics (BLEU/ROUGE/F1/EM) with embedding-based novelty/diversity against AMT refs per paper, or remove them; they are meaningless for this task.
- Split LLM judge rubric into paper-aligned dimensions: novelty, feasibility, usefulness (each 1-5); drop conflated 'quality' score.

### `cpers`
- Replace single-score rubric with the 12-question TTCT scoring prompt from cpers_annotator_notes.md:104-161 and parse dimension averages + overall_creativity.
- Add (or swap in) a rhetorical device annotator emitting per-device binary labels per cpers_annotator_notes.md:165-185.

### `creai_cps`
- Regenerate/expand data.json to full 240-item set and de-duplicate scenarios before any scoring run.

### `creation_mmbench`
- Diagnose why predictions show 'no text provided' with base64_images=[]: verify that MultimediaObject ordering (images-first then text) is accepted by the Gemini client adapter and that image paths are reachable at inference time; add a smoke test that prints the rendered request for one instance
- Replace generic 1-5 rubric with instance-specific Creation-MMBench judge: pass per-instance criteria ('subjective requirement','groundtruth alignment'), images, reference answer, produce VFS (1-10) and Reward; use gpt-4o-0806
- Implement dual evaluation (swap response/reference positions) and aggregate VFS + Reward

### `creative_process`
- replace the generic creativity/originality judge with an embedding-based metric module: load thenlper/gte-large, compute cosine SS between consecutive items, hierarchical-cluster with cutoff 1.14 (VF), emit total_jumps / jump_rate / unique_categories / mean_SS / jump_profile_slope per response sequence — this is the paper's actual evaluation
- diagnose the identical-output issue (verify temperature is propagating to the client call, disable HELM response cache for this run, or bump num_outputs); without real sampling variance the 5-rep design yields nothing
- emit three run_specs (creative_process:task=vf, :task=aut_brick, :task=aut_paperclip) or make the scenario iterate all three tasks so aut_brick and aut_paperclip are actually run

### `creativemath`
- Raise max_tokens substantially (>=4096, ideally 8192) so competition proofs are not truncated; current 512-token cap makes correctness judgements meaningless for AIME/USAMO/IMO tiers.
- Replace the single Likert correctness judge with the paper's 3-stage pipeline (correctness YES/NO -> coarse novelty YES/NO -> fine novelty YES/NO) and compute the 5 ratio metrics. Either use LLMAsJuryAnnotator with 3 judges or explicitly document the single-judge adaptation and rename the metric.

### `creatset`
- Replace pointwise rubric with the verbatim Chinese pairwise judge prompt (creatset_annotator_notes.md:21-40); compare model output R against gen_resp_2 (GPT4o-mini-c) per-instance and report win-rate.

### `critics_story`
- Replace GenericLLMJudgeMetric 1-5 quality rubric with the CritiCS pairwise persona_comparision judge (Interesting/Coherent/Creative over A vs B pairs), or at minimum split the run into two RunSpecs — one for the 'generation' subset with a pairwise protocol against a baseline, and one for the 'judgment' subset using the scenario's exact_match ground truth that is currently orphaned.
- Wire the 'judgment' subset into an actual RunSpec (args={'subset': 'judgment'}) so the human-annotated Q1 labels in doc-storygen-v2 are used for evaluation — this is the only part of the benchmark with paper-aligned ground truth.

### `crowd_vote`
- Replace the generic llm_judge_quality rubric in run_specs/crowd_vote_run_specs.py with the 4-dimension rubric from metric_notes/crowd_vote_annotator_notes.md (originality, brand_relevance, creative_potential, conciseness on 1-5 scales) so the judge actually measures creativity rather than generic quality.

### `cs4`
- Replace generic creativity rubric with a per-constraint satisfaction annotator: feed story + numbered constraint list, require yes/no + quoted evidence per constraint, compute satisfied/total (cs4_annotator_notes.md:41-59).
- Raise max_tokens well above 512 (paper stories are long); 512 is causing systematic truncation of outputs before constraints can be satisfied.

### `dat`
- Swap SemanticDiversityMetric for a GloVe-840B-300d-based DAT scorer as specified in metric_notes/dat_annotator_notes.md:41-52 (avg pairwise cosine distance × 100 on first 7 valid-vocabulary unique words). Without this, reported DAT scores are not the DAT score — they're an embedding-model-specific diversity proxy and cannot be compared to the literature's 0-100 scale or the ~78 human mean.
- Investigate the identical-predictions-across-trials issue in display_predictions.json. Confirm whether HELM's request cache is keying only on prompt text (which is identical across all 100 trials by design) and make trial_idx part of the cache key, or disable cache for num_trials-style scenarios.

### `dat_creative_writing`
- Expand run_specs/dat_creative_writing_run_specs.py into nine separate run_spec_functions (or one parameterised via subset arg passed to ScenarioSpec) so the strategy-DAT and writing subsets actually execute. As written, 8 of 9 subsets never run.
- Replace SemanticDiversityMetric with the three spec metrics: GloVe-840B-300d DAT scorer for dat + dat_{strategies} subsets (matching metric_notes:15-50 and dat.yaml's recommendation), and DSI + Lziv for synopsis/flash_fiction/haiku (metric_notes:61-89). Current scorer does not measure any of the three.
- Fix the identical-predictions-across-trials caching issue (same root cause as dat.yaml): either salt HELM's request cache with instance_id or disable cache for this scenario. Without this, the num_instances=100 design produces 1 unique output, not 100.

### `data_narrative`
- Decide whether data_narrative is kept in the creativity suite at all (NOTES_FOR_VIJETA flags it borderline). If kept, at minimum add an LLM-judge annotator scoring the 4 paper dimensions; otherwise the headline metrics are unfaithful to the paper.

### `deep_math`
- Replace lexical BasicGenerationMetric with an LLM-as-judge annotator that applies the paper's lenient correctness rubric (core solution components, tolerate minor gaps) and emits a 0/1 or graded correctness score; populate metric_notes/*.md with the rubric.
- Raise max_tokens substantially (>=4096-8192) so proofs/constructions are not truncated mid-argument; current 512 cap invalidates most outputs.

### `dpt`
- Replace _RUBRIC_LLM_JUDGE_CREATIVITY in run_specs/dpt_run_specs.py:15-24 with the 5-dimension rubric from metric_notes/dpt_annotator_notes.md:22-58, and update GenericLLMJudgeAnnotator (or swap to a multi-dimension annotator) to parse and emit the 5 per-dimension scores plus Overall. Without this, the benchmark measures a generic creativity Likert, not the paper's construct.
- Change judge_model_name from 'openai/gpt-4' to 'openai/gpt-4o' (annotator_notes recommendation) or to gpt-4o-mini / claude-3.5-haiku to match the paper's reported baselines (Pearson r=0.74-0.76 vs human consensus). Current gpt-4 has no published calibration against the paper's human-rater consensus.

### `eqbench_creative_writing_v3`
- Either drop num_outputs to 1 (scenario already duplicates instances for 3 iterations) or collapse scenario to 32 instances and keep num_outputs=3 — do not do both. Current config produces 288 completions collapsed into 96 concatenated predicted_text fields, which corrupts per-iteration judging.

### `esp_dataset`
- Rewrite the prompt to include an explicit instruction (e.g., 'Write a caption for this image in the following style: {style_description}.'). Current bare '\n{style}:' yields 100% refusal outputs in the trial.
- Verify multimedia image delivery — refusal text ('haven't provided any text') suggests the image MediaObject may not be reaching the model via this adapter path; test with a known-working vision run_spec pattern.

### `fann_or_flop`
- Replace BasicGenerationMetric names with an Arabic-appropriate suite: wire an LLM judge annotator implementing the GPT-4o rubric in annotator_notes.md (faithfulness/fluency/overall 1-5, temp 0.0) plus BLEU, chrF++, and Arabic BERTScore against raw_explanation.
- Raise max_tokens substantially (e.g., 2048-4096) so outputs cover all verses; current 512 causes truncation visible in sample predictions.

### `fscg8`
- Remove fscg8 from the text-only creativity sweep, or gate it behind a diffusion/image-generation adapter. Text LLMs cannot meaningfully produce the visual outputs FSCG-8 evaluates.
- Replace the GenericLLMJudge 'story quality' rubric with the paper's FID/KID/Vendi/MSS/CLIP/SSCD pipeline on generated images, or drop the dataset. The current rubric is from an unrelated story-continuation task and cannot score this prompt type.

### `future_ideas`
- Implement IAScore and IDI as paper-primary metrics: add a SentenceBERT-based metric that (1) segments predicted_text into idea-sentences (newline+bullet or sent-tokenize), (2) computes mean cosine(sim(gen_idea_i, Future_work_ref)) for IAScore, (3) computes 1 - mean pairwise cosine sim across gen_ideas for IDI. Keep BLEU/ROUGE as secondary.
- Raise max_tokens well above 512 (observed truncation on EVERY sampled output in both gemini-2.5-flash and gemini-2.5-pro display_predictions.json). Suggest 2048 or model-max; idea-count is a dependent variable that must not be gated by the token budget.

### `futuregen`
- Add LLM-as-judge annotator scoring novelty, hallucination, feasibility (per paper) and wire it into metric_specs; document judge model + rubric in metric_notes files.
- Add BERTScore metric (helm BERTScoreMetric) to match paper's core automatic metrics.

### `fuxibench`
- Parameterize run spec by subset (accept subset arg, produce one RunSpec per subtask in SUBSETS) so couplet_gen/poem_gen/poem_nmt_inv/poem_appre actually run.
- Implement the paper's metrics: port FormatEvaluator/cipai_utils for pacc (ci_gen), couplet_format_acc for cacc (couplet_gen), SacreBLEU+jieba for BLEU (poem_appre). Without these, results are not comparable to FuxiBench leaderboard.
- For poem_gen/poem_nmt_inv, replace the generic 1–5 Likert judge with a binary Y/N lacc judge using the Figure-4 criteria (包含要点, 与标准答案一致, 符合问题和任务要求, 符合事实, 简洁明了), matching lacc_evaluation_api semantics. Prefer fine-tuned Qwen2-7B-Instruct or document the gpt-4o substitution explicitly.

### `gauss`
- Raise max_tokens to >=4096 (ideally 8192) so graduate-level proofs and open-exploration answers aren't truncated; 512 truncates 12a/12b across all observed runs and makes 12b's 'find multiple solutions' rubric unreachable.
- Replace the generic 0-7 Likert rubric with a per-instance judge prompt that threads problem_statement, standard_solution, rubric, and total_score from Instance.extra_data (as specified in metric_notes/gauss_eval_metrics_notes.md §'Judge Prompt Template'), and cap the score at total_score per problem. Current setup produces scores exceeding the problem's total_score.

### `geo_story`
- Implement the paper's three metrics as HELM metrics/annotators: (a) corpus-wide IDF uniqueness (stopword-filtered, lowercase, score*total_docs normalization per measure_uniqueness.py); (b) spaCy en_core_web_trf NER count of LOC/FAC/GPE minus the anchor city/country (per measure_informativeness.py); (c) GPT-4 emotion tagging with the exact prompt 'Help me identify which of the following emotions: Joy, Hardships, Fear, Sadness, Serenity; are recognized within the story given in the prompt. Only output the names of the emotions found in the prompt.' at temp=0.7, max_tokens=50.

### `grapheval_ai_researcher`
- Replace llm_judge_quality with a classification metric: compare predicted decision (last line of output, strip whitespace) against Reference CORRECT_TAG; compute per-class accuracy and macro-F1. This matches the paper's reported metric and uses the ground-truth label already attached.
- If an LLM judge is retained at all, rewrite the rubric to inject the gold decision and score binary correctness + near-miss (Poster vs Oral) rather than generic `quality`.

### `grapheval_iclr`
- Add a classification metric that parses predicted_text with a regex like r'(Reject|Accept \(Poster\)|Accept \(Oral\)|Accept \(Spotlight\))' and compares to the reference decision. Report accuracy + macro-F1 (and optionally macro-precision/recall) to match the paper's reported metrics. This is the authoritative evaluation for this scenario.
- Lower temperature from 0.7 to 0.1 to match the paper's hyperparameter setting. At 0.7 the 4-class decision is noisy and non-comparable to paper-reported baseline numbers.

### `grapheval_review_advisor`
- Replace GenericLLMJudgeMetric with classification metrics that consume the gold decision reference (ExactMatchMetric and/or QuasiExactMatch plus per-class / macro F1) so the run reproduces the paper's accuracy/precision/recall/F1. Strip the predicted_text to the decision line before matching.
- Remove or repurpose the AnnotatorSpec / _RUBRIC_LLM_JUDGE_QUALITY block, or re-scope it to a secondary diagnostic. The current rubric scores a construct ('advisory helpfulness') that is orthogonal to the task and actively misranks models that get the decision right but differ in surface wording.

### `graphrag_bench`
- Wire an LLM-as-judge annotator for answer_correctness and coverage_score (RAGAS-style, gpt-4-turbo or substitute) to match the paper's metrics; BasicGenerationMetric alone will not yield interpretable scores for open-ended creative writing.

### `historical_analogy`
- Align the rubric scale to paper's 1-4 with anchors from metric_notes/historical_analogy_annotator_notes.md:36-46, or explicitly document the 1-5 deviation. Current 1-5 scores are not comparable to paper results.
- Add stop_sequences=['==== case', '\nInput Event:'] (or remove the one-shot example) in run_specs/historical_analogy_run_specs.py:46 to prevent models from continuing the few-shot format and regurgitating next-case inputs as their 'answer'.

### `hypobench`
- Parameterize the run_spec over VALID_TASKS — generate 7 run_specs (one per real-world task) via run_spec_function args, so hypobench actually covers its 7 real tasks instead of only deceptive_reviews. Update eval_scripts/hypobench.sh accordingly.
- Either (a) implement the paper's two-step Accuracy pipeline (generate hypothesis, then run second LLM pass to classify held-out test examples with that hypothesis), or (b) register the documented proxy metrics — ROUGE-L + BERTScore vs. known_hypotheses — via HELM's open_ended MetricSpecs. Current novelty/significance/verifiability scores do not correspond to any paper metric for HypoBench.
- Replace the scientific-hypothesis rubric text (run_specs:15-46) with task-appropriate criteria, OR drop the LLM-judge metrics entirely in favor of the paper's pipeline. Scoring 'deceptive reviews' decision rules on 'scientific breakthrough significance' is a category error.

### `hypogen`
- Fix the paper citation in scenarios/hypogen_scenario.py — replace arXiv:2409.04109 with the correct HypoGen source paper, or remove the misleading citation if provenance is unverified.
- Verify the bit/flip role semantics against the original HypoGen paper PDF. If HF card is correct and scenario prompt inverts the roles, the prompt and reference mapping must be swapped before any results are meaningful.
- Either add BLEU/ROUGE (BasicGenerationMetric) to match the scenario's own docstring promise, or update the docstring to stop promising metrics the run_spec doesn't compute.

### `idrbench`
- Rewrite metric_notes/idrbench_annotator_notes.md to describe the correct paper (arXiv:2507.15736, Shen et al.) — the current notes reference a different IDRBench (2601.06676) and will mislead reviewers.
- Replace BasicGenerationMetric with a task-appropriate metric: parse 'Your verdict:' line from predicted_text, compare to y_true, and report accuracy + macro-F1 for IPI and I3; parse 'Your choice:' for I2R. Current exact_match/BLEU/ROUGE on free-text will report ~0 and is not informative.
- Fix I2R gold-label assignment in scenarios/idrbench_scenario.py:443-450. Match candidates[0/1] against target_paper['id'] explicitly rather than always tagging Paper 1.

### `ii_bench`
- Diagnose why predicted_text is identical and base64_images is empty across all instances — image + question text are not being delivered. Verify MultimediaObject path in scenarios/ii_bench_scenario.py:119-128 actually flows through the ADAPT_MULTIPLE_CHOICE_JOINT adapter to the Gemini backend; consider ADAPT_GENERATION_MULTIMODAL or a vision-aware MC adapter.
- Resolve the prompt-assembly collision: the scenario writes a full 'question + A-F options + Answer:' prompt, but the adapter_spec also sets output_prefix='Answer: ' and uses MC_JOINT formatting that rewrites references. Choose one path (either scenario-built prompt with ADAPT_GENERATION, or let the adapter build the MC prompt from Instance.references) — current setup appears to strip the question body.

### `infochartqa`
- EXCLUDE infochartqa from the creativity benchmark launch per NOTES_FOR_VIJETA.md contamination flag. Task is multimodal chart QA with exact_match metric - orthogonal to creativity. Drop the run_spec or move it to a separate 'visual_reasoning_controls' suite if retention is desired for discriminant validity.

### `irfl`
- Fix position bias: actually shuffle all_uuids with a per-instance seed and set correct_index to the post-shuffle position; current code gives every instance answer=A
- Diagnose why base64_images is empty in display_predictions.json — MediaObject(location=image_path) is not being encoded/sent to Gemini. Likely a HELM multimedia adapter path issue; without this the benchmark measures text-only refusal, not figurative-image understanding
- Expose config as a run_spec argument and register separate RunSpecs (or a single parameterized one) for idiom, metaphor, simile, open-simile; current setup covers 200/1087 examples
- Set temperature=0 and either cap max_tokens=3 or add a regex extractor for [ABCD] before exact_match; append explicit 'Answer with a single letter (A, B, C, or D).' to the prompt

### `layoutsam_eval`
- Either (a) drop layoutsam_eval from the text-only creativity sweep — it is a layout-to-image generation benchmark and text LLMs cannot produce images, or (b) explicitly relabel this as a new 'layout-to-caption' task (not LayoutSAM-Eval) and document that divergence.
- If kept, replace the generic LLM-judge rubric with a reference-aware metric (ROUGE-L/BLEU against global_caption as the scenario docstring already promises, plus a judge that sees the gold caption and the region-wise attribute/spatial claims). Current judge ignores references.

### `litbench`
- Replace GenericLLMJudgeMetric with a pairwise-accuracy exact-match metric that parses the model's output for A/B and compares to the CORRECT_TAG reference. This is the paper's primary metric.
- Remove the GenericLLMJudgeAnnotator entirely (or repurpose it only if running a judge-as-model study). The current generic quality rubric is orthogonal to LitBench.
- Fix adapter: set temperature=0.0, max_tokens<=8, and either drop output_prefix='Answer: ' or rewrite PROMPT_TEMPLATE so the final instruction matches the expected answer surface. Verify the model actually emits 'A' or 'B'.

### `liveideabench`
- Replace the single llm_judge_creativity annotator with a multi-dimensional annotator that issues the exact critic prompt from metric_notes/liveideabench_annotator_notes.md:14-32, parses the JSON block, and emits originality/feasibility/clarity on 1-10.
- Add the pairwise fluency annotator (keyword-grouped A/B/C/D comparator, map to 10/7/4/1) and the post-hoc flexibility metric (30th percentile across dims) to match the paper's five headline scores.
- Switch judge_model_name from openai/gpt-4 to the disclosed CRITIC_MODELS panel (claude-3.5-sonnet, gpt-4o, qwen-2.5-72b, deepseek-chat, gemini-2.0-flash-thinking) via jury annotator, or at minimum document a single-judge adaptation and note non-comparability to the LiveIdeaBench leaderboard.

### `llm4biohypogen`
- Implement and register LLMAsJuryAnnotator with GPT-4 (or GPT-4o) judge applying the 4-dimension 0-3 rubric from metric_notes/llm4biohypogen_annotator_notes.md:42-66 and expose 4 per-instance scores plus their mean as metrics. Without this, the run does not evaluate the paper's primary metrics.
- Expand coverage to all 4 splits by instantiating 4 run specs (gpt-3.5/seen, gpt-3.5/unseen, gpt-4/seen, gpt-4/unseen) or by parameterizing get_llm4biohypogen_spec to accept model_version/test_type; otherwise unseen-contamination analysis from the paper cannot be reproduced.

### `llm_discussion`
- Pass args={'test': 'all'} in run_specs/llm_discussion_run_specs.py so all 120 items across AUT/Similarities/Instances/Scientific are evaluated, matching the paper; current AUT-only run reproduces ~25% of the benchmark.
- Wire an LLMAsJuryAnnotator with GPT-4 (or GPT-3.5-turbo) that scores each prediction on Fluency (int), Flexibility (int), Originality (1-5), and Elaboration (1-5) per metric_notes/llm_discussion_annotator_notes.md:57-103. Without it, llm_discussion results cannot be compared to the paper.

### `macgyver`
- Parameterize run_specs/macgyver_run_specs.py to accept subset and prompt_strategy args and emit separate RunSpecs per (subset × strategy); at minimum run vanilla on all three subsets and vanilla/divergent_convergent/reflection on the full set to reproduce Table 3 of the paper
- Remove exact_match/quasi_exact_match/f1_score from metric_specs — they are actively misleading on open-ended generations; retain rouge_l/bleu_4 only as auxiliary signals and promote llm_judge to primary
- Redesign judge rubric to emit the 8 categorical labels from metric_notes/macgyver_annotator_notes.md (or at least the binary solvable-correctly-identified flag for unsolvable subset); 1-5 scalar cannot separate 'correctly says unsolvable' from 'wrong solution'

### `matdesign`
- Convert the upstream Excel dataset (Materials Discovery & Design Dataset.xlsx, 50 rows) to JSON, host it at a stable URL, and make MatDesignScenario.get_instances download it — remove the /tmp/ local path hack and the 1-row hardcoded fallback. Until this is fixed the benchmark is testing a single example.
- Raise max_tokens well above 512 (20 suggestions × ~100 tokens each ≈ 2000+ tokens minimum; use 4096–8192) so the 20-suggestion JSON is not truncated. Currently every model output is structurally invalid JSON due to mid-generation cutoff.
- Either implement the paper's AccelMat evaluation (3-critic consensus on YES/NO per suggestion, iterative refinement, final o1-preview scoring on the 3 closeness + 6 quality rubrics) OR explicitly document that HELM is using a simplified single-judge quality score and update the rubric to match the paper's constraint-satisfaction framing (e.g., "fraction of 20 suggestions that meet goal + all constraints strictly"). Current generic 1–5 design-quality rubric is unrelated to the paper.

### `meta4xnli`
- Register separate RunSpecs for each of the five SUBSETS (interpretation_en, interpretation_es, interpretation_en_cot, interpretation_es_cot, detection_en) via a subset arg; current config evaluates 1/5 of the declared benchmark
- Switch adapter from ADAPT_MULTIPLE_CHOICE_JOINT to ADAPT_GENERATION so the scenario-built Table-29 prompt is sent verbatim without a synthetic A/B/C letter block appended; keep References as label strings for exact_match
- Set temperature=0, max_tokens~=8, and restrict stop_sequences to ['\n','.'] for label classification; 0.7/512 is incorrect for deterministic NLI
- Either drop detection_en from the benchmark or flag it explicitly as an 'off-paper generative variant' in run_specs + docs so detection F1 is not compared to the paper's fine-tuned BIO numbers

### `metaphor_generation`
- Add an explicit instruction (e.g., 'Rewrite the following literal sentence as a metaphorical sentence by replacing the main verb. Output only the rewritten sentence.') in adapter_spec.instructions or as a prompt prefix in the scenario
- Reduce max_tokens (e.g., 64) and/or add stop_sequences=['\n'] so outputs are single sentences comparable to the 156 one-line references

### `mops`
- Rewrite rubrics to 0-100 integer scale matching metric_notes/mops_annotator_notes.md and the paper, and update the metric parsing to accept a single integer rather than a 1-5 choice. Remove the 'movie premise / existing films' framing; use 'story premise' consistently.
- Add the completeness dimension as a third annotator + metric (rubric from metric_notes/mops_annotator_notes.md:26-32), scoring whether character, setting, event, ending, and twist are present.
- Remove BasicGenerationMetric (exact_match/quasi_exact_match/f1_score/rouge_l/bleu_1/bleu_4) or mark all six as non-primary diagnostic metrics. They do not reflect the MoPS evaluation and will dominate leaderboard sort order spuriously.

### `music_theory_bench`
- Set temperature=0 (or 0.0) for deterministic MC scoring; 0.7 injects noise into a task with a single correct letter.
- Remove stop_sequences=['\n'] OR switch adapter to ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL (likelihood-based scoring per choice) so free-form CoT does not break exact_match. Current config lets models emit 500 tokens of prose that never yield a parseable 'A'/'B'/'C'/'D' against the single-letter references.
- Audit abc_score field handling: inspect dataset rows where stem references a score and confirm whether to inline abc_score into the prompt (scenarios/music_theory_bench_scenario.py:102-104 currently skips it).

### `ocw`
- Replace GroupMatchScoreMetric with a correct implementation: parse prediction into 4 groups (split on newline, then comma), strip 'Group X:' prefixes and connection annotations, compare as sets-of-frozensets against reference groups, report groups_correct (0-4) and wall_solved (0/1) as separate MetricNames per metric_notes/ocw_eval_metrics_notes.md
- Current Jaccard metric returns ~1.0 for any output that echoes the 16 clues and ~0 for malformed output — scores are near-uninterpretable; any headline numbers from this run should be discarded and recomputed after metric fix

### `outline_to_story`
- Decide and document: is this task 'prompt-to-story' (current impl) or faithful 'outline-to-story' (paper)? If the latter, preprocess references to extract cascaded per-paragraph keyword outlines (RAKE or similar) and feed those instead of the raw prompt. If the former, rename the scenario/group to avoid implying O2S-paper reproduction.

### `permpst`
- Implement a custom regression/correlation metric that (a) parses predicted Score from the ```json {"Score": N} ``` completion using the existing _extract_score_from_completion logic, (b) parses reference score from Reference.output.text (already numeric), and (c) aggregates Pearson r, Spearman rho, and Kendall-Tau over the full set (plus per-reviewer if reviewer id is preserved). Replace BasicGenerationMetric with this metric for primary scoring.
- Remove the LLM judge entirely for PerMPST, or replace its rubric with a reviewer-alignment rubric that takes both the reviewer history and the ground-truth reviewer score/review as inputs. The current _RUBRIC_LLM_JUDGE_QUALITY references 'permuted sentence translation / reconstruction' and is orthogonal to this task.
- Populate metric_notes/permpst_annotator_notes.md (currently empty) or delete the AnnotatorSpec. Orphaned empty docs plus a wrong-task rubric are a reproducibility hazard.

### `poetmt`
- Add an output-extraction step (scenario post-processor or annotator preprocessor) that isolates the English translation from the model's surrounding commentary before BLEU/ROUGE/judge scoring, or reframe the prompt to forbid commentary ('Output only the English translation, no preamble or analysis.').
- Raise max_tokens to at least 1024 (preferably 1536) so discourse-level Tang regulated verse does not truncate mid-poem; currently sampled outputs are cut off in the translation itself.
- Update judge prompt to include the source Chinese poem and (optionally) the reference translation as context, as documented in metric_notes/poetmt_eval_metrics_notes.md:64-81 and implied by the paper. Scoring BS/BF/BM on an English text with no source grounding is not what the paper measures.

### `pollux_creativity`
- Replace the two generic English rubrics with the exact POLLUX judge prompt (annotator_notes:190-215) using per-instance criteria_name and rubrics from extra_data, and switch the 1-5 scale to 0-4. Require a parser that extracts integer between [RESULT] and [END].
- Either switch judge_model_name to ai-forever/pollux-judge-32b-r (preferred) or keep GPT-4 but supply the Russian-language POLLUX prompt template. openai/gpt-4 with English creativity/originality rubric is not a POLLUX evaluation.

### `pron_vs_prompt`
- Raise max_tokens to ~1200-1500 so models can produce the ~600-word synopsis the prompt asks for; current 512 guarantees truncation and invalidates literary-quality judgments.
- Rewrite judge rubrics in run_specs/pron_vs_prompt_run_specs.py to match the paper's 0-3 scale, novel/literary (not 'promotional/advertising') framing, and add the missing Relevance (0-4) and Literary-quality (0/1 anthology + own_voice) dimensions from annotator_notes.md:20-89.

### `pun_eval`
- Replace generic 5-point rubric with paper's verbatim binary pun-detection prompt (metric_notes/pun_eval_annotator_notes.md:27-41); parse Choice field and score 0/1
- Add separate run_spec for task='explanation' (or parameterize) — scenario supports it but no spec exercises the explanation branch

### `puzzleworld`
- Diagnose why Gemini returned generic 'no text provided' responses — verify MultimediaObject with image/png + text/plain is being serialized to the Gemini client; base64_images=[] in output suggests HELM is not attaching images at the client layer
- Add 'Answer:' regex extractor before exact_match (e.g., post-process to capture text after final 'Answer:' token) — otherwise all scores will be ~0 regardless of model correctness
- Set temperature=0 for reproducible final-answer scoring; raise max_tokens to >=2048 to accommodate the CoT the system prompt demands

### `recombination_extraction`
- Implement a custom metric (e.g., metrics/recombination_extraction_metric.py) that extracts <answer> JSON, normalizes analogy<->inspiration key aliases, and computes: classification accuracy/F1 (empty vs non-empty), entity soft-F1, relation soft-F1 per paper Section 4
- Add GPT-4o-mini annotator for soft entity matching (paper Appendix B); without it F1 at entity/relation levels is not faithful
- Align gold reference keys with prompt vocabulary (map dataset 'analogy'* -> 'inspiration'*) in scenario before emitting Reference, or handle bidirectionally in metric
- Raise max_tokens to >=1024 (scratchpad + JSON answer); current 512 truncates a majority of predictions before <answer> closes
- Set temperature=0 and max_train_instances=0 — PROMPT_E2E is self-contained zero-shot in the paper; few-shot prepending breaks the prompt template

### `research_idea_execution`
- Rename scenario or re-scope: either (a) rebuild around the actual execution study (4-page executed papers from Execution_Study_Data.zip, reviewer panel on 5-dim 1-10) — matching paper's namesake — or (b) rename to 'research_idea_peer_review' and align annotator rubrics with the 5-dim 1-10 ideation rubric the prompt already uses.
- Replace the three generic 1-5 annotators (novelty/feasibility/quality of an 'execution plan') with annotators that grade the generated *peer review* — either the paper's 5-dim 1-10 calibration check or the 4-dim review-quality rubric already documented in metric_notes/research_idea_execution_annotator_notes.md:14-64.
- Fix judge rubric wording in run_specs/research_idea_execution_run_specs.py:15-46: current text evaluates an 'execution plan' artifact not produced by the task; update to grade review calibration/reasoning on the actual 5-dim peer review output.

### `scar`
- Add MetricSpec for metrics.f1_metric.F1Metric to run_specs/scar_run_specs.py so the 'f1' stat required by registry_metrics.yaml is produced (diagnosis/fix already drafted in debugging_scripts/metrics-check/clin/scar_fixes.md)

### `science_analogies`
- Replace basic_metrics bundle with paper-aligned eval: either (a) an LLM-judge rubric scoring meaningfulness/novelty/soundness/comprehensibility on 1-5 per paper's AMT dimensions, or (b) a precision metric matching predicted analogy against short reference Explanation (paper's automatic metric). Current exact_match/quasi_exact_match will be ~0 for all models and f1/rouge/bleu against a 1-sentence reference is misleading

### `scimon`
- Populate metric_notes/scimon_eval_metrics_notes.md and scimon_annotator_notes.md with paper-aligned rationale. At minimum document (a) why rel_sent is the reference target and not "output", (b) which paper-reported metrics are being approximated, (c) acknowledged gap vs. the paper's human novelty/relevance/technical-depth eval.
- Add a novelty-sensitive metric. Either wire an LLM-as-judge rubric scoring novelty/relevance/technical-depth against the background context (paper's own criteria), or compute n-gram/embedding novelty against the background context as a divergence-from-source proxy. Current metric set cannot distinguish "restated the context" from "proposed a new idea".

### `sdat`
- Implement SemanticDistanceMetric using ibm-granite/granite-embedding-278m-multilingual (per metric_notes/sdat_eval_metrics_notes.md:41 and run_specs/sdat_run_specs.py:35-36 TODO). Replace the exact_match MetricSpec. Without this, the 'sdat' column in results is measuring nothing task-relevant.
- Add a word-extractor covering the three formats in metric_notes:52-67 (numbered list, bullets, comma-separated), with the edge-case handling at metric_notes:124-143 (wrong count → pad/penalty, duplicates → natural 0 self-distance, OOV, non-compliant). Unit-test on the exact markdown-numbered form observed in display_predictions.json.
- Fix the identical-response-across-100-instances issue (same root cause as dat.yaml). Either salt HELM's request cache with instance/trial_idx for repeated-prompt scenarios or disable cache for sdat. Without this fix, 100 instances collapse to 1 sample and the statistical-reliability rationale in sdat_scenario.py:87-98 is void.

### `showerthoughts`
- Switch rubric anchors to 6-point scale and replace hand-written rubrics with paper's verbatim prompts (metric_notes/showerthoughts_annotator_notes.md:44-93) so scores map to paper Table 2.
- Add Logical Validity and General Score annotators (annotator_notes.md:17-21) — currently only 3/5 paper dimensions evaluated.
- Redesign task to avoid 300 identical outputs: either (a) generate N showerthoughts in a single call matching paper's Section 4.1 prompt then split, or (b) vary seed/topic per instance. Current setup produces 1 unique generation judged 300x.

### `simile_generation`
- Tighten prompt to force a single-line output, e.g., 'Rewrite the following literal sentence as a simile by replacing the bracketed phrase with a like/as comparison. Output only the rewritten sentence, nothing else.' and preserve (do not strip) the brackets so the model knows which span to rewrite
- Reduce max_tokens (e.g., 64) and add stop_sequences=['\n','\n\n'] so outputs are single sentences comparable to the one-line Human1/Human2 references; current 512-token outputs defeat BLEU/ROUGE

### `slang_generation`
- Add a semantic-novelty metric implementing novelty.py (SBERT all-mpnet-base-v2 Euclidean distance between parsed slang definition and WordNet/dictionary senses of the parsed slang word) — this is the paper's primary metric and is currently unimplemented

### `speak_to_structure`
- Replace generic ValidityMetric + LLM judge with RDKit-based metrics: (1) SMILES validity via Chem.MolFromSmiles, (2) per-subtask Success Rate using paper's SMARTS tables / atom/bond counting / Tanimoto / property deltas, (3) Novelty for MolCustom, (4) Tanimoto for MolEdit/MolOpt. Without this, results are not TOMG-Bench scores.
- Emit per-subtask metric groupings (MolCustom.AtomNum, MolCustom.FunctionalGroup, MolEdit.AddComponent, MolOpt.LogP, ...) rather than a single aggregate.

### `splat`
- Replace n-gram metrics with an LLM-as-judge annotator that grades final-scenario correctness against the reference answer (binary or graded), matching the paper's protocol; document the judge prompt in metric_notes/splat_annotator_notes.md
- Either (a) explicitly document and justify the single-turn simplification as a deliberate deviation from the paper's multi-turn framework, or (b) implement the multi-turn player-judge loop; current silent divergence misrepresents what is being measured
- Raise max_tokens (≥1024) or add a prompt constraint requesting concise answers; ~20% of sampled predictions are mid-sentence truncated

### `ss_gen`
- Add annotators for remaining 4 dimensions (Descriptiveness, Empathy, Grammaticality, Relevance) to match paper's 5-dimension GPT-4 eval, or document that only Coherence is scored.

### `story_generation_rocstories`
- Resolve name/scope contradiction: either (a) pass args={'dataset':'roc'} in run_specs/story_generation_rocstories_run_specs.py so the scenario matches its name, or (b) rename the scenario/group to 'story_generation' (roc+wp) and add a sibling 'story_generation_writingprompts' run spec. Current setup silently mixes corpora.
- Raise max_tokens significantly (e.g., 1024-2048) AND/OR set explicit generation_config for reasoning models to reserve output budget. At 512 most Gemini-3 outputs truncate to ~20 tokens with finish_reason=length, yielding degenerate references metrics. Re-run trial after the change and confirm stories are complete.

### `sudoku_bench`
- Raise max_tokens to >=8192 (or a reasoner-appropriate limit) so the model can emit reasoning plus the 81-digit <ANSWER> block; current 512 guarantees 0% solve rate
- Add an <ANSWER>(.+?)</ANSWER> regex post-processor and strip whitespace/newlines before exact_match; without extraction scores will be 0 regardless of correctness
- Set temperature=0 for reproducible single-shot solve-rate scoring to match the paper's evaluation protocol

### `ttcw`
- Decide the intended use: either (a) reframe as a creative-writing generation benchmark where the model writes a story and an LLM-as-judge applies the 14 TTCW questions (requires a judge pipeline, not a classification metric), or (b) keep as a judge-alignment probe but acquire full story texts (the dataset provides URLs) and document it explicitly as a judgment task, not a creativity task.
- Fix the adapter/prompt collision: either remove the custom 'Answer:' suffix from scenarios/ttcw_scenario.py and let ADAPT_MULTIPLE_CHOICE_JOINT format everything, or switch to ADAPT_GENERATION with exact_match and strip the MC prefixes.
- Set temperature=0 for the binary prediction adapter.

### `writingbench`
- Replace the static generic rubric with a checklist-aware annotator that templates the per-instance 5 criteria (name + 1-2/3-4/5-6/7-8/9-10 rubrics from extra_data.checklist) into the judge prompt matching annotator_notes.md lines 71-130, emits 5 JSON {score, reason} objects per instance on 1-10, and aggregates mean across the 5. This is the paper's defining methodological contribution and is currently bypassed.
- Decide at project level: (a) restrict to Literature & Arts via `args={'domain': 'literature'}` for 183 creative queries, or (b) keep full 1,000 and document that 817/1000 instances are professional writing, not creativity. Current default runs the contaminated full set. See NOTES_FOR_VIJETA.md section 1 for team discussion context.

### `yesbut`
- Diagnose why predicted_text is identical and base64_images is empty across all instances — image and likely prompt text are not being delivered. Verify the MultimediaObject in scenarios/yesbut_scenario.py:107-116 actually flows through ADAPT_GENERATION to the Gemini backend; same root-cause as ii_bench. Test on 2-3 instances before scaling to 1,084.

## DIVERGENT (55)

| Benchmark | Prompt | Data | Metric | Judge | Output | Paper |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `alpaca_eval_2` | false | true | false | partial | false | https://arxiv.org/abs/2404.04475 |
| `arena_hard_creative` | true | true | partial | partial | true | https://arxiv.org/abs/2406.11939 |
| `arena_hard_v01` | partial | true | false | partial | false | https://github.com/lmarena/arena-hard-auto |
| `artinsight` | partial | true | false | false | false | https://arxiv.org/abs/2502.19263 |
| `balancecc_prompt_generation` | partial | true | false | n/a | true | https://arxiv.org/abs/2309.16496 |
| `banner_request_400` | partial | true | false | partial | false | https://arxiv.org/abs/2503.11060 |
| `brainteaser` | partial | false | false | n/a | false | https://arxiv.org/abs/2310.05057 |
| `calligrapher` | partial | true | false | false | false | https://arxiv.org/abs/2506.24123 |
| `conceptual_design` | partial | true | false | partial | true | https://arxiv.org/abs/2306.01779 |
| `cpers` | true | partial | false | false | true | https://arxiv.org/abs/2509.18401 |
| `creai_cps` | true | false | partial | partial | false | https://github.com/Beaty-Lab/CREAI-item-generation |
| `creation_mmbench` | partial | true | false | partial | false | https://arxiv.org/abs/2503.14478 |
| `creative_process` | true | true | false | false | partial | https://arxiv.org/abs/2405.00899 |
| `creatset` | true | true | false | false | true | https://arxiv.org/abs/2505.19236 |
| `crowdcounter` | unknown | unknown | unknown | unknown | unknown | https://arxiv.org/abs/2410.01400 |
| `cs4` | true | true | false | false | true | https://arxiv.org/abs/2410.04197 |
| `deep_math` | true | true | false | false | partial | https://arxiv.org/abs/2505.08744 |
| `discovery_bench` | unknown | unknown | unknown | unknown | unknown | https://arxiv.org/abs/2407.01725 |
| `esp_dataset` | partial | true | partial | n/a | false | https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Fu… |
| `fann_or_flop` | partial | true | false | false | true | https://arxiv.org/abs/2505.18152 |
| `fscg8` | partial | true | false | false | false | https://arxiv.org/abs/2408.02226 |
| `futuregen` | true | true | false | false | true | https://arxiv.org/abs/2503.16561 |
| `grapheval_ai_researcher` | true | true | false | false | true | https://arxiv.org/abs/2503.12600 |
| `grapheval_review_advisor` | true | true | false | false | true | https://arxiv.org/abs/2503.12600 |
| `graphragbench-wrongone` | unknown | unknown | unknown | unknown | unknown | — |
| `humor_transfer` | true | true | false | true | false | https://arxiv.org/abs/2508.19402 |
| `hypobench` | partial | partial | false | false | partial | https://arxiv.org/abs/2504.11524 |
| `hypogen` | unknown | true | false | false | true | https://arxiv.org/abs/2409.04109 |
| `ii_bench` | partial | partial | true | n/a | false | https://arxiv.org/abs/2406.05862 |
| `infochartqa` | true | true | true | n/a | true | https://arxiv.org/abs/2505.19028 |
| `irfl` | partial | true | false | n/a | false | https://arxiv.org/abs/2303.15445 |
| `layoutsam_eval` | false | true | false | false | true | https://arxiv.org/abs/2412.03859 |
| `litbench` | partial | true | false | false | false | https://arxiv.org/abs/2507.00769 |
| `llm4biohypogen` | true | partial | false | false | true | https://arxiv.org/abs/2407.08940 |
| `macgyver` | partial | true | partial | partial | true | https://arxiv.org/abs/2311.09682 |
| `matdesign` | true | false | false | false | false | https://arxiv.org/abs/2501.13299 |
| `meta4xnli` | partial | true | partial | n/a | true | https://arxiv.org/abs/2404.07053 |
| `metaphor_generation` | false | true | partial | n/a | false | https://aclanthology.org/2021.naacl-main.336/ |
| `mops` | partial | true | false | false | true | https://arxiv.org/abs/2406.05690 |
| `newyorker_humor` | true | true | partial | n/a | true | https://arxiv.org/abs/2209.06293 |
| `noveltybench` | unknown | unknown | unknown | unknown | unknown | https://arxiv.org/abs/2504.05228 |
| `ocw` | true | partial | false | n/a | true | https://arxiv.org/abs/2306.11167 |
| `permpst` | true | true | false | false | true | https://arxiv.org/abs/2310.03304 |
| `puntuguese` | partial | true | partial | n/a | false | https://aclanthology.org/2024.lrec-main.1167/ |
| `puzzleworld` | true | true | partial | n/a | false | https://arxiv.org/abs/2506.06211 |
| `rebus_puzzle` | unknown | unknown | unknown | unknown | unknown | https://arxiv.org/abs/2505.23759 |
| `recombination_extraction` | true | true | false | false | partial | https://arxiv.org/abs/2505.20779 |
| `scimon` | unknown | true | partial | n/a | true | https://arxiv.org/abs/2305.14259 |
| `sdat` | true | true | false | n/a | partial | https://arxiv.org/abs/2505.09068 |
| `simile_generation` | partial | true | partial | n/a | false | https://aclanthology.org/2020.emnlp-main.524/ |
| `splat` | false | true | false | n/a | partial | https://arxiv.org/abs/2410.06733 |
| `sudoku_bench` | partial | true | partial | n/a | false | https://arxiv.org/abs/2505.16135 |
| `ttcw` | false | partial | partial | false | partial | https://arxiv.org/abs/2309.14556 |
| `unfun_corpus` | partial | true | false | n/a | false | https://arxiv.org/abs/2403.00794 |
| `yesbut` | true | partial | partial | n/a | false | https://arxiv.org/abs/2409.13592 |

### `alpaca_eval_2`
_Current implementation does not match the AlpacaEval 2.0 paper: it runs a 1-5 Likert single-response quality rating rather than a pairwise binary preference against a baseline, omits the paper's headline LCWR metric, and probably uses the wrong baseline. Per NOTES_FOR_VIJETA.md, this benchmark is already flagged as borderline-creativity (general instruction-following); faithfulness issues here are substantive and should be fixed before the production run or results should not be reported as AlpacaEval 2.0._

**Gaps:**
- run_specs/alpaca_eval_2_run_specs.py:15-24 defines a 1-5 Likert rubric (_RUBRIC_WIN_RATE), but AlpacaEval 2.0's paper (Dubois et al. 2024) uses a binary pairwise preference between model output and a GPT-4-turbo baseline, not a 1-5 quality rating.
- run_specs/alpaca_eval_2_run_specs.py:50 uses GenericLLMJudgeMetric with metric_name='win_rate', but the scored output (display_predictions.json annotations: win_rate in {1..5}) is a Likert score, not a win/loss. True AlpacaEval WR = fraction of instances where model beats baseline.
- The judge prompt embedded by GenericLLMJudgeAnnotator does not match the official AlpacaEval pairwise template documented in metric_notes/alpaca_eval_2_annotator_notes.md:15-36 (JSON-formatted ranking of model_1 vs model_2). The reference baseline from the dataset is never shown to the judge as 'model_2'.
- Length-Controlled Win Rate (LCWR), the paper's headline metric and the point of the paper, is not computed anywhere. No GLM fit, no length-difference feature, no lcwr field in annotations.
- scenarios/alpaca_eval_2_scenario.py:91 comment claims baseline is 'text_davinci_003 or gpt4', but AlpacaEval 2.0 specifically uses GPT-4-turbo baseline outputs; the downloaded alpaca_eval.json from tatsu-lab default is text_davinci_003 reference. Baseline identity is ambiguous and likely wrong for 2.0 semantics.
- adapter_spec (run_specs:45) uses temperature=0.7 for generation; AlpacaEval convention is model-default sampling for the evaluee but this is borderline, not strictly specified. Judge temperature=0.0 and max_new_tokens=1024 are reasonable.
- Sample predictions (benchmark_output/runs/gemini_lite/alpaca_eval_2_model=google_gemini-2.5-flash-lite/display_predictions.json) show many responses truncated mid-sentence at 512 tokens (max_tokens=512), which disadvantages models on long-form prompts and distorts any length-sensitive metric.

### `arena_hard_creative`
_Scenario loading and 250-instance creative_writing filter look correct and predictions have the expected shape with annotations attached. Principal divergence is methodological: this is a single-response Likert eval labeled 'win_rate', not the paper's pairwise ELO protocol. Rubric text and annotator notes disagree on scale and dimensions._

**Gaps:**
- Original Arena-Hard is pairwise (win/tie/loss vs baseline) with GPT-4.1/Gemini-2.5; run_specs uses single-response scoring with openai/gpt-4-turbo as judge (arena_hard_creative_run_specs.py:57). Scenario docstring acknowledges divergence (arena_hard_creative_scenario.py:33-34).
- Rubric in run_specs is a 1-5 comparative scale referencing a 'reference' that is never passed to the judge (arena_hard_creative_run_specs.py:15-24); annotator_notes.md specifies a 4-dimension 1-10 absolute rubric — the two disagree.
- metric_name='win_rate' is a misnomer: returned value is a 1-5 Likert (see display_predictions.json win_rate:3,4,4...); no pairwise wins are computed.
- max_tokens=512 likely truncates long-form creative outputs (observed predictions appear cut off mid-sentence in display_predictions.json).
- Arena-Hard v2.0 paper/data from lmarena-ai is newer than arXiv 2406.11939 (v1); paper URL references v1, scenario loads v2 data — version mismatch in citation.

### `arena_hard_v01`
_Flagged borderline-creativity per NOTES_FOR_VIJETA.md section 1 (generic Chatbot Arena QA mixing coding/math/creative; v0.1 has no creative-writing subset — v2.0 does but is not publicly implemented here). Beyond the contamination concern, the implementation is not faithful to Arena-Hard-Auto: the paper's core methodology is pairwise win-rate vs a GPT-4 baseline, and this run_spec instead emits a 1-5 quality score with no baseline, so leaderboard numbers will not be comparable to any published Arena-Hard result._

**Gaps:**
- Paper uses PAIRWISE comparison vs GPT-4 baseline; implementation is POINTWISE 1-5 Likert (run_specs/arena_hard_v01_run_specs.py:15-24, _RUBRIC_WIN_RATE)
- No baseline answers loaded: scenario sets references=[] (scenarios/arena_hard_v01_scenario.py:86-87), so judge cannot compute a true win rate
- Metric is labeled 'win_rate' but stores 1-5 scores (display_predictions.json: annotations.generic_llm_judge_win_rate.win_rate in {1,2,3,5}) — mis-named, not a win rate
- Judge prompt in annotator_notes.md:16-35 describes pairwise 'Assistant A/B' template, but run_spec passes the pointwise rubric instead — notes and code disagree
- max_tokens=512 truncates long coding/math answers (run_specs:44); paper allows longer generations
- Only turns[0] used; multi-turn prompts in dataset are dropped (scenarios:80-82)

### `artinsight`
_Rubric mismatch, absent few-shot calibration, and empty base64_images combined with identical per-instance outputs indicate the scenario is not faithfully implemented and models are not actually seeing the artwork. Do not run full eval until multimodal delivery is verified and the rubric is corrected._

**Gaps:**
- run_specs/artinsight_run_specs.py:15-24 uses a generic 1-5 'art analysis' rubric; paper specifies a 0-16 five-criterion rubric (not-presumptive, not-reductive, sufficient detail, all elements captured, miscellaneous deductions) per metric_notes/artinsight_annotator_notes.md:18-27.
- run_specs/artinsight_run_specs.py:53-64 uses GenericLLMJudgeAnnotator without the 6 few-shot image+description+score exemplars the paper requires (annotator_notes.md:29-36); judge is not primed for calibrated scoring.
- metric_notes/artinsight_eval_metrics_notes.md is empty (1 line) — no eval-metric spec recorded.
- display_predictions.json show identical predicted_text across all 30 instances per model (e.g., gemini_pro rows 5-77, gemma_3_27b rows 5-77); images are not reaching the model or the prompt is ignoring them — outputs describe a photograph of women baking, not children's artwork. base64_images is [] for every row.
- scenarios/artinsight_scenario.py:95-104 prompt is a paraphrase reconstruction, not the verbatim optimal prompt (acknowledged at line 33-34).
- adapter_spec temperature=0.7 (run_specs:45) differs from paper's deterministic settings; no judge_temperature for scoring matches (0.0) but judge does not receive the image (GenericLLMJudgeAnnotator likely text-only — judge_config_matches fails vision requirement in annotator_notes.md:42-43).

### `balancecc_prompt_generation`
_Scenario is a reasonable text-only adaptation of CCEdit's GPT-4V dataset-creation prompt, but BalanceCC's original evaluation is multimodal user studies on edited videos; scoring LLM creative prompts against a single human reference via BLEU/ROUGE is a poor proxy. Borderline-creativity flag (per NOTES_FOR_VIJETA.md) is warranted: this is prompt-engineering for a multimodal video-design pipeline, further from core linguistic creativity than other benchmarks. Predictions look well-formed and on-task._

**Gaps:**
- scenario uses 'Compound Change' in prompt text (line 50) but dataset uses 'Multiple Change' (scenario.py:25) — label/prompt mismatch may confuse model
- run_specs only uses n-gram overlap metrics (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) against a single human target prompt (run_specs:35) — inappropriate for open-ended creative generation where many valid paraphrases exist; paper itself uses human user studies, not automatic reference matching
- no LLM-judge / creativity annotator configured (annotators=None, run_specs:44); metric_notes files are empty (0 lines each), so no documented metric rationale
- scenario adapts Appendix B.1 prompt for single-prompt generation rather than the paper's batch/multi-editing-type generation — noted in source comments but means prompt_matches only partially
- task is text-only reformulation of a multimodal video-editing pipeline; without video input or image-grounded judging, creativity signal is weak

### `banner_request_400`
_Scenario loads data and splits cleanly (TEST_SPLIT only, ~400 instances), and the shared-cache patch (fa01ec93/e62d9976) looks correct. But the run spec collapses the paper's 6-dimension LLM-jury into one generic rubric, and existing trial predictions show the model isn't producing banner blueprints at all — suggesting a broken prompt/image pipeline that would invalidate any full-scale run tomorrow. Do not launch until outputs are verified on-task._

**Gaps:**
- run_specs/banner_request_400_run_specs.py:15-24 uses a single generic `llm_judge_quality` rubric instead of the paper's 6 dimensions (TAA/LPS/AQS/CTAE/CPYQ/BIS) documented in metric_notes/banner_request_400_annotator_notes.md:9
- run_specs/banner_request_400_run_specs.py:59 sets judge_temperature=0.0, but annotator_notes.md:10 specifies 0.3 per paper eval.py
- metric_notes/banner_request_400_eval_metrics_notes.md is 0 bytes (empty) — eval metrics notes never written
- scenarios/banner_request_400_scenario.py:46-74 uses a home-grown TASK_PROMPT rather than the paper's prompts/foreground_designer_prompt.py verbatim
- display_predictions show generic/irrelevant outputs across multiple models (gemini_flash returns identical 'too vague' stub for every instance; gemini_pro returns a kitchen-photo description) — logo image and/or prompt text is not reaching the model correctly
- run_specs/banner_request_400_run_specs.py:44 max_tokens=512 is too small for a full JSON blueprint with elements array
- run_specs temperature=0.7 (line 45) for a structured JSON generation task is likely too high
- scenarios/banner_request_400_scenario.py:107 iterates every advertiser x 4 pairs with no subsampling — this yields ~400 instances and is fine for the 400 name, but no max_eval_instances guard

### `brainteaser`
_Scenario code is clean but the run spec and data split decisions cause material divergence from the paper. The biggest blocker is the prompt/metric mismatch: CoT-style generations cannot be scored by exact_match on a letter, so yesterday's runs likely underreport accuracy severely. SP-only coverage and train-split evaluation are secondary but real faithfulness issues._

**Gaps:**
- scenarios/brainteaser_scenario.py:35 loads only SP config; paper evaluates both SP (Sentence Puzzle) and WP (Word Puzzle) — WP is missing entirely
- scenarios/brainteaser_scenario.py:35 uses split='train'; paper reports results on held-out test set, risk of evaluating on training data
- run_specs/brainteaser_run_specs.py:30 temperature=0.7 with num_outputs=1; MC accuracy benchmarks conventionally use greedy (temp=0) for determinism
- run_specs/brainteaser_run_specs.py:31-32 max_tokens=512 + stop_sequences=['\n'] combined with ADAPT_MULTIPLE_CHOICE_JOINT is mismatched — models emit multi-paragraph CoT ending in '$\boxed{B}$', exact_match on a single letter will score ~0
- display_predictions.json shows free-form essays, not letter tokens; no answer-extraction/regex post-processing is configured
- metric_notes/brainteaser_annotator_notes.md and brainteaser_eval_metrics_notes.md are both empty (0-line files) — no documented rubric
- scenario drops 'choice_list' field and relies on choice_order+label; verify alignment once against raw dataset to avoid label-shift bug

### `calligrapher`
_Borderline-creativity as flagged in NOTES_FOR_VIJETA.md sec 1 — this is multimodal typography/design (image generation), not text-to-text creativity. Even setting the borderline fit aside, the implementation is fundamentally divergent: an image-generation benchmark wired to a text-chat LLM with an unrelated poetic-self-reference judge, producing vacuous identical outputs across the test set._

**Gaps:**
- Task-model mismatch: paper is a text-to-image diffusion benchmark requiring image generation (scenarios/calligrapher_scenario.py:1-189), but run_spec drives text-only LLM generation (run_specs/calligrapher_run_specs.py:35-47). Evaluated models (gemini/gemma text-chat) cannot produce styled text images.
- Metric substitution: paper's metrics are FID, CLIP style-sim on masked regions, DINO style-sim, and OCR accuracy (metric_notes/calligrapher_eval_metrics_notes.md:7-31). Run_spec replaces all with a GenericLLMJudgeMetric scoring 'poetic quality' / 'self-referential text' (run_specs/calligrapher_run_specs.py:15-24, 49-50) — unrelated to typography style transfer.
- Rubric nonsensical for task: rubric asks about 'self-referential text' and 'linguistic creativity' (run_specs/calligrapher_run_specs.py:15-24), but scenario prompt is literally just "The text is '{text}'." (scenarios/calligrapher_scenario.py:51). Predictions are meta chatbot replies ('Hello! It looks like your message came through blank...') all scoring 1 (benchmark_output/runs/gemini_3_pro/.../display_predictions.json, gemini_2_flash/...).
- calligrapher_annotator_notes.md is empty (1 line); no annotator design documented.
- Adapter is ADAPT_GENERATION with text-only I/O; images in MultimediaObject are not plumbed to text-only models, so instances degrade to empty prompts (explains identical 'blank message' fallback responses across all 100 self_* instances).

### `conceptual_design`
_Scenario loads the correct 12 problems and AMT reference CSVs and prompt text derives from the repo's zero_shot.py, but it departs from the paper in both generation regime (single vs. 100) and evaluation (lexical n-gram metrics plus a generic GPT-4 quality judge, instead of embedding novelty/diversity and expert multi-dimensional ratings). Output shape is sensible and judge scores are emitted, but the scoring is not faithful to the source._

**Gaps:**
- Prompt adaptation: paper requests batch of 100 solutions; scenario requests 1 (conceptual_design_scenario.py:16-17, 72-77). Acknowledged in docstring but diverges from source.
- Metrics mismatch: paper uses sentence-embedding novelty/diversity (cosine) vs AMT refs plus expert feasibility/usefulness/novelty ratings; run spec uses exact_match/quasi_exact_match/f1/ROUGE-L/BLEU-1/BLEU-4 (conceptual_design_run_specs.py:50), which are inappropriate for open-ended ideation with many valid refs.
- LLM judge rubric scores generic 'quality (creativity/feasibility/completeness/relevance)' on 1-5 (conceptual_design_run_specs.py:15-24); paper uses separate novelty, feasibility, usefulness dimensions.
- Judge model 'openai/gpt-4' (conceptual_design_run_specs.py:58) is not the paper's methodology — paper relies on human experts, not LLM-as-judge.
- metric_notes/conceptual_design_annotator_notes.md and eval_metrics_notes.md are empty (1 line each) — no documentation.
- Prompt variants 'base/novel/diverse/unique' exist in code but run spec hardcodes default (args={}) — only 'base' is exercised.

### `cpers`
_Scenario loading, Persian prompt template, and temperature=1 align with the paper. However the judge pipeline is materially different: a generic English 1-5 literary-quality rubric with GPT-4o replaces the paper's 12-item TTCT protocol plus rhetorical device detection validated against Claude 3.7 Sonnet. Predictions look sane (Persian text, 1-5 integer scores) but the scores are not the paper's metric._

**Gaps:**
- Judge model is openai/gpt-4o (cpers_run_specs.py:59), paper specifies Claude 3.7 Sonnet validated via ICC (cpers_annotator_notes.md:13-14).
- Rubric is a single generic 1-5 literary quality scale (cpers_run_specs.py:15-24); paper prescribes 4 TTCT dimensions x 3 questions (12 items) + rhetorical device detection (cpers_annotator_notes.md:22-97).
- No rhetorical device annotator (simile/metaphor/hyperbole/antithesis) despite being a core paper finding (cpers_annotator_notes.md:71-97).
- Full 4,371-item test split is used (cpers_scenario.py:95); paper's human-validated subset is 200 texts; unclear whether full set was scored by the judge in the paper.
- Extra BasicGenerationMetric (exact_match/bleu/rouge) is irrelevant for open-ended creative generation (cpers_run_specs.py:50).
- cpers_eval_metrics_notes.md is empty (1 line).

### `creai_cps`
_Prompt text matches baseline from Prompts.py verbatim. However, the benchmark dataset is severely truncated (10 items, 5 unique) vs. the 240 claimed in scenario docstring, and the original proprietary RoBERTa scorers are replaced by surface-overlap metrics plus a generic LLM judge. Split is TEST_SPLIT zero-shot which is appropriate, but truncated data invalidates meaningful scoring._

**Gaps:**
- data.json contains only 10 items with 5 unique scenarios (duplicated); scenario docstring claims 240 (creai_cps_scenario.py:26). display_predictions.json shows repeated predictions at id1/id3/id5 and id0/id4 confirming duplication.
- Proprietary RoBERTa originality/quality scorers from paper are unavailable; run spec substitutes BLEU/ROUGE/F1 (run_specs/creai_cps_run_specs.py:50) which metric_notes acknowledges 'measure similarity, not creativity'.
- LLM judge uses a generic single-axis 1-5 creativity rubric (run_specs/creai_cps_run_specs.py:15-24); paper rubric covers multiple facets (novelty, appropriateness, effectiveness) per annotator notes.
- creai_cps_annotator_notes.md is empty (0 lines).

### `creation_mmbench`
_Scenario loading and data split (765 test) look correct and the pattern-B disk-path patch is in place, but predictions indicate the model is effectively receiving empty input and the judge is a generic quality Likert rather than the paper's instance-specific dual-evaluation VFS/Reward. As configured, this run does not reproduce Creation-MMBench._

**Gaps:**
- display_predictions.json shows empty base64_images and canned 'no text provided' replies across all instances (benchmark_output/runs/trial_10inst/creation_mmbench_model=google_gemini-2.5-flash-lite/display_predictions.json); multimedia payload is not reaching the model at inference time despite pattern-B disk-path patch (scenarios/creation_mmbench_scenario.py:155-179)
- Judge rubric is a generic 1-5 creative-quality Likert (run_specs/creation_mmbench_run_specs.py:15-24) instead of the paper's instance-specific dual-evaluation with Visual Factuality Score (1-10) and Reward (-100..+100) using per-instance 'criteria' dict (metric_notes/creation_mmbench_annotator_notes.md:14-40)
- The instance 'criteria' (subjective requirement / groundtruth alignment) is attached as a Reference but never wired into the judge prompt; GenericLLMJudgeAnnotator ignores it (run_specs/creation_mmbench_run_specs.py:53-64)
- No dual-evaluation / position-swap pass implemented (required by paper Section 3.3)
- Judge is text-only GPT-4o with no image passed, so Visual Factuality cannot be assessed
- max_tokens=512 truncates long creative/professional writing responses (run_specs/creation_mmbench_run_specs.py:44)

### `creative_process`
_Prompts are verbatim from scripts_LLM/together_call.py and num_instances=5 matches the paper's 5 reps per model-temp combo, so input side is faithful. Evaluation side is entirely wrong: a generic GPT-4 creativity/originality judge replaces the paper's embedding + clustering + jump-profile pipeline, the empty annotator_notes file confirms this was never fleshed out, and trial outputs show the sampler returning identical sequences across reps (killing any diversity signal the metric would measure anyway). Only 1 of 3 paper tasks (vf) was scheduled. Abstract fetched; full PDF not parsed but metric_notes/creative_process_eval_metrics_notes.md already captures the canonical pipeline and the implementation does not match it._

**Gaps:**
- run_specs/creative_process_run_specs.py:60-86 scores responses with a generic 1-5 LLM judge for 'creativity' and 'originality' — the paper's entire contribution is process analysis (gte-large embeddings, hierarchical clustering at cutoff=1.14 for VF, semantic-similarity chains, jump_cat/jump_SS, jump profile slope) and NONE of that pipeline is implemented in HELM metrics or anywhere post-hoc in this repo
- metric_notes/creative_process_annotator_notes.md is empty (0 lines) — only the eval_metrics notes exist, and those describe the paper's real metrics, not the judge actually wired up
- run_specs/creative_process_run_specs.py:69 sets judge_model_name='openai/gpt-4' — not one of the 8 paper baselines and unrelated to the paper's evaluation (paper uses no LLM judge at all)
- display_predictions.json shows collapsed outputs: all 5 vf_* instances for gemini-2.5-pro emit the identical 30-animal list despite temperature=0.7; gemma-3-27b-it likewise emits identical lists across reps — repetition-based diversity analysis (the whole point of 5 reps) is defeated, suggesting the sampler is not actually sampling, deterministic caching, or that num_outputs=1 with seed reuse is nulling temperature
- scenario only instantiates task='vf' by default (run_specs args={}) — aut_brick and aut_paperclip tasks defined in the scenario are never exercised by the run_spec, so 2 of 3 paper tasks are missing from the trial run
- judge scores are uniformly 1/5 on every single instance across both models checked; judge is giving no signal and is not a faithful stand-in for the paper's quantitative jump-profile metrics

### `creatset`
_Data loading (hf_hub_download on CreataSet-test_with_labeling_400.jsonl, 50x8 domains), zero-shot adapter, and Chinese instruction-as-input are all faithful. The judge, however, is a generic English 1-5 pointwise creativity rubric with GPT-4o rather than the paper's verbatim Chinese pairwise prompt scored against baseline model responses. Predictions look sane (Chinese output, integer 1-5 annotations) but the reported score is not the paper's win-rate / Bradley-Terry metric and ignores the dataset's human-calibrated pairwise labels._

**Gaps:**
- Judge is pointwise 1-5 English rubric (creatset_run_specs.py:15-24, 51-63); paper/CrEval uses PAIRWISE Chinese prompt '更有创意的回复是：Response X' against the 3-criterion rubric (creatset_annotator_notes.md:18-47).
- Paper's primary metric is win-rate vs. gen_resp_2 (GPT4o-mini-c) or full 4-model Bradley-Terry pairwise ranking (creatset_annotator_notes.md:70-76); current pipeline generates one solo score per response with no baseline comparison.
- gen_resp_1..4 reference responses and avg_score/labeling human calibration data are read-and-discarded by the scenario (creatset_scenario.py:23-28); no calibration against human Bradley-Terry scores.
- Recommended evaluator CrEval-7b (Aman/CrEval-7b) not wired; only GPT-4o used (creatset_run_specs.py:58).
- BasicGenerationMetric exact_match/quasi_exact_match/f1/rouge_l/bleu_1/bleu_4 is meaningless for Chinese open-ended creative generation against a single curated reference (creatset_run_specs.py:50).
- creatset_eval_metrics_notes.md is empty (1 line).

### `crowdcounter`
### `cs4`
_Scenario data loading and prompt template ({instruction}\n\nConstraints:\n{list}\n\nWrite a story...) match the paper and GitHub source. Instance construction, constraint-level metadata, and TEST_SPLIT are correct. The divergence is entirely on the eval side: the paper's automatic constraint-satisfaction metric — the whole point of CS4 — is replaced by a single generic 1-5 creativity score from GPT-4o, with no per-constraint check, no constraint-level stratification, and a 512-token cap that truncates outputs. Predictions themselves look sane (coherent stories, judge scores are integers 1-5)._

**Gaps:**
- Metric replaced: paper's primary metric is constraint satisfaction ratio (per-constraint yes/no via GPT-4) (cs4_annotator_notes.md:11-17,41-54); implementation uses a single generic 1-5 creativity rubric (cs4_run_specs.py:15-24,49-64).
- Missing paper metrics: Coherence, N-gram Diversity, Perplexity, QUC, RCS (cs4_annotator_notes.md:18-38); none implemented.
- No stratification/reporting by constraint level (7/15/23/31/39) — the core experimental variable of the paper (cs4_annotator_notes.md:84-89,103-106). Scenario stores num_constraints in extra_data (cs4_scenario.py:143-149) but no metric slices on it.
- Judge prompt does not reference the per-constraint list nor require evidence citations as the paper prescribes (cs4_annotator_notes.md:41-54,56-59).
- max_tokens=512 (cs4_run_specs.py:44) truncates stories mid-sentence (visible in display_predictions.json for gemini-2.5-pro id92 etc.), biasing any creativity/constraint judgment downward.
- Default dataset_type='instruction' (cs4_scenario.py:61) uses only 250/500 instances — Story-based half is excluded unless explicitly overridden.
- cs4_eval_metrics_notes.md is empty (1 line).

### `deep_math`
_Scenario data loading and prompt construction are faithful (correct URLs, 179 items, bilingual, no extra framing). Evaluation layer is the failure point: empty references + lexical metrics + no judge means the current pipeline cannot measure mathematical correctness at all, and 512-token cap truncates nearly every response. Must not be run as-is for reproduction. Paper abstract accessed; full methodology details (judge model, exact rubric) not in abstract._

**Gaps:**
- Metric mismatch: paper uses lenient expert/LLM-judge scoring of mathematical correctness emphasizing core solution components (abstract explicit). run_specs/deep_math_run_specs.py:34-36 wires only BasicGenerationMetric with exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4 against EMPTY references (scenarios/deep_math_scenario.py:153,176). These lexical metrics on empty gold will be ~0 and carry no signal.
- No annotator/judge: annotators=None in run_specs/deep_math_run_specs.py:44 despite scenario docstring (lines 23-25) stating evaluation requires expert review or LLM-as-judge.
- Empty metric_notes: metric_notes/deep_math_annotator_notes.md and metric_notes/deep_math_eval_metrics_notes.md are both 1-line/empty files — no judge rubric, no scoring criteria defined.
- max_tokens=512 (run_specs/deep_math_run_specs.py:29) truncates proofs/constructions; observed display_predictions.json outputs for gemini-2.5-pro are visibly cut mid-proof (e.g. id7, id169, id107, id161 end mid-sentence). Paper's O3-Mini 70% number implies full-length constructive answers.
- temperature=0.7 (run_specs/deep_math_run_specs.py:30) is unusual for math proofs; paper does not specify but reproduction typically uses greedy/low-temp. Minor.
- task_type default 'all' yields 179 instances as intended (78 proof + 101 counterexample), matching paper counts. Mixed Chinese/English preserved. Prompts used verbatim (no extra framing) which matches scenario docstring.

### `discovery_bench`
### `esp_dataset`
_Reference tagging fix in e2eb48b7 is correct but insufficient: the prompt is so minimal that Gemini 2.5 Flash Lite returns a generic refusal for every one of the 10 sampled instances, so downstream BLEU/ROUGE/CLIP numbers will all measure refusal text, not stylistic generation. Blocking issue is the prompt (and likely image-delivery) plumbing, not the reference schema._

**Gaps:**
- Prompt format in scenarios/esp_dataset_scenario.py:147-157 is only '\n{style}:' (e.g. '\nnews:') alongside the image. This bare-label prompt gives no task instruction; the trial predictions in benchmark_output/runs/trial_10inst/esp_dataset_model=google_gemini-2.5-flash-lite/display_predictions.json show every one of 10 instances returning 'I'm sorry, but you haven't provided any text for me to work with...', confirming the model did not understand the task from the prompt alone.
- Adapter input_suffix='\n' and output_suffix='\n' (run_specs/esp_dataset_run_specs.py:24-26) further strip/obscure the multimedia content; combined with the no-instruction prompt this yields empty/refusal outputs across all sampled runs.
- Metric set in run_specs/esp_dataset_run_specs.py:34-37 adds exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4, and CLIPScore. Paper reports BLEU-4/METEOR/CIDEr (per scenario docstring line 26); METEOR and CIDEr are missing and CLIPScore (image-text, not caption-similarity) is not a standard ESP metric.
- No metric_notes/esp_dataset_annotator_notes.md or esp_dataset_eval_metrics_notes.md content — both files exist but are empty (0 lines read).
- Scenario docstring claims 996 images; no assertion on loaded instance count. Paper/GitHub do not disclose exact count to cross-check and WebFetch of both CVPR page and esper repo did not return prompt-format or metric details to verify the './\n{style}:' choice.
- Commit e2eb48b7 correctly adds CORRECT_TAG so references are scorable (line 161), but the underlying reference-metric pipeline is moot while predictions are refusals.

### `fann_or_flop`
_Scenario loads dataset and constructs a reasonable Arabic prompt, but the run spec is effectively a placeholder — no judge, no paper metrics, and output truncation at 512 tokens. metric_notes/fann_or_flop_eval_metrics_notes.md is empty. Not safe to run as-is for paper-comparable results._

**Gaps:**
- Run spec uses BasicGenerationMetric with exact_match/quasi_exact_match/f1/rouge_l/bleu_1/bleu_4 (run_specs/fann_or_flop_run_specs.py:35) — none of these are paper metrics. Paper uses LLM-as-judge (GPT-4o, faithfulness/fluency/overall 1-5), BERTScore via AraBERT, BLEU + chrF++, and mDeBERTa bidirectional NLI entailment.
- annotators=None (run_specs/fann_or_flop_run_specs.py:44) — no LLM judge is wired despite annotator_notes.md documenting the exact GPT-4o rubric. Faithfulness/fluency/overall scores will not be produced.
- Scenario prompt is a custom Arabic instruction (scenarios/fann_or_flop_scenario.py:63-81) acknowledged in the docstring as 'No explicit model prompt specified in the paper' — this is a reasonable stand-in but not paper-faithful.
- BERTScore is AraBERT-based in the paper; HELM BERTScore metric (not even wired here) defaults to English RoBERTa — would need explicit Arabic model config.
- chrF++ and mDeBERTa entailment are not in the metric list at all.
- max_tokens=512 (run_specs/fann_or_flop_run_specs.py:29) is likely too short for verse-by-verse Arabic explanations of full poems; sample predictions (benchmark_output/runs/gemini_pro/.../display_predictions.json) show outputs truncated mid-first-verse, meaning even the current metrics score incomplete generations.
- Temperature 0.7 (run_specs/fann_or_flop_run_specs.py:30) not specified in paper; paper judge is temp 0.0 but generation temp is undocumented.
- Full 6,984-instance train split used without subsampling (scenarios/fann_or_flop_scenario.py:106,138) — judge cost would be very large if judge were wired.

### `fscg8`
_Flagged borderline in NOTES_FOR_VIJETA.md section 1 (multimodal layout/design group) — FSCG-8 is fundamentally an image-generation benchmark and unlikely to belong in a text-to-text creativity suite. Independent of the borderline fit, the implementation is divergent from the paper in every axis that matters: adapter modality wrong, metrics absent, rubric copy-pasted from an unrelated story-completion task, reference images loaded but unused, predictions empty or off-topic factual replies scoring 1._

**Gaps:**
- Task-modality mismatch: FSCG-8 (ProCreate, ECCV 2024, arxiv 2408.02226) is a text-to-IMAGE generation benchmark across 8 categories (pokemon, one_piece, amedeo_modigliani, apple, frank_gehry, burberry, nouns, rococo), measuring sample diversity and fidelity of generated images (scenarios/fscg8_scenario.py:1-48). Run_spec drives text-only ADAPT_GENERATION with a text LLM (run_specs/fscg8_run_specs.py:35-47). Text LLMs cannot produce images.
- Metric substitution: paper's metrics are FID, KID, Precision, Recall, Vendi, MSS, CLIP prompt fidelity, SSCD (metric_notes/fscg8_eval_metrics_notes.md:22-41). Run_spec replaces them all with a single GenericLLMJudgeMetric scoring 'fine-grained story completion' quality (run_specs/fscg8_run_specs.py:49-51). None of the paper's diffusion-diversity metrics are implemented.
- Rubric nonsensical for task: rubric text evaluates 'fine-grained story completion... narrative coherence, consistency with the story context' (run_specs/fscg8_run_specs.py:15-24). FSCG-8 prompts are short object/style noun phrases like 'a Rococo style chandelier' or 'an Apple laptop charger' — no story to continue. Rubric appears copy-pasted from another scenario (fine-grained story continuation).
- Output shape reveals the mismatch concretely: predictions are either empty strings scoring 1 (many amedeo_modigliani/frank_gehry/one_piece instances on gemini-2.5-flash) or off-topic factual chatbot answers ('You're likely thinking of the Apple HomePod...' for instance apple_46 scoring 1; 'Apple silicon M-series chips...' for apple_5) — not story completions, not image generations, not evaluable under the declared rubric.
- fscg8_annotator_notes.md is empty (1 line); no annotator rationale documented.
- Reference images are saved to disk (scenarios/fscg8_scenario.py:140-153) but never compared to model outputs, because no image-based metric is wired in and text models return no images. base64_images is [] on every instance.

### `futuregen`
_Prompt text matches the RAG notebook verbatim and dataset loading from HF is correct, but the evaluation pipeline is substantially under-specified relative to the paper — missing BERTScore and the signature LLM-as-judge (novelty/hallucination/feasibility) framework. Output truncation at 512 tokens is visible in predictions. Blocking fixes are needed before results will be comparable to the paper's reported numbers._

**Gaps:**
- Paper evaluates with ROUGE, BLEU, BERTScore AND an LLM-as-judge framework assessing novelty, hallucination, and feasibility; run_spec only uses BasicGenerationMetric (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4).
- No BERTScore metric configured despite being a core paper metric.
- No LLM-as-judge annotator/metric wired up (annotators=None); judge model (Claude in paper) not configured.
- metric_notes/futuregen_annotator_notes.md and futuregen_eval_metrics_notes.md are empty stubs (1 line each) — evaluation design not documented.
- max_tokens=512 may truncate RAG prompt outputs (observed outputs cut mid-sentence in display_predictions.json); paper outputs are multi-bullet long-form.
- temperature=0.7 unjustified; paper likely uses deterministic/low-temp generation for judge reproducibility (not verified from abstract-only fetch).
- Scenario uses full NeurIPS subset (278 papers) as TEST_SPLIT with no held-out train/dev separation documented.

### `grapheval_ai_researcher`
_Prompt is a verbatim copy of Baselines/Prompt/basic_prompt.txt (cross-checked via raw.githubusercontent.com). Data path (test_set.jsonl from Data/AI_Researcher) and fields (title, abstract, decision) match upstream. Output shape across gemini-2.5-pro, gemini-2.5-flash, and gemma-3-27b is clean two-line `Overall Score (0-100)= NN\n<decision>` - models follow the format. The faithfulness failure is not at the prompt/data level but at the metric level: the scenario captures the correct label but scores a generic 1-5 quality rubric instead of decision accuracy, so reported numbers do not reflect the paper's task. Combined with the 9-instance test set and the triple-variant issue, this benchmark contributes noisy non-creativity signal to the suite in its current form._

**Gaps:**
- Metric mismatch (primary issue): the GraphEval task is a 4-way paper-decision classification evaluated with F1 / accuracy against the ground-truth `decision` field (Reject / Accept Poster / Oral / Spotlight). run_specs/grapheval_ai_researcher_run_specs.py:49-52 registers only a generic `llm_judge_quality` 1-5 rubric scoring `accuracy, depth of reasoning, relevance, and quality of the research evaluation`. The reference decision is attached to the instance (scenarios/grapheval_ai_researcher_scenario.py:111-112) but is never compared to the model's predicted decision. No classification_accuracy / F1 / exact-match metric is wired up.
- Judge rubric is generic, not task-specific: `_RUBRIC_LLM_JUDGE_QUALITY` scores `research evaluation quality` on a 1-5 scale without access to the ground-truth decision, so the judge cannot assess whether the predicted Accept/Reject label is correct. display_predictions.json confirms judge scores vary 1-4 independent of label correctness (e.g. `Accept (Poster)` gets scores 1, 2, and 4 across instances).
- Judge model drift: run_specs/grapheval_ai_researcher_run_specs.py:57 hardcodes openai/gpt-4 (not gpt-4o / o1); the original paper reports LLM-as-judge variants with different backbones. Minor but worth noting.
- Output format capture: scenario.py:16-18 says `We extract the decision from model output` but no extraction logic exists in get_instances; the full `predicted_text` (Score + decision) is passed to the judge. Harmless for current pipeline but does not match the stated design.
- Test split is tiny (9 instances; scenario.py:20) — single-run variance is huge, and combined with a 1-5 quality judge the metric carries almost no signal.
- TRIPLE-VARIANT OVER-REPRESENTATION: grapheval_ai_researcher, grapheval_iclr, and grapheval_review_advisor are three subsets of the same ulab-uiuc/GraphEval release (AI_Researcher, ICLR, Review_Advisor) with the same exact prompt and same decision-classification task. Counting all three as independent creativity benchmarks triple-weights one source (NOTES_FOR_VIJETA.md section 1).
- CREATIVITY-VALUE AMBIGUITY: the task is peer-review decision prediction (pick one of 4 labels given title+abstract). It is a classification/judgment task, not a generative creativity task. Keeping it in the creativity suite likely over-counts `idea evaluation` as `creativity` (NOTES_FOR_VIJETA.md section 1 flag).

### `grapheval_review_advisor`
_Same failure mode as the aligned-case triple-variant concern: prompt is faithfully lifted from the repo, but the scoring layer is a generic advisory LLM-judge that ignores the gold classification label the scenario already supplies. Predictions are clean 'Overall Score = X\n{Decision}' strings across models, so swapping in an ExactMatch/F1 classification metric is a small mechanical fix that would make the run actually reproduce GraphEval's Review_Advisor evaluation. As currently configured, the reported llm_judge_quality numbers do not measure the paper's task and should not be treated as a Review_Advisor reproduction. Paper PDF body was not fully extractable via WebFetch; cross-check relied on GitHub README (accuracy/precision/recall/F1) and the baseline prompt file already embedded in the scenario._

**Gaps:**
- Metric divergence (triple-variant): paper/repo evaluates Review_Advisor as 4-class decision classification with accuracy, precision, recall, F1 against ground-truth decision labels (Reject / Accept (Poster) / Accept (Oral) / Accept (Spotlight)). run_specs/grapheval_review_advisor_run_specs.py:49-51 discards the labeled reference entirely and instead scores outputs with a GenericLLMJudgeMetric on a generic 1-5 'llm_judge_quality' advisory rubric unrelated to the paper's task.
- Rubric semantically unrelated: _RUBRIC_LLM_JUDGE_QUALITY (run_specs/grapheval_review_advisor_run_specs.py:15-24) asks the judge to rate 'quality of the generated review advisory response' / 'accuracy of advice, relevance to the review context, and overall helpfulness'. The scenario prompt actually asks for a decision classification + 0-100 score, so the judge is evaluating something the model was never instructed to produce and ignoring whether the decision matches the ground truth in scenarios/grapheval_review_advisor_scenario.py:116.
- Reference label wasted: scenario stores the gold decision as a Reference with CORRECT_TAG (scenarios/grapheval_review_advisor_scenario.py:116) but no exact-match / classification metric (ExactMatch, F1, accuracy) is registered to consume it. display_predictions across runs (e.g., gemini_pro id1 -> 'Reject', id27 -> 'Accept (Oral)', id886 -> 'Reject'; gemma_3_27b id27 -> 'Accept (Poster)') are perfectly shaped for string-match classification scoring but are instead scored 1-5 by a generic judge, producing near-random quality numbers (e.g., id27 with identical 'Accept (Oral)'/'Accept (Poster)' outputs receives quality=1; id1 'Reject' receives quality=5).
- Judge config: openai/gpt-4 @ temp=0.0, max_new_tokens=512 is reasonable in isolation, but the paper never specifies an LLM-judge protocol for Review_Advisor (it uses classification metrics), so there is no paper judge config to match.
- Metric notes are empty: metric_notes/grapheval_review_advisor_annotator_notes.md and metric_notes/grapheval_review_advisor_eval_metrics_notes.md are both 1-line / empty files, so there is no documented justification for replacing the paper's classification metric with an LLM-judge advisory rubric.
- Prompt itself is faithful: SYSTEM_PROMPT in scenarios/grapheval_review_advisor_scenario.py:47-92 is copied from Baselines/Prompt/basic_prompt.txt with the 6 dimensions, 4 few-shot examples, decision distribution note, and output format; titles/abstracts are pulled from ReviewAdvisor_test_set.jsonl (test split, 1025 items). Output shape (Overall Score + decision) is respected by models.

### `graphragbench-wrongone`
_Dropped 2026-04-21 (see drops.json): duplicate of graphrag_bench pointing at the
wrong upstream paper. arXiv:2506.02404 is generic CS-textbook Q&A, not a creative
generation benchmark — tagged "creativity" in scenario only by copy-paste, with
no creative-generation signal in any of the 5 question types (FB/MC/MS/TF/OE all
have deterministic ground-truth answers). Naming linkage is also broken: scenario
file uses hyphen (graphragbench-wrongone_scenario.py) while run_spec import path
uses the same hyphenated module — Python cannot import a hyphenated module name,
so the ScenarioSpec class_name "scenarios.graphragbench-wrongone_scenario.GraphRAGBenchScenario"
would fail at load time in a clean environment. stats.json exists only because a
prior sweep cached outputs. Metrics are exact_match/F1/ROUGE/BLEU — wrong family
for creativity evaluation even if the data were appropriate. Redundant with the
correctly-scoped graphrag_bench entry._

### `humor_transfer`
_Scenario fetches test splits directly from HF/S3 per paper sources. Prompt structurally matches Alpaca-style Appendix C template. Paper PDF could not be fully extracted via WebFetch; Appendix C verbatim text not verified. Key break: reference label space ('Yes'/'No') does not match what the prompt elicits from models ('Funny'/'Not Funny'), making metrics unreliable. display_predictions.json confirms 'Funny'/'Not Funny' and rambling outputs from gemini-2.5-pro._

**Gaps:**
- Reference labels are 'Yes'/'No' (per paper Appendix C), but models output 'Funny'/'Not Funny' or verbose multi-paragraph reasoning; exact_match/quasi_exact_match will be ~0 regardless of correctness.
- Prompt asks to classify as 'funny or not funny' — instruction wording drives 'Funny'/'Not Funny' responses that do not align with the 'Yes'/'No' reference labels.
- temperature=0.7 and max_tokens=512 for a binary classification task encourages rambling; paper uses deterministic fine-tuned inference.
- No annotator/judge used — relies purely on surface-form metrics (exact_match, f1, rouge_l, bleu) which are ill-suited to a binary label that may be embedded in reasoning.
- run_spec uses args={} so only sarcasm_headlines subset runs; amazon_questions subset is never instantiated.
- metric_notes files (annotator_notes.md, eval_metrics_notes.md) are empty.
- Full test splits are very large (26,709 + 19,142) with no sampling cap — expensive and unaligned with paper's evaluation protocol.

### `hypobench`
_HypoBench as implemented is a single-instance, single-task, scientific-novelty-rubric eval — which is neither the paper's Accuracy pipeline nor the HDR metric, and only covers 1 of 7 real-world tasks. The annotator_notes file is empty (1 line), suggesting onboarding was incomplete. Duplication with hypogen flagged in NOTES_FOR_VIJETA.md is NOT a benchmark-level duplicate (distinct papers/datasets) but IS a code-level duplicate — hypogen's scientific-creativity judge was reused verbatim for hypobench's social/behavioral tasks, which is the root cause of the metric mismatch. Arxiv abstract and HypoBench website fetches confirmed 12 tasks (7 real + 5 synthetic) and 'hypothesis discovery rate' as a paper-level metric but did not yield full HDR prompt details; cross-check relied on local metric_notes/hypobench_eval_metrics_notes.md and the ChicagoHAI/HypoBench-datasets repo structure referenced in the scenario. Not a faithful reproduction; treat current scores as uninterpretable relative to paper._

**Gaps:**
- Duplication concern vs hypogen: NOTES_FOR_VIJETA.md:43 flags hypogen/hypobench possible overlap. Verified DISTINCT upstream papers/datasets: hypobench = ChicagoHAI HypoBench (arXiv:2504.11524, 12 tasks observation->hypothesis from labeled examples, datasets repo ChicagoHAI/HypoBench-datasets), hypogen = UniverseTBD/hypogen-dr1 bit-flip from paper abstracts (arXiv:2409.04109). Task formats are different (labeled observation induction vs. abstract bit-flip). NOT a duplicate benchmark. However, the run_specs ARE near-duplicates of each other (same GenericLLMJudgeAnnotator wiring, overlapping novelty rubric text, same generation adapter) — the code-level duplication is real even if the dataset distinction is legitimate.
- Task coverage gap: scenarios/hypobench_scenario.py:64-72 exposes 7 real-world tasks via VALID_TASKS but run_specs/hypobench_run_specs.py:52-55 calls HypoBenchScenario() with args={} — defaults to task='deceptive_reviews' only. Only 1 of 12 paper tasks is actually run. Paper claims 7 real + 5 synthetic (194 datasets); scenario additionally excludes journal_cross/journal_same (scenario:41) and synthetic tasks (scenario:39-40). Single-task run without task parameterization severely undercovers the benchmark.
- Single instance per task: scenario.get_instances (scenarios:106-153) returns exactly ONE Instance (one sampled prompt with num_observations=10 labeled examples). display_predictions.json confirms n=1 (only id0 present across gemini_pro and gemini_flash runs). The paper evaluates across many train/val/test partitions; a single-instance eval produces essentially a single score per model per task and collapses all statistical signal.
- Metric does not match paper: metric_notes/hypobench_eval_metrics_notes.md:16-26 documents paper uses a TWO-STEP pipeline (generate hypothesis -> apply as classifier -> measure accuracy on held-out test) for real tasks, and HDR (LLM-judge semantic match vs. ground_truth_hypotheses) for synthetic tasks. run_specs/hypobench_run_specs.py:72-75 instead registers LLM-judge NOVELTY/SIGNIFICANCE/VERIFIABILITY rubrics on a 1-5 scale. This is a completely different construct than the paper's Accuracy or HDR, and is not a 'proxy' of either. The metric_notes file explicitly says the current implementation is a proxy using ROUGE-L/BERTScore (open_ended) — but the run_specs do not even register open_ended metrics, so the documented proxy isn't running either.
- Judge rubrics mismatched to task: NOVELTY/SIGNIFICANCE/VERIFIABILITY rubrics (run_specs:15-46) describe evaluation of SCIENTIFIC hypotheses; HypoBench real tasks are behavioral/social (deceptive_reviews, stress detection, retweet prediction, AI-content detection). Scoring a 'hotel-review-authenticity decision rule' on 'major scientific breakthroughs' (rubric:34) is a category error. The novelty rubric text is essentially copy-pasted from hypogen_run_specs.py with minor edits, reinforcing the code-duplication flag above.
- Judge config: openai/gpt-4 @ temperature=0.0, max_new_tokens=256 — paper uses GPT-4 judges, so model/temp OK in isolation. Paper's HDR judge prompt is a binary match decision between H_gen and H_gt (metric_notes:37-39); run_specs uses a 1-5 Likert. judge_max_new_tokens=256 adequate for a single integer.
- Reference construction: scenario passes metadata.known_hypotheses as Reference list (scenarios:141-145). These are consumed by an open_ended metric in the paper's proxy design, but no open_ended MetricSpec is registered, so references are ignored by the active GenericLLMJudge metric (which scores the generation alone, not against references).
- Prompt format: scenario:132-138 assembles system+user via {observations} + 'Generate 5 hypotheses' — matches config.yaml batched_generation templates pulled live from the ChicagoHAI repo (scenarios:62, 109). Prompt fidelity to the paper's Section 3 batched-generation template is reasonable. num_hypotheses hardcoded to 5 (scenario:133) matches paper default.
- Adapter: max_tokens=512 truncates mid-hypothesis in observed runs — display_predictions.json shows predicted_text ending mid-sentence at 'Reviews that describe specific, often minor, functional failures or maintenance issues within the room or' (gemini_flash id0) and 'Truthful reviews' (gemini_pro id0, 5 hypotheses requested but cut off). 512 tokens is insufficient for 5 full hypotheses with rationales; paper's batched generation uses larger budgets.
- Sampling seed: seed=42, num_observations=10 hardcoded (scenarios:80). Paper sweeps K observations and splits; fixed 10 is one point on that curve.

### `hypogen`
_Scenario loads UniverseTBD/hypogen-dr1 test split (50 examples) correctly and writes a well-formed bit-flip prompt, and display_predictions across all 14 model runs show the judge firing with reasonable integer scores 1–5. But the faithfulness posture has three real issues (a) the cited paper (2409.04109) is not the HypoGen paper, (b) the bit/flip role definitions may be inverted relative to the HF dataset card, and (c) the docstring advertises BLEU/ROUGE that the run_spec does not actually compute — only an LLM-judge novelty score. Separately, hypogen and hypobench are confirmed NOT duplicate datasets (different source papers, different task formulations), so the NOTES_FOR_VIJETA flag resolves to "distinct, keep both", but the llm_judge_novelty metric name collides across them and should be namespaced. All three blocking fixes are cheap; the HF-vs-scenario bit/flip semantics check is the most important one to settle before quoting any HypoGen numbers._

**Gaps:**
- Paper citation in scenario docstring is wrong. Scenario cites arXiv:2409.04109 ("Can LLMs Generate Novel Research Ideas?" by Si/Yang/Hashimoto) as the HypoGen source, but that is a different paper about human-vs-LLM idea blind review. The actual HypoGen / bit-flip dataset is from UniverseTBD (Astro/AstroLLaMA line), not from 2409.04109. Provenance/citation must be corrected before results are defensible.
- Bit/flip role definitions may be inverted. Scenario docstring and prompt assert bit = "conventional wisdom / limitation" and flip = "novel insight". The HuggingFace dataset card (UniverseTBD/hypogen-dr1) describes BIT as "core insight/method description" and FLIP as "problem statement or reframing" — inconsistent with the scenario's prompt framing. If the HF card is authoritative, every prompt is asking the model to generate the wrong field and the reference is mislabeled. Needs verification against the original HypoGen paper PDF (not resolvable from abstract fetch).
- {'No reference string-overlap metric configured. Scenario docstring advertises "Evaluation': 'open_ended (BLEU, ROUGE against author-written flip)", but run_spec wires only a single LLM-judge novelty metric (llm_judge_novelty) and no BasicGenerationMetric — so BLEU/ROUGE promised in the scenario are not actually computed.'}
- Prompt is a "standard instruction" authored by the scenario (scenario comment admits "paper does not publish exact prompts"). Not verified against paper.
- Judge rubric hardcodes judge_model_name="openai/gpt-4" (legacy GPT-4) with no paper justification; paper's evaluation design not documented (no benchmark-specific judge choice or rubric provenance captured).
- metric_notes/hypogen_annotator_notes.md and hypogen_eval_metrics_notes.md are empty stub files (1 line each). Evaluation design fully undocumented.
- Scenario drops every field except abstract+bit+flip; spark and chain_of_reasoning (both human-labeled narrative fields in the dataset) are discarded without justification. Could plausibly serve as multi-reference ground truth.
- max_tokens=512 sometimes truncates mid-sentence (visible in gemma_3_4b display_predictions.json — outputs cut at "Crucially, the weights will be adjusted..."). Paper's gold flips are 195–609 chars, but models emit far longer rationales.
- temperature=0.7 unjustified for a single-generation novelty-judge setup; no paper basis.
- {'Potential benchmark redundancy with hypobench. NOTES_FOR_VIJETA.md Section 1 flags "hypogen / hypobench — both hypothesis generation; confirm they\'re distinct". Verdict from this audit after reading both scenarios': 'they ARE distinct tasks (hypogen = retrieve/reconstruct a paper\'s specific flip given its abstract+bit; hypobench = generate N hypotheses that explain labeled observations for 7 real-world tasks, sourced from ChicagoHAI/HypoBench-datasets, arXiv:2504.11524) — different inputs, different references, different paper. So not a duplication in the "drop one" sense. BUT both use identically-named llm_judge_novelty rubrics from different run_specs with slightly different wording, which will be confusing when metrics are aggregated by metric_name across the suite — worth namespacing (e.g., hypogen_novelty vs hypobench_novelty) or at least documenting.'}

### `ii_bench`
_Scenario logic and metric (exact_match over A-F) are correct in principle, but the observed outputs demonstrate that neither the image nor the question reaches the model — every instance returns the same 'no question provided' refusal with empty base64_images. Do not run full eval until multimodal + prompt delivery is fixed and verified on a handful of instances._

**Gaps:**
- benchmark_output/runs/trial_10inst/ii_bench_model=google_gemini-2.5-flash-lite/display_predictions.json rows 1-71 show identical predicted_text across all 10 instances ('This is a bit of a trick question!...'); base64_images is [] for every row — the image is not reaching the model and the question text is being stripped/replaced before inference.
- Model replies 'there's no question or context' on every row, confirming both the image AND the question body built in scenarios/ii_bench_scenario.py:108-116 are absent at inference time; only the A-F options and 'Answer:' appear to have been delivered. Multimodal delivery and prompt assembly through MultimediaObject (lines 119-128) are broken.
- scenarios/ii_bench_scenario.py:75 hard-codes split='dev' (35 examples) because test labels are hidden; this is a reasonable fallback (acknowledged at lines 39-41) but diverges from the paper's 1,399-item test evaluation and shrinks the domain-balance design (6 domains, see lines 14-15).
- run_specs/ii_bench_run_specs.py:30 sets temperature=0.7; II-Bench paper evaluations use deterministic decoding (greedy / temperature=0) for MC accuracy, so scores will have extra variance.
- run_specs/ii_bench_run_specs.py:20-32 uses ADAPT_MULTIPLE_CHOICE_JOINT with output_prefix='Answer: ', while scenarios/ii_bench_scenario.py:116 already appends '\nAnswer:' to the prompt — the adapter likely double-emits the 'Answer:' cue and/or overrides the scenario prompt, which is consistent with the observed empty-question outputs.
- metric_notes/ii_bench_annotator_notes.md and metric_notes/ii_bench_eval_metrics_notes.md are both empty (1 line) — no recorded metric spec; paper metric is plain MC accuracy, which matches exact_match in principle but is undocumented here.
- CoT flag (scenarios/ii_bench_scenario.py:59-66, 113-114) is wired but run_spec does not pass use_cot; paper's headline comparisons include CoT results, so only the zero-shot condition is covered.

### `infochartqa`
_DROP FROM LAUNCH. Scenario faithfully reproduces the paper's chart-QA task (prompt format, exact_match metric, splits all match the HuggingFace dataset card and paper). The unfaithfulness is not to the source paper but to the creativity-benchmark framing: this is a factual visual-QA task mis-tagged as creativity. Predictions across gemini_pro and gemma_3_27b confirm short factual answers (numbers, categorical labels, yes/no). No creative generation, no judge, no annotator - nothing measures creativity. Recommend exclusion. If the team wants visual-reasoning discriminant-validity evidence, a ~500-item subsample would suffice at <1% of current cost._

**Gaps:**
- Scope mismatch with creativity benchmark: NOTES_FOR_VIJETA.md flags infochartqa as NOT creativity - it is a pure multimodal chart QA task with 58,857 exact-match questions. The scenario docstring itself states 'Task: Multimodal question answering on infographic charts' (scenarios/infochartqa_scenario.py:8-9).
- Task is visual recognition / numeric value extraction on charts, not open-ended creative generation. Display predictions confirm this: responses are terse factual answers (e.g. 'Nevada', '310', '19.1', 'Yes/No', 'C') - see benchmark_output/runs/gemini_pro/... and gemma_3_27b/... display_predictions.json.
- Metric is exact_match / quasi_exact_match / f1 / rouge_l / bleu (run_specs/infochartqa_run_specs.py:35) - entirely correctness-based, with no creativity judge or creative-output annotator (annotators=None, line 44). This faithfully reproduces the paper's evaluation but has no bearing on creativity.
- metric_notes/infochartqa_annotator_notes.md and metric_notes/infochartqa_eval_metrics_notes.md are both empty (single-line files), indicating no creativity-oriented evaluation was ever designed for this scenario.
- Scale incompatibility: 58,857 test questions would dominate the suite at substantial API cost while contributing zero creativity signal. Already lists 14 completed runs in benchmark_output/runs/*/infochartqa_model=* - budget already being spent on an out-of-scope task.
- Groups tag includes 'creativity' (run_specs/infochartqa_run_specs.py:43) and scenario tags include 'creativity' (scenarios/infochartqa_scenario.py:64) despite the task being factual chart QA. Tag is misleading - likely auto-added by scenario scaffolding, not reflective of task content.
- Paper (arxiv 2505.19028) confirms the benchmark tests 'visual recognition and reasoning' over infographic charts with pictorial/metaphor elements. Even the 'visual_metaphor' subset (462 Qs) is metaphor *comprehension*, not creative production.

### `irfl`
_Three compounding issues make the current irfl runs uninterpretable: (1) every correct answer is hardcoded to A, so any position-biased model scores 100% and any other scores 0% independent of figurative understanding; (2) images are not actually being delivered to the VLM (empty base64_images, refusal-style text outputs); (3) only 1 of 4 configs runs. Even if multimedia delivery is fixed, the no-shuffle bug invalidates the metric. Data split choice (test) is correct per the HF dataset card._

**Gaps:**
- scenarios/irfl_scenario.py:178 hardcodes correct answer to position A (correct_index=0) — no shuffling; comment says 'shuffle to randomize position' but code never shuffles. Massive position bias; chance-accuracy ceiling of 100% if model learns to always output A
- scenarios/irfl_scenario.py:78 only loads idiom-detection-task by default; run_specs/irfl_run_specs.py:18 passes args={} so metaphor/simile/open-simile configs are never evaluated — paper evaluates all three figurative types
- run_specs/irfl_run_specs.py:30 temperature=0.7 with num_outputs=1 for a deterministic 4-way MC task; paper-style accuracy eval should be greedy (temp=0)
- run_specs/irfl_run_specs.py:29,31 max_tokens=512 + stop_sequences=['\n'] with ADAPT_MULTIPLE_CHOICE_JOINT but no answer-extractor; exact_match on letter will fail whenever model emits any prose
- display_predictions.json (trial and trial_10inst) shows model returned 'Please provide the question or context for the options A, B, C, and D' for every instance — images are not reaching the model (base64_images: []). Multimedia pipeline is broken end-to-end
- run_specs/irfl_run_specs.py:27 max_train_instances=0 zero-shot is fine, but instructions='' relies entirely on scenario-internal prompt which lacks explicit 'respond with a single letter A/B/C/D' directive
- metric_notes/irfl_annotator_notes.md and irfl_eval_metrics_notes.md are both empty (0-line files) — no documented rubric or human-baseline reference
- scenarios/irfl_scenario.py:104 image-existence check uses >10000 files threshold; zip contains 10062 — brittle but likely fine

### `layoutsam_eval`
_Borderline per NOTES_FOR_VIJETA.md sec 1 — flagged under 'Multimodal layout/design — unclear if they count as text-to-text creativity'. Upstream is explicitly a layout-to-image generation benchmark; HELM integration strips the image-generation half and invents a text-only captioning reframing with no paper provenance, then scores it with a generic LLM rubric that ignores the human-annotated global_caption reference. Flag as borderline-creativity AND divergent-from-paper; recommend drop or full redesign._

**Gaps:**
- Task reframing away from the paper: LayoutSAM-Eval is a Layout-to-Image (L2I) generation benchmark (arXiv:2412.03859). The paper's eval uses VLM-based Yes/No spatial + attribute QA (MiniCPM-V-2.6) plus FID, CLIP, PickScore, IS on generated images (scenarios/layoutsam_eval_scenario.py:17-19 acknowledges this). Scenario reframes it as text-only 'layout -> one-paragraph scene description' (scenarios/layoutsam_eval_scenario.py:21-42, 67-76) — the paper has no such text-only prompt (scenario admits 'No text-only prompt specified in the paper').
- Metric substitution: paper metrics are VLM QA / FID / CLIP / PickScore / IS on images; run_spec uses a single GenericLLMJudgeMetric 'llm_judge_quality' with a generic 1-5 rubric on spatial accuracy/completeness/clarity (run_specs/layoutsam_eval_run_specs.py:15-24, 49-50). Scenario docstring line 54 also advertises ROUGE-L/BLEU against global_caption but neither is actually registered.
- Reference mismatch: scenario exposes global_caption as a CORRECT_TAG reference (scenarios/layoutsam_eval_scenario.py:115) suggesting reference-based metrics were intended, yet run_spec only wires an LLM judge and the judge rubric never sees the reference — judge scores descriptions in a vacuum without the gold caption.
- Judge config: judge_model_name='openai/gpt-4o', temperature=0.0, max_new_tokens=256 (run_specs/layoutsam_eval_run_specs.py:57-60). Paper does not prescribe an LLM judge at all, so nothing to match against — this is a fully invented eval.
- metric_notes/layoutsam_eval_annotator_notes.md and metric_notes/layoutsam_eval_eval_metrics_notes.md are both empty (1 line each) — no annotator/metric design documented.
- Outputs are shape-reasonable (single-paragraph scene descriptions across gemini/gemma models; scores 1-5 span the rubric, e.g. gemini-2.5-flash id398=4, id3833=5; gemma-3-1b id398=1), but because the task itself is off-paper, high scores do not reflect the paper's L2I quality — they reflect fluent paragraph writing about an abstracted layout.

### `litbench`
_As wired today, the litbench run does not measure LitBench. It measures GPT-4's 1-5 literary-quality rating of Gemini's free-text commentary on pairs of stories, while ignoring the dataset's gold preference labels entirely. Scenario code correctly constructs the pairwise task with position randomization and CORRECT_TAG references, but no metric consumes them. Must not ship without a pairwise-accuracy metric and adapter fixes._

**Gaps:**
- Primary metric drift: LitBench paper's headline metric is pairwise preference accuracy vs. human labels (Claude-3.7-Sonnet ~73%, trained RMs ~78%). run_specs/litbench_run_specs.py:49-52 registers only a generic 5-point llm_judge_quality rubric (GenericLLMJudgeMetric), which does not compare the prediction against the gold chosen/rejected label at all. No accuracy metric is computed, despite scenarios/litbench_scenario.py:238-247 correctly wiring CORRECT_TAG references to A/B.
- Judge misuse: annotator is openai/gpt-4 with a generic literary-quality rubric (run_specs/litbench_run_specs.py:15-24, 53-64). The paper evaluates the model-under-test AS the judge predicting chosen-vs-rejected; LitBench does not require an external LLM judge over the model's free-text output. The current pipeline ignores the scenario's correct_letter references and instead asks GPT-4 to rate the model's prose quality 1-5.
- Output shape broken: display_predictions.json shows every prediction is 500+ word literary commentary ending mid-sentence (truncated at max_tokens=512) rather than the required 'Preferred: [A or B]'. Adapter uses ADAPT_MULTIPLE_CHOICE_JOINT with temperature=0.7 and stop_sequences=['\n'], but output_prefix='Answer: ' plus a prompt ending in 'Preferred: [A or B]' creates conflicting instructions; model ignores the format and rambles. Zero predictions in the sampled file contain 'Preferred:' or a clean A/B choice.
- Adapter/temperature mismatch: ADAPT_MULTIPLE_CHOICE_JOINT with max_tokens=512 and temperature=0.7 is inappropriate for a forced-choice A/B task; paper/GitHub dataloader expects a single-token deterministic choice. Temperature should be 0.0 and max_tokens small (e.g. 8).
- Prompt template adapted: scenarios/litbench_scenario.py:128-141 prepends the writing prompt and a 5-aspect rubric, acknowledged as a deviation from the LLAMA_PROMPT Direct variant in SFTDataLoaderDirect (scenario docstring lines 18-22, 46-48). Minor faithfulness risk but documented.
- Data rehydration dependency: scenario depends on api.pullpush.io (scenarios/litbench_scenario.py:70-73) to rehydrate 2,381 Reddit comments; silent skip on fetch failure (lines 183-184) can shrink N without warning. Paper reports 2,480 test pairs; scenario docstring says 2,381. Discrepancy not explained.
- Stop sequence bug: stop_sequences=['\n'] combined with a prompt whose final newline precedes 'Preferred:' may cause the model to terminate before producing its answer under some adapters.
- Empty metric notes: metric_notes/litbench_{annotator,eval_metrics}_notes.md are both 0 bytes, so there is no recorded spec for the intended metric; cross-check relied solely on scenario/run_spec code and the arXiv abstract.

### `llm4biohypogen`
_Scenario prompt exactly reproduces Table 11 5-shot template and pulls official test splits from the TsinghuaC3I/LLM4BioHypoGen repo, which is the strongest faithful element. The evaluation harness, however, is gutted: the paper's headline evaluation is the 4-dimension GPT-4 judge, and no annotator is registered. Running as-is produces only BLEU-4/ROUGE-L on one of four splits - not a faithful reproduction. Paper abstract access via arXiv succeeded but full-text details (exact judge prompt, temperature) were not retrievable via WebFetch; cross-check relied on annotator_notes.md which cites Section 4.2 of the paper._

**Gaps:**
- Judge/annotator NOT wired: run_specs/llm4biohypogen_run_specs.py:46 sets annotators=None. The paper's core evaluation is a GPT-4 judge on 4 dimensions (Novelty, Relevance, Significance, Verifiability, 0-3 scale) per metric_notes/llm4biohypogen_annotator_notes.md, but no LLMAsJuryAnnotator or annotator spec is registered. Only BLEU-4 and ROUGE-L are computed (run_specs/llm4biohypogen_run_specs.py:35), which are secondary metrics in the paper.
- Split coverage: scenario defaults to model_version='gpt-3.5', test_type='seen' (scenarios/llm4biohypogen_scenario.py:71) and run_spec passes args={} (run_specs/llm4biohypogen_run_specs.py:17), so only 1 of 4 splits (gpt-3.5 seen / gpt-3.5 unseen / gpt-4 seen / gpt-4 unseen) is exercised. Paper emphasizes seen-vs-unseen contrast to quantify contamination; this is lost.
- Prompt is hard-coded 5-shot (Table 11) but AdapterSpec has max_train_instances=0 and is labeled zero-shot (run_specs/llm4biohypogen_run_specs.py:27). The 5 few-shot exemplars are baked into the scenario prompt string, which matches paper Table 11, but the run_spec comment is misleading. Temperature=0.7 is a guess; paper does not specify generation temperature for this task.
- bert_score / SelfBLEU missing: annotator_notes lists SelfBLEU as a diversity metric and BERTScore is referenced in code TODO (run_specs/llm4biohypogen_run_specs.py:37-38). Neither is registered. No multi-sample generation (num_outputs=1) so SelfBLEU cannot be computed even if wired.
- Output shape is sane: display_predictions.json (benchmark_output/runs/trial/llm4biohypogen_model=google_gemini-2.5-flash-lite and trial_10inst) show numbered (1)(2)(3) hypothesis lists matching the target format (3 numbered statements per prediction). Data download from TsinghuaC3I/LLM4BioHypoGen GitHub raw URLs appears to work.

### `macgyver`
_Scenario implementation is faithful to the paper's prompt texts and correctly ingests the xlsx dataset with proper subset filters, but the run spec collapses the paper's 3×4 design matrix (strategies × subsets) into a single vanilla/all configuration and applies exact-match metrics that are definitionally wrong for open-ended creative generation. Cross-model prediction audit shows unstable LLM-judge scoring on unsolvable items, consistent with the rubric gap flagged above. Fix the metric set and parameterize the run spec before trusting leaderboard numbers._

**Gaps:**
- run_specs/macgyver_run_specs.py:33 instantiates MacgyverScenario with args={} — hard-codes subset='all' and prompt_strategy='vanilla'; the three prompt strategies (vanilla, divergent_convergent, reflection) and four subsets (all, solvable, unsolvable, unconventional) declared in scenarios/macgyver_scenario.py:54-55 are never exercised, so the paper's central comparison across prompting techniques (Appendix D.3, Figures 15-16) is not reproduced
- scenarios/macgyver_scenario.py:122-131 collapses the paper's two-turn reflection protocol into a single prompt by appending Round-2 instructions after Round-1; paper specifies multi-round conversation where Round 2 sees Round 1's generated solution before verifying (Figure 16). Current single-shot framing cannot replicate iterative reflection even if the strategy were selected
- run_specs/macgyver_run_specs.py:45 temperature=0.7 with num_outputs=1; paper's GPT-4 baseline collection (code/collect_solutions/collect_GPT4_solutions.py) uses lower/default temperatures for deterministic single-solution generation — non-zero temperature adds noise to an already-small eval
- run_specs/macgyver_run_specs.py:44 max_tokens=512 is adequate for vanilla but insufficient headroom for divergent_convergent (affordance listing + summary + solution) and reflection (full re-derivation); truncation risk if those strategies are later enabled
- run_specs/macgyver_run_specs.py:50 BasicGenerationMetric applies exact_match/quasi_exact_match/f1_score to open-ended creative solutions — paper explicitly states (Section 4.2) that automatic n-gram overlap is a weak proxy and relies on human annotation; EM/QEM against a single gold solution will floor near 0 and is not meaningful
- run_specs/macgyver_run_specs.py:58 judge_model_name='openai/gpt-4' uses legacy GPT-4 rather than gpt-4-turbo/gpt-4o; paper uses GPT-4 as strongest model under test, not as judge, so there is no exact judge-config to match, but rubric granularity diverges
- run_specs/macgyver_run_specs.py:15-24 rubric is a 1-5 Likert on correctness/feasibility only; metric_notes/macgyver_annotator_notes.md:18-28 lists 8 categorical labels from the paper (efficient, inefficient, infeasible, correct_right_reason, correct_wrong_reason, wrong_partial_correct, wrong_solution, correct_unsolvable). Collapsing to a 5-point scalar loses the unsolvable-detection signal that is the whole point of the 377-problem unsolvable subset
- metric_notes/macgyver_eval_metrics_notes.md is empty (1 line); no documented rubric or metric rationale committed
- scenarios/macgyver_scenario.py:154 sets a single gold reference per instance; for unsolvable items the gold 'solution' field describes why it is infeasible — BLEU/ROUGE against such prose is semantically incoherent as a correctness signal
- display_predictions across gemini_2.5_flash and gemma_3_27b runs show divergent judgments on identical items (e.g. id34 scored 1 vs 5, id268 scored 2 vs 3, id1209 scored 5 vs 1) suggesting the judge is not stable at judge_temperature=0 or the rubric is under-specified for unsolvable-detection cases

### `matdesign`
_MATDESIGN as currently wired is not a faithful implementation of the AccelMat paper. The scenario ships a 1-example hardcoded fallback (because its data source is a /tmp/ path that never exists), and every matdesign_model=* run in benchmark_output/runs contains only instance_id "id0" — so the advertised "50 examples" is effectively 1. On top of that, max_tokens=512 truncates the 20-suggestion JSON mid-output for every model (confirmed in gemini-2.5-pro and gemini-2.5-flash display_predictions, outputs end with no closing brace mid-Suggestion_2), so even that one example produces invalid JSON. The evaluation is a single GenericLLMJudgeMetric with a generic 1–5 "design quality" rubric and gpt-4 judge — bearing no resemblance to the paper's 3-critic consensus + iterative refinement + o1-preview closeness/quality rubrics documented in the (well-populated) annotator_notes.md. The eval_metrics_notes.md is an empty stub. Scenario docstring, paper URL, and reference handling are correct; the dataset plumbing, generation budget, and evaluation design are all blocking issues that must be fixed before any matdesign number can be reported. Annotator notes already document the correct AccelMat design, so the gap is implementation, not design knowledge._

**Gaps:**
- Dataset loading is broken/fake. Scenario checks a hardcoded Linux path (/tmp/matdesign_data.json) that never exists on this Windows host, so every run falls back to a SINGLE hardcoded example (the self-healing hydrogel) instead of the paper's 50-example 2024 materials-science dataset. Confirmed in display_predictions.json for every matdesign_model=* run — only instance_id "id0" is present across all 15 model outputs. The scenario's own docstring advertises "50 examples" while the code delivers 1.
- Dataset is Excel-format in the upstream repo (Materials Discovery & Design Dataset.xlsx) and the scenario comment acknowledges "The JSON file should be hosted on GitHub... In the actual implementation, convert the Excel file to JSON and host it." No such conversion or hosting was done; BASE_URL is set but never fetched. This is a TODO stub shipped as production code.
- {'Evaluation framework is completely wrong. Paper uses the AccelMat multi-agent system': '3 critic agents (GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Flash @ temp 0.7) that per-suggestion label "Meets_the_goal_statement_and_satisfies_all_constraints_strictly" YES/NO, a GPT-4o summarizer, up to 5 iterative refinement rounds, then an o1-preview evaluator scoring Concept/Property/Keyword overlap (closeness) plus 6 quality axes (Alignment, Scientific Plausibility, Innovation, Testability, Feasibility, Impact) on 5-point rubrics. Run_spec instead wires a single GenericLLMJudgeMetric with judge_model_name="openai/gpt-4" (legacy GPT-4, not GPT-4o) at temperature 0.0 and a generic 1–5 "quality of materials design" rubric. None of the 3-critic consensus, iterative refinement, concept/property/keyword overlap, or 6-axis quality breakdown are implemented.'}
- Rubric text is generic and paper-unrelated. _RUBRIC_LLM_JUDGE_QUALITY evaluates "scientific accuracy, novelty, completeness, and practical applicability" on a generic 1–5 scale — not tied to the paper's constraint-satisfaction YES/NO criterion or to the 6 AccelMat quality dimensions.
- metric_notes/matdesign_eval_metrics_notes.md is an empty stub (1 line). Only the annotator notes file is populated (and it correctly documents the paper's real evaluation — confirming the run_spec diverges from what the maintainer already knew was required).
- Output shape unvalidated. Task requires 20 JSON-formatted suggestions per instance; max_tokens=512 is far too short for 20 materials+methods+reasoning blocks. Observed display_predictions truncate mid-second-suggestion in gemini-2.5-flash output (predicted_text cuts at "...the embedded ionic liquid ensures high ionic conductivity and low hysteresis due" with no closing brace). Gemini-2.5-pro output also truncates mid-Suggestion_2. Every model's generation is structurally broken.
- judge_max_new_tokens=512 likely too short to fairly evaluate a (truncated) 20-suggestion response.
- Paper citation in scenario docstring (arXiv:2501.13299) is correct; repo URL is correct. Provenance itself is fine — the implementation is the problem.
- Reference field wraps ground-truth materials+methods into a single Reference with tag "reference_only", which is appropriate (paper explicitly says models should generate NOVEL suggestions, not reproduce), but the reference is never consumed by any metric since only an LLM judge runs.

### `meta4xnli`
_Scenario code faithfully reproduces the Table 29 NLI prompt strings (zero-shot and CoT verbatim including idiosyncratic trailing periods), and the data source (HiTZ/meta4xnli int_eval xnli_test_met) + exact_match metric are appropriate for the interpretation task. However, three compounding issues prevent the current runs from being interpretable as a Meta4XNLI replication: (1) only the default English zero-shot subset actually runs — the four other declared subsets are silently dropped because run_specs passes args={}; (2) the adapter is ADAPT_MULTIPLE_CHOICE_JOINT which will overlay an A/B/C letter block on top of the scenario's free-text prompt, so the prompt seen by the model is NOT the Table 29 prompt; (3) the detection_en subset uses a self-admitted custom generative prompt while the paper uses fine-tuned encoders/decoders with BIO/constrained decoding, so any detection F1 here is an off-paper probe, not a paper replication. Temperature 0.7 on a 3-way label task is also wrong. Interpretation data split is correct._

**Gaps:**
- run_specs/meta4xnli_run_specs.py:18 passes args={} so only the default subset 'interpretation_en' is instantiated; the other four declared subsets (interpretation_es, interpretation_en_cot, interpretation_es_cot, detection_en) are never run — paper evaluates English+Spanish and both zero-shot+CoT
- scenarios/meta4xnli_scenario.py:50-53 admits the detection_en prompt is custom-written and NOT from the paper; paper uses fine-tuned encoders with BIO tagging or fine-tuned decoders with constrained decoding, not zero-shot generative listing of metaphor tokens — results will not be comparable to paper numbers
- run_specs/meta4xnli_run_specs.py:21 uses ADAPT_MULTIPLE_CHOICE_JOINT but scenario builds its own free-text prompt and packs all three labels into References; joint adapter will append a letter-choice block ('A. entailment\nB. neutral\nC. contradiction\nAnswer: ') on top of the scenario's '... -> ...:' prompt — prompt is NOT verbatim Table 29 at inference time
- run_specs/meta4xnli_run_specs.py:30 temperature=0.7 with num_outputs=1 for a closed-label classification task; paper protocol is deterministic decoding for NLI accuracy
- run_specs/meta4xnli_run_specs.py:24 input_suffix='\n' + output_prefix='Answer: ' conflict with the scenario's own '... -> ...:' terminator and the CoT template's 'Answer:' terminator — double 'Answer:' tokens appear in the final prompt under CoT
- scenarios/meta4xnli_scenario.py:155 references are label strings but the adapter is MC_JOINT; exact_match compares generation to the reference string 'entailment'/'neutral'/'contradiction', so any letter-only output ('A'/'B'/'C') scores 0 — adapter/metric mismatch
- scenarios/meta4xnli_scenario.py:89-94 CoT examples include trailing periods on 'contradiction.' and 'neutral.' but NOT on 'entailment' — copied verbatim from Table 29 per comment, but this inconsistency may leak format cues
- metric_notes/meta4xnli_annotator_notes.md and meta4xnli_eval_metrics_notes.md are both empty (1-line/empty files) — no documented rubric, no paper-reported baselines (encoder F1, decoder accuracy) for calibration
- display_predictions.json shows only 10 instances for trial_10inst (expected) but filenames indicate only interpretation_en ran; no Spanish or CoT or detection outputs exist — consistent with single-subset gap above
- scenarios/meta4xnli_scenario.py:123 loads split='xnli_test_met' for interpretation which is correct (metaphor-containing test subset, ~580/lang); detection loads 'det_en_finetune' split='test' which is the fine-tuning split, not an independent eval split — acceptable but worth noting

### `metaphor_generation`
_Scenario data loading is correct (156 test pairs from MetaphorGenNAACL2021 human1test.txt/human2test.txt, <V> tags stripped) and the test-split assignment matches the paper. The faithfulness failure is in adaptation: with an empty instruction, instruction-tuned chat models (all Gemini/Gemma variants inspected) produce long analytical essays rather than metaphorical rewrites, which makes the configured BLEU/ROUGE/F1 metrics uninformative. This is a blocking prompt-format issue, not a data issue._

**Gaps:**
- Prompt is a bare literal sentence with no instruction (run_specs/metaphor_generation_run_specs.py:22 instructions=''; scenarios/metaphor_generation_scenario.py:90 prompt=literal_sentence) — instruction-tuned models interpret the input as a discussion prompt rather than a metaphor-rewrite task; observed outputs (benchmark_output/runs/gemini_flash/.../display_predictions.json; benchmark_output/runs/gemma_3_27b/.../display_predictions.json) are multi-paragraph essays analyzing the sentence rather than one-line metaphorical rewrites
- max_tokens=512 and temperature=0.7 (run_specs/metaphor_generation_run_specs.py:29-30) allow and encourage long essay responses; paper targets a single sentence output — no stop_sequences or output-length control
- Metrics are surface-overlap only (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4 — run_specs:35); MERMAID paper relies on BLEU plus human preference (66% over baselines), and these overlap metrics are near-meaningless when predictions are 500-token essays vs 1-sentence references
- metric_notes/metaphor_generation_annotator_notes.md and metric_notes/metaphor_generation_eval_metrics_notes.md are both empty (0/1 lines) — no documentation of metric choice or annotator protocol
- No LLM judge / semantic-similarity / metaphoricity classifier is configured (annotators=None, run_specs:44); paper's central evaluation (human judgment of metaphoricity) has no analogue here
- max_train_instances=0 with empty instructions means the model receives zero task signal — paper fine-tunes a seq2seq model on parallel data; zero-shot with no instruction is not a comparable setup and should minimally include an instruction like 'Rewrite the following literal sentence as a metaphorical sentence'

### `mops`
_Scenario correctly loads ManTle/mops curated split (100 rows, 14 themes) and wires the six component fields into a coherent synthesis prompt. Sampled predictions from gemini_pro and gemma_3_27b are coherent 1-3 sentence premises integrating all six modules, so the input side is essentially faithful. The metric side is not: scale (1-5 vs 0-100), dimensions (2 of 3), rubric framing (movie vs story), and inclusion of reference-overlap metrics all diverge from the paper. As wired today the run measures a GPT-4-turbo 1-5 rating of movie-ish premise quality plus BLEU/ROUGE noise, not MoPS's fascination/completeness/originality 0-100 protocol or its breadth/density diversity claim. Paper PDF text extraction failed (binary); cross-check used HF dataset page, HF paper abstract, and GAIR-NLP/MoPS repo README._

**Gaps:**
- Scoring-scale mismatch: paper and metric_notes/mops_annotator_notes.md:8-10, 19-41 specify a 0-100 integer scale per dimension (fascination, completeness, originality) with GPT-4-turbo. run_specs/mops_run_specs.py:15-35 instead encodes 1-5 Likert rubrics (_RUBRIC_LLM_JUDGE_FASCINATION, _RUBRIC_LLM_JUDGE_ORIGINALITY). Sampled display_predictions.json confirms scores bounded at 3-5, confirming the 1-5 rubric is in force.
- Missing third dimension: paper evaluates three dimensions (fascination, completeness, originality). run_specs/mops_run_specs.py:60-87 only registers fascination and originality annotators/metrics; completeness is dropped despite being documented in metric_notes/mops_annotator_notes.md:26-32. This is the dimension most specific to MoPS's modular-components claim.
- Rubric wording drift: rubrics in run_specs/mops_run_specs.py:16, 27 repeatedly say 'movie premise or story concept' and reference 'existing films', but the task is story-premise synthesis from ManTle/mops (theme/background/persona/event/ending/twist). The film framing is not in the paper and may bias the judge toward cinematic clichés.
- Reference-based metrics inappropriate: run_specs/mops_run_specs.py:61 registers exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4 against the dataset's single reference premise. The task is open-ended synthesis with many valid surface realizations; the paper does not use n-gram overlap against the reference premise, and near-zero overlap scores on sampled outputs (which are coherent and faithful to components) will be uninformative at best and misleading at worst.
- Diversity metrics omitted: paper's headline claim of superiority is on set-level diversity via 2D semantic breadth/density (ManTle/mops repo README confirms 'Breadth and Density metrics proposed in the paper'). Not implemented; acknowledged in metric_notes/mops_eval_metrics_notes.md/annotator_notes.md:47-49. Per-instance judge scores alone cannot reproduce the paper's primary contribution evaluation.
- Prompt template synthesized: scenarios/mops_scenario.py:56-66 constructs an instruction from scratch ('Given the following story components, write a concise story premise (1-3 sentences)...'). Scenario docstring (lines 10-13) acknowledges the paper does not publish a prompt; this is a reasonable reconstruction but should be documented as a deviation. The _strip_prefix helper (lines 44-49) also silently mutates field content when a module label is embedded in the value.
- Judge prompt templates in metric_notes/mops_annotator_notes.md:19-41 are not actually used; run_specs rubrics are entirely different text with different scale. Annotator notes are orphaned documentation.
- Adapter: temperature=0.7, max_tokens=512, num_outputs=1 (run_specs/mops_run_specs.py:46-58). Paper does not specify generation hyperparams for the model-under-test; sampled predictions look coherent and 1-3 sentences, so output shape is sane. metric_notes/mops_eval_metrics_notes.md is empty (0 bytes).

### `newyorker_humor`
_Matching prompt format (A-E choices with cartoon description) matches paper's text-only 'from description' setup. Scenario implementation covers all three tasks but run_spec only triggers default matching. display_predictions.json shows clean single-letter outputs (A-E), confirming MC adapter works. Paper PDF not extractable via WebFetch; arXiv abstract confirms three tasks and accuracy-based evaluation for matching, head-to-head preference for explanations. The biggest faithfulness issue is scope collapse: 1 of 3 tasks, 1 of 5 folds, plus a text-only variant presented without the multimodal context from the paper._

**Gaps:**
- Run spec uses args={} so only the 'matching' task runs with the default (non-fold) split; the 'ranking' and 'explanation' tasks from the paper are never instantiated despite being implemented in the scenario.
- Paper supports 5-fold cross-validation (matching_1..matching_4 etc.); run_specs does not parameterize cross_val_fold, so only a single split is evaluated.
- Metric spec includes rouge_l, which is meaningless for the matching task (single-letter A-E outputs) and only relevant for the explanation task that is never run.
- Scenario emits TRAIN, VALID, and TEST splits without any max_eval_instances cap; full val=531 and test=528 for matching run unthrottled — paper's evaluation focuses on test split.
- temperature=0.7 and max_tokens=512 are inappropriate for a deterministic multiple-choice task; paper uses argmax/deterministic decoding for MC accuracy.
- Prompt uses textual descriptions (image_description + image_uncanny_description + entities + questions) — faithful to paper's 'from description' text-only variant, but the paper's headline human/CLIP numbers are the 'from pixels' multimodal variant; this should be flagged as the text-only condition.
- No explanation-task annotator/judge configured; paper evaluates explanations via human head-to-head preference (GPT-4 vs human). Even if explanation task were instantiated, exact_match/rouge_l would not reflect the paper's evaluation protocol.
- exact_match likely uses raw reference text (letters A-E) — OK for matching/ranking, but label field for explanation task is a long gold explanation, making exact_match ~0.

### `noveltybench`
_Current implementation is a data-loading shell only: prompts load correctly but neither the diversity classifier nor the quality reward model exists, and the run captured a single generation per prompt rather than the N=10 sample needed for pairwise diversity. Verdict DIVERGENT because core evaluation metrics are entirely absent and multi-generation contract is unmet._

**Gaps:**
- No diversity classifier metric (core paper contribution missing)
- No quality reward model metric
- Only single generation per instance in current run output (n=10 not materialized)
- max_tokens=512 truncates long-form WildChat completions
- annotator_notes.md is empty (1 line)
- Cannot reach full paper text to confirm exact N, temperature, max_tokens

### `ocw`
_Prompt, dataset, and scenario construction faithfully follow the paper. The blocker is the metric: a global-Jaccard-over-tokens implementation does not measure grouping accuracy at all — a model that prints the 16 clues in any random order scores the same as a perfect solution. The documented metric (per-group set match + wall-solved) must be implemented before any OCW numbers are reportable. Secondary faithfulness issues (temperature, few-shot) are fixable but not fatal._

**Gaps:**
- metrics/group_match_score_metric.py:28-38 computes a single global Jaccard over the union of all 16 tokens in prediction vs reference — this is NOT the paper's metric. Because both pred_set and ref_set always contain the same 16 clues, Jaccard is ~1.0 regardless of whether groups are correct; the metric is trivially near-perfect and cannot distinguish solved from unsolved walls
- metrics/group_match_score_metric.py ignores newline-separated group structure entirely; it flattens everything into one set, discarding the 4x4 partition that defines the task
- Paper's primary metrics (per-group exact-set match 0-4, wall-solved binary) documented in metric_notes/ocw_eval_metrics_notes.md are NOT implemented — documented rubric and code disagree
- metric_notes/ocw_annotator_notes.md is empty (0 lines) — no annotator rubric recorded (acceptable since task is programmatic, but file should either be populated or removed)
- run_specs/ocw_run_specs.py:30 temperature=0.7; OCW paper's run_openai.ipynb baseline uses deterministic generation (temperature=0) for reproducibility on a structured puzzle task
- run_specs/ocw_run_specs.py:27 max_train_instances=5 enables few-shot from TRAIN_SPLIT, but paper reports zero-shot baseline; few-shot demos may leak solution patterns and change the task
- scenarios/ocw_scenario.py:83 loads train+validation+test but adapter may sample eval instances from any split depending on HELM config — verify test-only evaluation
- No parsing/post-processing for model output variants noted in metric_notes (Group X: prefixes, connection annotations, hallucinated words, wrong word counts); display_predictions.json shows id99 emits 'n/a\n\nHere is your Connecting Wall solution:\n\n...' which would break any group-aware parser

### `permpst`
_Input/prompt side is faithful: scenario correctly downloads Facebook Research PerMPST tar.gz, loads review.valid.c{k}.jsonl, passes the dataset's pre-formatted reviewer-history + new-plot prompt verbatim, and stores per-instance ground-truth score as a numeric reference. Sampled predictions from gemini_pro and gemma_3_27b show clean ```json {Review, Score} ``` outputs with scores spanning 2-9 in reasonable agreement with plot tone, so the model-under-test is clearly producing the right artifact. The evaluation side, however, does not compute the paper's metrics (Pearson/Spearman/Kendall over predicted-vs-ground-truth scores). Instead it runs n-gram overlap of the full JSON string against a single numeric reference token (guaranteed near-zero) and a copy-pasted 1-5 'permuted sentence translation' quality judge that is from a different task entirely. As wired, the permpst run cannot reproduce the PerSE/PerMPST paper's claims or rank models on personalization ability. Paper abstract confirmed via WebFetch (full PDF not extracted); metric spec cross-checked against metric_notes/permpst_eval_metrics_notes.md (annotator notes file is empty)._

**Gaps:**
- Correlation metrics missing: paper (arxiv 2310.03304 abstract; metric_notes/permpst_eval_metrics_notes.md:30-77) evaluates PerMPST via Pearson r, Spearman rho, and Kendall-Tau between predicted and ground-truth reviewer scores (1-10 ordinal). run_specs/permpst_run_specs.py:50 registers only BasicGenerationMetric with {exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4}. None of these compute a correlation over the set of predicted vs reference scores, so the paper's headline metric is not produced.
- Score never extracted for scoring: scenarios/permpst_scenario.py:144-173 defines _extract_score_from_completion and stores ground-truth score as the Reference text (e.g. '8.0'), but the BasicGenerationMetric compares the full raw predicted JSON string (e.g. '```json{"Review": "...", "Score": 8}```') against the single-token '8.0' reference. exact_match and quasi_exact_match will be ~0 by construction; f1/rouge/bleu are computed over unrelated tokens. The predicted Score is never parsed out on the prediction side, so no score-level comparison occurs.
- LLM judge rubric is wrong task: run_specs/permpst_run_specs.py:15-24 _RUBRIC_LLM_JUDGE_QUALITY says 'Evaluate the quality of the generated permuted sentence translation or story... accuracy of reconstruction' on a 1-5 scale. This rubric is copied from a permuted-sentence-reconstruction task and has nothing to do with PerMPST (personalized reviewer-score prediction). metric_notes/permpst_annotator_notes.md is empty (1 line / 0 bytes), so there is no canonical rubric backing this.
- Judge cannot measure personalization: the paper's contribution is personalized alignment (predicted score matches THIS reviewer's score for THIS plot given their history). A single-example 1-5 GPT-4 quality judge on the generated review text cannot assess reviewer-specific score agreement; it measures generic text quality of the review blurb. Sampled display_predictions.json (gemini_pro, gemma_3_27b) show judge scores clustered at 4-5 regardless of whether predicted Score is 2, 3, 7, 8, or 9, confirming the judge is blind to the score-prediction objective.
- Judge model weak: AnnotatorSpec uses 'openai/gpt-4' (run_specs/permpst_run_specs.py:58). Paper and annotator notes do not call for an LLM judge at all for PerMPST; the evaluation is deterministic correlation on extracted scores. Even if a judge were warranted, the scale (1-5) does not match the task's 1-10 reviewer scale.
- Per-reviewer / stratified-by-k aggregation missing: metric_notes/permpst_eval_metrics_notes.md:110-113, 159-172 specifies overall + per-reviewer + stratified-by-k correlations (92 reviewers, k in 0-5). Scenario defaults to k=1 and does not expose aggregation by reviewer; run_specs does not sweep k. Only a single k=1 overall pass is wired.
- Annotator notes orphaned/empty: metric_notes/permpst_annotator_notes.md is empty. metric_notes/permpst_eval_metrics_notes.md documents a correct scipy-based pearson/spearman/kendall pipeline (lines 98-108) that is not implemented anywhere in run_specs or a custom metric class.
- Adapter hyperparams: ADAPT_GENERATION, max_train_instances=0, temperature=0.7, max_tokens=512, num_outputs=1 (run_specs/permpst_run_specs.py:35-47). Prompt is constructed in-dataset (scenarios/permpst_scenario.py:196-203 passes the pre-formatted 'prompt' field verbatim), matching the paper's in-context personal profile format. Sampled outputs from gemini_pro and gemma_3_27b are well-formed ```json {Review, Score} ``` blocks with scores in 2-9, so output shape is sane and score extraction would succeed if wired on the prediction side.

### `puntuguese`
_Scenario correctly loads Superar/Puntuguese test split and binarizes label 1/0 to Yes/No references. display_predictions.json across gemini-2.5-pro and gemma-3-27b show that the multiple-choice-joint adapter is not reliably constraining outputs; many traces emit full explanations in English despite stop_sequences=['\n']. The metric_notes files are empty, so there is no documented expected metric value or annotator config. Paper baseline is F1=68.9% from fine-tuned Portuguese encoders, not LLM zero-shot — faithfulness to the dataset is high, but faithfulness to the paper's evaluation protocol is low. Anthology page for paper returned only abstract-level details via WebFetch._

**Gaps:**
- Scenario builds a custom Portuguese-English-mixed prompt ('Text: {text}\n\nIs this text humorous?') in English, while the paper's humor recognition experiments are run on Portuguese-native text with fine-tuned encoder classifiers (BERTimbau etc.), not zero-shot English-instructed LLM prompting — prompt faithfulness to paper is only nominal.
- metric_notes/puntuguese_{annotator,eval_metrics}_notes.md are EMPTY (0 lines) — no documented evaluation protocol or annotator config.
- Adapter uses temperature=0.7 and max_tokens=512 for a deterministic binary classification task; should be temperature=0 with tiny max_tokens to force a clean Yes/No.
- display_predictions.json shows severe output-shape problems: many predictions contain long multi-paragraph English explanations after the letter (e.g., 'A. Yes\n\n**Explanation:**...'), and at least one double-answer ('A. Yes\nAnswer: A'). MultipleChoiceClassificationMetric with ADAPT_MULTIPLE_CHOICE_JOINT typically parses leading letter, but stop_sequences=['\n'] is evidently not stopping generation in these traces — indicates the stop token is being ignored or the model ignored the Answer: prefix convention.
- Paper reports F1=68.9% as baseline from fine-tuned models; run_spec uses MultipleChoiceClassificationMetric which reports accuracy/F1 over MC choices — numbers are not directly comparable to paper's fine-tuned baseline.
- No max_eval_instances cap; entire test split (~980 items given 4,903 total with standard 80/10/10) is evaluated, which is fine for faithfulness but costly.
- Prompt does not include a system/instruction preamble in Portuguese; paper corpus is Portuguese — asking an English question ('Is this text humorous?') about Portuguese text introduces a language-mismatch bias the paper never tested.

### `puzzleworld`
_Scenario prompt construction faithfully reproduces PUZZLE_SYSTEM_PROMPT + PUZZLE_USER_PROMPT from the paper's repo, and uses the correct HF dataset and single train split. But the just-passed max=10 run is a false positive: every prediction is a boilerplate 'no content provided' apology, meaning the model never saw the puzzle (text or images) — the run completed without errors but produced zero signal. Combined with missing answer extraction and temp=0.7, current config cannot measure model ability on this benchmark._

**Gaps:**
- benchmark_output/runs/trial_10inst/puzzleworld_model=google_gemini-2.5-flash-lite/display_predictions.json: all 10 predictions are identical apology strings ('I'm sorry, but you haven't provided any text...') with base64_images=[] — model received no content; multimedia pipeline is not delivering images (or prompt text) to the model despite scenario attaching MediaObjects
- scenarios/puzzleworld_scenario.py:159 silently swallows hf_hub_download failures via bare 'except Exception: continue' — if images fail to download, instance proceeds with only the text MediaObject and no indication; masks the exact failure seen in predictions
- run_specs/puzzleworld_run_specs.py:30 temperature=0.7 with num_outputs=1; paper-style final-answer accuracy benchmarks normally use greedy (temp=0) for deterministic scoring against a single canonical solution
- run_specs/puzzleworld_run_specs.py:29 max_tokens=512 is likely insufficient for multi-step puzzle CoT the SYSTEM_PROMPT explicitly requests ('write out your steps as you go'); paper's reasoner allows long generations
- run_specs/puzzleworld_run_specs.py:35 scores with exact_match/quasi_exact_match/f1/rouge/bleu directly on raw generation; paper evaluates 'Answer: <answer>' canonical string — no regex extractor is configured, so exact_match will match the full CoT against a 1-word solution and score ~0 even when the model is correct
- metric_notes/puzzleworld_annotator_notes.md and puzzleworld_eval_metrics_notes.md are both empty (1 line) — no documented rubric, human baseline, or stepwise-accuracy plan
- scenario uses single 'train' split for all 667 puzzles (correct per HF repo) but does not implement the paper's stepwise accuracy metric; only final-answer exact_match is attempted
- difficulty='all' default is fine, but with max=10 the sample is not stratified across easy/medium/hard (paper reports 140/355/172) — trial subset hits only whatever ordering HF returns

### `rebus_puzzle`
_Paper is "Puzzled by Puzzles: When Vision-Language Models Can't Take a Hint" (arXiv:2505.23759), Kyunnilee et al. The scenario's prompt, dataset, and skill taxonomy faithfully mirror the released code, but predictions across three inspected model runs are identical degenerate strings, indicating a systemic image-delivery failure at inference; results from any completed run are not interpretable until that is fixed._

**Gaps:**
- Images not reaching models at inference; predictions degenerate and identical across instances for Gemini family runs.
- base64_images is empty in display_predictions.json even though scenario constructs MediaObject with image/png location.
- LLM-judge rubric omits the ground-truth answer reference, so judge cannot reliably score correctness.
- Generation temperature 0.7 diverges from paper's greedy/deterministic decoding for accuracy evaluation.
- Metric suite (EM + QEM + F1 + ROUGE + BLEU + 1–5 judge) is broader than the paper's accuracy metric; headline metric unclear.

### `recombination_extraction`
_Scenario prompt construction is faithful to PROMPT_E2E from CHIMERA src/util.py, and the train/test split mapping is correct. However the evaluation pipeline is fundamentally misaligned with the paper: BasicGenerationMetric on raw JSON strings cannot capture classification/entity/relation F1, the analogy vs inspiration key mismatch guarantees near-zero f1_score on inspiration cases, max_tokens truncates answers, and few-shot adaptation corrupts the zero-shot prompt. No LLM judge is wired. Needs a custom metric + annotator before results are meaningful._

**Gaps:**
- run_specs/recombination_extraction_run_specs.py:35 uses only BasicGenerationMetric (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4); paper evaluates with JSON-structured soft-matching F1 at 3 levels (classification, entity, relation) using GPT-4o-mini as judge — none of these metrics are faithful
- scenarios/recombination_extraction_scenario.py:139 gold reference is raw json.dumps(readable_relations) using dataset's 'analogy'/'analogy-src'/'analogy-target' keys, but prompt instructs model to emit 'inspiration'/'inspiration-src'/'inspiration-target'; token-level f1_score/exact_match will be guaranteed-wrong on every inspiration example due to key mismatch
- no custom metric parses <answer> tags or JSON; display_predictions show models emit '<scratchpad>…</scratchpad><answer>{…}</answer>' (often truncated mid-answer at max_tokens=512) — BasicGenerationMetric scores the entire verbose string against a compact JSON reference
- run_specs max_tokens=512 is too small: predictions in benchmark_output/runs/gemini_pro/.../display_predictions.json are frequently truncated mid-<answer> block (e.g., id399, id531, id514, id437), invalidating JSON parsing downstream
- run_specs temperature=0.7; CHIMERA paper evaluates deterministic extraction (temp=0 conventional for structured IE tasks)
- adapter_spec has max_train_instances=5 (few-shot) but scenario embeds the full PROMPT_E2E verbatim per instance with no demonstration slot; HELM will prepend 5 train examples as raw prompt/reference pairs that do not match the prompt's expected schema, polluting context
- metric_notes/recombination_extraction_annotator_notes.md is empty (1 line); eval_metrics_notes.md explicitly flags that a custom JSON-parsing metric + GPT-4o-mini soft matcher is required but none is implemented
- annotators=None; paper's evaluation is LLM-as-judge (GPT-4o-mini) per Appendix — a HELM annotator is required to reproduce HDR/soft-F1

### `scimon`
_Scenario correctly loads 194 human-verified instances from the official gold_subset.zip and routes rel_sent as the CORRECT_TAG reference; display_predictions.json for gemini-2.5-flash-lite on the 10-instance trial shows well-formed single-sentence outputs, so the data pipeline and adapter are functioning. The faithfulness problem is at the evaluation layer: the SciMON paper exists to measure novelty (via human judgment of novelty/relevance/technical-depth), and this run_spec scores only lexical+semantic similarity against a single sentence from the same paper — which cannot separate 'restated the background' from 'proposed a novel idea'. Both metric_notes files are empty stubs, the zero-shot prompt is benchmark-authored without paper justification, and exact_match/quasi_exact_match are inert on this task. Blocking fix is to populate the notes and add at least one novelty-sensitive metric (LLM judge or context-divergence proxy) before quoting SciMON numbers._

**Gaps:**
- Prompt template is authored by the scenario, not taken from the paper. SciMON (Wang et al., ACL 2024) primarily fine-tunes seq2seq models and does not publish a canonical zero-shot template; the scenario acknowledges this but still produces a single custom instruction ("Based on the above context, write one sentence...") without documenting the choice or comparing to the paper's GPT-3.5/4 zero-shot wording (if any).
- rel_sent is used as the sole CORRECT_TAG reference. Scenario docstring itself flags that automatic metrics on rel_sent measure semantic similarity, not novelty, and that the paper's primary evaluation is human judgment of novelty/relevance/technical depth — none of which is implemented here.
- Metric set is generic lexical-overlap plus BERTScore (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4, BERTScore). No novelty metric, no retrieval/citation-grounding metric, no LLM-as-judge, no human-eval proxy — so the scenario cannot score the property ("novelty") the paper exists to evaluate. Exact_match/quasi_exact_match on a free-form generation task are near-zero by construction and contribute noise, not signal.
- metric_notes/scimon_annotator_notes.md and scimon_eval_metrics_notes.md are empty stub files (1 line each). Evaluation design, rubric provenance, and metric justification fully undocumented.
- annotators=None in run_spec despite scenario docstring pointing at annotator_notes.md for novelty-sensitive scoring. Disconnect between intent and wiring.
- {'The "input" field from the dataset (e.g. "grammar is used for OtherScientificTerm") is surfaced verbatim in the prompt as "Relationship': '...". The raw "OtherScientificTerm" token is an IE-extractor tag, not natural language, and leaks evaluation-pipeline metadata into the model\'s context; a paper-faithful prompt would either translate the relation into prose or drop the type token.'}
- max_tokens=512, temperature=0.7 are not justified against the paper. Gold rel_sent targets are single sentences (~20–40 tokens); 512 invites verbose multi-clause outputs that will be penalized by BLEU/ROUGE against a terse reference, inflating the mismatch between automatic metrics and actual idea quality.
- Scenario drops output, neg_sample, forward, cos_sim, and annotation flag fields. Skipping neg_sample is defensible (not in gold_subset) but "output" (the short entity phrase) is discarded without noting that some SciMON evaluations target that field instead of rel_sent — choice of target is a faithfulness decision that should be documented.
- Only 10 instances were run in trial_10inst (display_predictions.json has 10 entries) against the 194-instance gold_subset. Expected for a smoke test, but the scenario does not down-sample deterministically; full-run behavior over all 194 has not been audited here.
- Inspection of 10 predicted_text entries from gemini-2.5-flash-lite shows the model produces plausible single-sentence scientific-idea descriptions, but several restate the source entity definitionally (id113 "utilizing Generic to ensure the generated programs are executable", id125 "introduces toxic span detection") — which will score well on BERTScore/ROUGE against rel_sent but does not test novelty at all. This is the exact failure mode the paper's human eval is designed to catch and the scenario does not.

### `sdat`
_sdat_annotator_notes.md is empty (0 lines) — only sdat_eval_metrics_notes.md carries the methodology.
Paper cross-check: S-DAT (Haase, Hanel, Pokutta 2025, AAAI/ACM AIES, arXiv:2505.09068) extends the
original DAT (Olson et al. 2021 PNAS, https://www.pnas.org/doi/10.1073/pnas.2022340118) from GloVe/English
to granite-embedding-278m-multilingual across 11+ languages. Full methodology was not extractable from
the arXiv abstract or PDF via WebFetch; cross-referenced against sdat_scenario.py docstring and
metric_notes/sdat_eval_metrics_notes.md, which cite the paper consistently (100 trials justification,
IBM granite embedding, pairwise cosine dissimilarity over 45 pairs, percentile calibration to
Olson N=8,572 dataset, correlations r=.60-.67 with DAT, r=.13-.27 with AUT). The scenario is correctly
specified on paper; the run_spec simply has not implemented the metric yet — it ships with a
placeholder exact_match that cannot score this task. Until the blocking items above are fixed,
sdat results are not interpretable.
Same caching/determinism artifact as dat.yaml — likely a HELM-wide issue for num_trials-style
identical-prompt benchmarks (also affects any future aut/aut-like repeated-trial scenarios)._

**Gaps:**
- Metric not implemented (blocking): run_specs/sdat_run_specs.py:37 registers only BasicGenerationMetric(['exact_match']) with no references=[] — exact_match is definitionally 0 for every instance. The paper's scoring (scenarios/sdat_scenario.py:26-31 and metric_notes/sdat_eval_metrics_notes.md:10-22) requires IBM granite-embedding-278m-multilingual to compute mean pairwise cosine dissimilarity over C(10,2)=45 word pairs, then optionally scale×100 and compare to S-DAT percentile table (5%=72.17, 50%=79.11, 95%=86.59). None of this is wired up. The run_spec itself flags this as a TODO at line 35-36.
- No word extractor: metric_notes/sdat_eval_metrics_notes.md:52-67 specifies parsing numbered lists / bullets / comma-separated forms into exactly 10 tokens. display_predictions.json for gemini-2.5-flash-lite shows the expected markdown-numbered form ('1.  Sun\n2.  Silence\n...') with a preamble ('Here are 10 words...') that must be stripped. No extractor exists; the paper's 10-word contract is unverifiable against current output.
- Observed determinism across trials (severe reliability defect): trial_10inst/display_predictions.json is byte-identical across id2, id16, id26, id54, id55, id73, id75, id86, id93, id95 — same 'Sun/Silence/Feather/Justice/Ocean/Dream/Stone/Music/Shadow/Kindness' response 10/10 times at temperature=0.7, num_outputs=1. S-DAT's statistical-reliability justification (scenarios/sdat_scenario.py:87-98, 100 instances) collapses to a single sample. Same issue as dat.yaml: HELM request cache is almost certainly keying only on prompt text, which is identical across all 100 instances by design.
- Embedding drift risk: metric_notes explicitly specifies granite-embedding-278m-multilingual (the S-DAT paper's model — 278M params, 11+ languages). If an implementer swaps in all-MiniLM/all-mpnet/GloVe (as happened for dat), scores are no longer on the S-DAT human-calibrated scale and percentile comparisons break. The run_spec comment is clear but the guard is only a TODO.
- Prompt faithfulness: scenarios/sdat_scenario.py:73-81 uses the original Olson et al. 2021 English prompt verbatim, which the S-DAT paper carries forward for English; correct. No multilingual variants are generated despite the paper's 11+ language scope — acceptable as a scoped English subset but worth noting the benchmark only probes the English slice of S-DAT.
- Decoding: temperature=0.7, num_outputs=1, max_tokens=512. No explicit top_p; paper does not specify generation hyperparameters for LLM evaluation, so no clear deviation, but 512 tokens is generous for a 10-word output and allows long preambles (observed) that the (missing) extractor must handle.
- Human baseline not surfaced: percentiles from Haase et al. 2025 validation (N=8,572 from Olson et al. data) are documented in the scenario docstring and metric_notes but not loaded into references or stats, so per-response percentile rank is not computable from current output shape.

### `simile_generation`
_Data loading is correct (150-row SimileEMNLP.csv, rows with '------' Human1 filtered, test split only) and matches the paper's evaluation data. The faithfulness failure is in adaptation: the zero-shot prompt with no output-shape constraint causes instruction-tuned models (all Gemini/Gemma variants inspected) to emit 3-5 bulleted simile options with commentary rather than a single rewritten sentence; combined with the removal of the bracket markers (which in the source data indicate the literal target to replace), the BLEU/ROUGE metrics are an extremely weak signal. This is a blocking prompt-format + output-shape issue, not a data issue._

**Gaps:**
- Prompt is a generic rewrite instruction (scenarios/simile_generation_scenario.py:73-77) that does not constrain output shape; paper's SCOPE model is a fine-tuned BART producing a single rewritten sentence — instruction-tuned chat models here produce multi-option menus instead (benchmark_output/runs/gemini_flash/.../display_predictions.json id114/id62/id33/id40/id86/id71/id134 all return 'Here are a few options...' with bullet lists; gemma_3_27b identical pattern)
- Scenario strips the bracket markers that delimit the target literal word (scenarios/simile_generation_scenario.py:70 literal.replace('[','').replace(']','')) — paper's SCOPE treats the bracketed token as the literal verb/phrase to be rewritten; removing brackets discards the signal the paper relies on and the prompt never tells the model which word to replace
- max_tokens=512 and temperature=0.7 with no stop_sequences (run_specs/simile_generation_run_specs.py:29-31) permit and encourage the long multi-option essay outputs observed, making surface-overlap metrics against one-line references largely meaningless
- Metrics are surface-overlap only: exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4 (run_specs:35). Paper's headline evaluations are (a) human pairwise preference vs literary experts (SCOPE wins 37%), (b) novelty rate (88% novel), (c) downstream story-evocativeness human judgment — none of these have an analogue here
- No LLM judge / metaphoricity / novelty / semantic-similarity annotator (annotators=None, run_specs:44); the paper's core claim cannot be reproduced from the configured metrics
- metric_notes/simile_generation_annotator_notes.md and metric_notes/simile_generation_eval_metrics_notes.md are both empty (1-line files) — no documentation of metric choice, BLEU variant, or reference handling
- max_train_instances=0 with empty adapter instructions (run_specs:23,27) — paper fine-tunes BART on parallel literal-simile corpus; zero-shot chat with in-scenario prompt is not directly comparable and the current prompt produces verbose menus rather than a single simile
- References are Human1/Human2 only (scenarios:82-84); paper also evaluates against the SCOPE system output — not a faithfulness bug per se, but BLEU against just two references undercounts valid paraphrases

### `splat`
_Data loading and split are fine (all 975 test puzzles, no train leakage). Core faithfulness problem is twofold: (1) the scenario silently reduces the paper's defining multi-turn interactive protocol to a single-turn monologue, and (2) the metric stack (BLEU/ROUGE/BERTScore/EM) cannot validly score lateral-thinking explanation correctness — the paper uses an LLM judge for exactly this reason. Current pipeline will produce numbers, but they will not track the construct the paper claims to measure. Truncation at 512 tokens further corrupts the n-gram scores._

**Gaps:**
- scenarios/splat_scenario.py:22-25 collapses paper's multi-turn player-judge framework into a single-turn direct-explanation task; paper's core contribution is the interactive Q&A protocol with intermediate question-answering accuracy AND final scenario accuracy — the former is impossible to measure here
- run_specs/splat_run_specs.py:35 scores with BLEU/ROUGE/F1/exact_match/quasi_exact_match + BERTScore; paper uses a judge model (WizardLM-2) to grade final scenario accuracy against reference — n-gram overlap on free-form lateral-thinking explanations is near-meaningless (different surface forms can both be correct)
- run_specs/splat_run_specs.py:29 max_tokens=512 is truncating long outputs mid-sentence — display_predictions.json id312 and id840 end abruptly ('preying on', 'chosen for the secu'), biasing overlap metrics downward
- run_specs/splat_run_specs.py:30 temperature=0.7; paper's evaluation protocol implies deterministic scoring against reference — non-determinism adds variance with no stated rationale
- scenarios/splat_scenario.py:109 no difficulty-stratified reporting despite scenario supporting easy/medium/hard; paper reports per-difficulty breakdowns (Table results)
- scenarios/splat_scenario.py:86-103 repo discovery via hardcoded path list (/tmp/splat, ../LateralThinking) is brittle; dataset loading tied to session-local clone
- metric_notes/splat_annotator_notes.md and splat_eval_metrics_notes.md are both empty (1-line/0-content) — no rubric, no judge prompt, no scoring definition
- display_predictions.json: models produce markdown-heavy, multi-paragraph speculative explanations with headers/bullets — ROUGE/BLEU against a terse reference answer will systematically under-score correct-but-verbose responses

### `sudoku_bench`
_Scenario data loading (SakanaAI/Sudoku-Bench HF, split='test'), board ASCII formatting, visual-element serialization, and 81-digit reference solution are all faithful to the paper/repo. However the e2eb48b7 escape patch left a user-visible 'r{x}c{y}' artifact in the prompt (should be literal 'rxcy'). The run config is non-viable: max_tokens=512 truncates every trial output before the <ANSWER> tag, no answer extractor is applied, temp=0.7 is non-deterministic, and only the challenge_100 subset is exercised. All 10 trial predictions are mid-reasoning truncations; current config yields a guaranteed 0% score that reflects harness failure rather than model ability._

**Gaps:**
- scenarios/sudoku_bench_scenario.py:125 literal '{{x}}' and '{{y}}' (post e2eb48b7 .format escape) render to the model as '{x}' and '{y}' — the paper's instruction is the concrete token 'rxcy'; current prompt ships 'r{x}c{y}' which is a format-placeholder-looking string that may confuse LLMs and does not match the official ONE_SHOT_VARIANT_PROMPT text
- run_specs/sudoku_bench_run_specs.py:29 max_tokens=512 is grossly insufficient: the expected answer alone is up to 81 digits, and the prompt explicitly solicits reasoning before the <ANSWER> tag; all 10 trial predictions in display_predictions.json are truncated mid-reasoning with no <ANSWER> tag emitted
- run_specs/sudoku_bench_run_specs.py:30 temperature=0.7 contradicts deterministic exact-match scoring against a single canonical 81-digit solution; paper reports single-shot average solve rate — greedy (temp=0) is standard
- run_specs/sudoku_bench_run_specs.py:35 scoring uses BasicGenerationMetric exact_match/quasi_exact_match/f1/rouge/bleu on raw generation — no <ANSWER>...</ANSWER> extractor is registered, so exact_match compares full CoT against the 81-digit solution and will always score 0 even when the model is correct
- run_specs/sudoku_bench_run_specs.py:18 args={} defaults to subset='challenge_100' only; nikoli_100 and ctc subsets documented in the scenario docstring are never instantiated, so the run cannot reproduce the paper's three reported subset results
- scenarios/sudoku_bench_scenario.py:169 all instances assigned TEST_SPLIT, but max_train_instances=0 in AdapterSpec is fine; however the docstring calls it 'ONE_SHOT_VARIANT_PROMPT' yet no worked example is provided — it is effectively zero-shot, inconsistent with the 'one-shot' naming
- metric_notes/sudoku_bench_annotator_notes.md and sudoku_bench_eval_metrics_notes.md are both empty (1 line) — no documented rubric, cell-level accuracy plan, or human baseline
- benchmark_output/runs/trial*/sudoku_bench_model=google_gemini-2.5-flash-lite/display_predictions.json: 10/10 predictions in trial_10inst are cut off mid-reasoning (no closing <ANSWER>), confirming max_tokens is blocking all answer emission

### `ttcw`
_As implemented, ttcw is a judgment-alignment task on severely under-specified stimuli (plot summaries), not a creativity benchmark. The original TTCW paradigm requires full story text and, for an LLM-as-writer evaluation, the model under test should generate stories while TTCW questions act as the rubric. Current wiring uses the evaluated model as a judge of unreadable stimuli and still mislabels itself as creativity. Prompt double-'Answer:' artifact confirms the MC-joint adapter and bespoke scenario prompt were not co-designed._

**Gaps:**
- Construct inversion: the original TTCW (Chakrabarty et al. 2024) has expert humans read a *full story* and answer 14 binary craft questions to evaluate *the story's creativity*. scenarios/ttcw_scenario.py:59-71 instead feeds the evaluated model a plot_summary plus the question and scores it against the expert's verdict — the evaluated model is being used as a judge of stories it cannot actually read, not as a creative writer. This is the wrong direction for a creative-writing benchmark.
- Stimulus is plot_summary, not the story. The Salesforce/ttcw_creativity_eval HF dataset only ships plot_summaries (full text is referenced by external URLs, noted in docstring line 27). TTCW questions such as 'sophisticated use of idiom or metaphor or literary allusion' or 'narrative pacing' are undecidable from a 1–2 sentence plot summary; the task as constructed is near-unanswerable.
- Adapter/prompt conflict: scenarios/ttcw_scenario.py:68-71 already writes a bespoke prompt ending in 'Answer:', but run_specs/ttcw_run_specs.py:21-31 uses ADAPT_MULTIPLE_CHOICE_JOINT with output_prefix='Answer: ', producing a doubled 'Answer:' tail and A./B. options appended after the scenario's own answer line (verified in scenario_state.json:66). Prompt shape is not what either the scenario or the adapter alone was designed for.
- Temperature 0.7 (run_specs/ttcw_run_specs.py:30) for a binary Yes/No classification prediction is inappropriate; original TTCW LLM-as-judge work uses deterministic settings.
- Instance construction duplicates stimulus: scenarios/ttcw_scenario.py:64 creates one instance per expert annotation (3 per dimension) with identical input but potentially different 'correct' label, so the same prompt appears up to 3× with conflicting gold — MultipleChoiceClassificationMetric will score these as independent items, inflating denominator and masking inter-rater disagreement rather than modelling it.
- metric_notes/ttcw_annotator_notes.md and metric_notes/ttcw_eval_metrics_notes.md are both empty (1 line each); no documentation of judge setup, majority-vote aggregation, or the construct mismatch.
- Trial run used max_eval_instances=1 (scenario_state.json:18); only 1 of 2,016 constructed instances executed, so output_shape_sane is only weakly verified. display_predictions.json shows 'B. No' which MC joint parses correctly, but single-sample coverage is inadequate.
- groups=['creativity','ttcw'] (run_specs/ttcw_run_specs.py:43) places this under creativity, but the task as implemented measures judgment-alignment with experts on under-specified stimuli, not creative generation.

### `unfun_corpus`
_Scenario loads test_unique_pairs_no_leakage.tsv (375) and val_unique_pairs_no_leakage.tsv (186) from the paper's GitHub repo — splits match the paper. Prompt wording 'You are a helpful assistant that edits humorous headlines to make them realistic' matches the paper's hit_llm_generation_v2.py system message in structure. Main faithfulness breaks: (1) zero-shot vs paper's 8-shot, (2) decoding settings produce verbose multi-candidate outputs (confirmed in display_predictions.json across gemini-2.5-flash and others) that cannot be fairly scored by lexical metrics, (3) no humor classifier / human-rating surrogate. arxiv full text not extractable via WebFetch (PDF binary)._

**Gaps:**
- Scenario is zero-shot but paper uses 8-shot prompts with randomly sampled high-quality human edits; the scenario's own docstring explicitly flags this divergence.
- Chat prompt is collapsed into a single string ('You are a helpful assistant... \n\n{headline}') rather than a true system+user message pair, and max_train_instances=0 prevents HELM from constructing few-shot demonstrations even if training data were available.
- temperature=0.7 with max_tokens=512 and no stop_sequences encourages verbose, multi-option, markdown-formatted responses — display_predictions.json shows the majority of Gemini outputs are long 'Here are a few options...' lists rather than a single edited headline.
- exact_match / quasi_exact_match will be near zero because predictions contain preambles, multiple candidate headlines, bold/italic markdown, and commentary; the reference is a single short headline.
- BLEU-1/BLEU-4/ROUGE-L/F1 will be heavily diluted by the boilerplate and multi-candidate text; no post-processor extracts a single headline from the model response.
- Paper's evaluation protocol also includes edit distance (token-level similarity to input), a learned humor classifier, and human ratings (realness, funniness, grammaticality, coherence); none are implemented — only surface-form lexical metrics.
- No annotator configured (annotators=None); metric_notes/unfun_corpus_annotator_notes.md and metric_notes/unfun_corpus_eval_metrics_notes.md are empty files.
- Both TEST (375) and VALIDATION (186) splits are emitted and scored together without a max_eval_instances cap; paper reports on the test split.

### `yesbut`
_Prompt wording matches paper's WHYFUNNY_PROMPT and scenario construction is correct in principle, but trial outputs show neither image nor prompt reached the model — every prediction is an identical refusal with empty base64_images. Do not run full 1,084-instance eval until multimodal delivery is fixed and verified. Metrics set also needs semantic-level additions (BERTScore / LLM judge) because the references are free-form sentences._

**Gaps:**
- benchmark_output/runs/trial_10inst/yesbut_model=google_gemini-2.5-flash-lite/display_predictions.json rows 1-71 show identical 'I'm sorry, but you haven't provided any text for me to work with' refusals across all 10 instances; base64_images is [] for every row — the image (and apparently the question text) is not reaching the model at inference time. Same pattern in benchmark_output/runs/trial/yesbut_model=google_gemini-2.5-flash-lite/display_predictions.json.
- scenarios/yesbut_scenario.py:107-116 builds MultimediaObject with image/jpeg + text/plain, but the trial outputs confirm the multimodal payload is not being delivered — identical to the ii_bench failure mode. The text prompt '\nWhy is this image funny/satirical?' matches the paper's WHYFUNNY_PROMPT, so phrasing is faithful; delivery is broken.
- run_specs/yesbut_run_specs.py:30 sets temperature=0.7; paper's Satirical Image Understanding task is not specified as stochastic — typical VLM generation eval uses temperature=0 (greedy). 0.7 injects variance into BLEU/ROUGE scores.
- run_specs/yesbut_run_specs.py:35 uses BasicGenerationMetric with {exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4}. Paper (EMNLP 2024) benchmarks with automated AND human evaluation, and typically includes BERTScore / GPT-4-as-judge for open-ended satire explanation (free-form, ~1-2 sentence references). exact_match and quasi_exact_match on an open-ended generation task are near-zero by construction and not informative; BERTScore and a semantic judge are missing.
- metric_notes/yesbut_annotator_notes.md and metric_notes/yesbut_eval_metrics_notes.md are both empty (1 line) — no recorded metric rationale or judge spec.
- scenarios/yesbut_scenario.py:96-98 silently drops ~9 examples with no difficulty label; default difficulty='all' is fine but the drop is undocumented in metric_notes.
- scenarios/yesbut_scenario.py:73-83 loads split='train' for a test evaluation; HuggingFace dataset bansalaman18/yesbut ships everything under 'train' so this is the only option, but all 1,084 satirical images are being placed in TEST_SPLIT with no held-out set (acceptable for eval-only, but noted).
- Paper frames three tasks (Detection, Understanding, Completion); only Understanding (WHYFUNNY) is implemented — scenario header (lines 12-14) correctly scopes to this, but the 1,463 non-satirical distractors and completion pairs are unused, so only ~1/3 of the benchmark surface is covered.

## MOSTLY_FAITHFUL (51)

| Benchmark | Prompt | Data | Metric | Judge | Output | Paper |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `aidanbench` | partial | true | partial | partial | true | https://openreview.net/forum?id=fz969ahcvJ |
| `amuse_chord_generation` | true | true | partial | n/a | partial | https://arxiv.org/abs/2412.18940 |
| `analobench` | true | true | partial | n/a | partial | https://arxiv.org/abs/2402.12370 |
| `arastories` | true | true | partial | partial | true | https://arxiv.org/abs/2407.07551 |
| `arn` | true | true | partial | n/a | true | https://arxiv.org/abs/2310.00996 |
| `c3_crosstalk` | partial | true | partial | n/a | true | https://arxiv.org/abs/2207.00735 |
| `cap_aut` | true | true | partial | true | true | n/a (in-house protocol — human/uva_pilot/scripts/run_all_… |
| `cap_design` | true | true | partial | true | unknown | unavailable (internal UVA pilot protocol, no public URL) |
| `cap_metaphor` | partial | true | partial | true | unknown | n/a (internal CAP / UVA pilot Study 3 — no public paper c… |
| `cap_sctt` | true | true | partial | true | true | unpublished (Beaty CAP pilot / UVA Study 3; no public pap… |
| `cap_story` | partial | true | partial | true | true | https://osf.io/preprints/psyarxiv/t63dm |
| `chinese_homophonic_puns` | true | true | partial | n/a | true | https://arxiv.org/abs/2405.15818 |
| `creativemath` | true | true | false | partial | partial | https://arxiv.org/abs/2407.14910 |
| `critics_story` | partial | true | false | partial | true | https://arxiv.org/abs/2410.02428 |
| `crowd_vote` | true | partial | false | false | true | https://arxiv.org/abs/2509.09702 |
| `cue_word_story` | true | true | partial | partial | true | https://arxiv.org/abs/2411.02316 |
| `dat` | true | true | false | n/a | partial | https://aclanthology.org/2023.findings-emnlp.858/ |
| `dat_creative_writing` | true | true | false | partial | false | https://doi.org/10.1038/s41562-024-02046-9 |
| `data_narrative` | partial | true | false | false | true | https://aclanthology.org/2024.emnlp-main.1073/ |
| `dpt` | partial | true | false | false | true | https://arxiv.org/abs/2502.03253 |
| `eqbench_creative_writing_v3` | true | true | partial | partial | true | https://arxiv.org/abs/2509.02534 |
| `future_ideas` | partial | true | false | partial | partial | https://arxiv.org/abs/2409.06185 |
| `fuxibench` | true | true | false | false | true | https://arxiv.org/abs/2503.15837 |
| `gauss` | true | true | partial | partial | true | https://arxiv.org/abs/2509.18122 |
| `geo_story` | partial | true | false | partial | true | https://arxiv.org/abs/2411.07320 |
| `grapheval_iclr` | true | true | false | false | true | https://arxiv.org/abs/2503.12600 |
| `graphrag_bench` | partial | true | false | false | true | https://arxiv.org/abs/2506.05690 |
| `historical_analogy` | partial | true | partial | partial | partial | https://aclanthology.org/2025.acl-long.200/ |
| `idrbench` | partial | partial | false | false | true | https://arxiv.org/abs/2507.15736 |
| `liveideabench` | true | true | false | false | true | https://arxiv.org/abs/2412.17596 |
| `llm_discussion` | true | partial | false | false | true | https://arxiv.org/abs/2405.06373 |
| `mixassist` | partial | true | partial | n/a | partial | N/A — no paper; dataset at https://huggingface.co/dataset… |
| `music_theory_bench` | true | true | partial | n/a | false | https://arxiv.org/abs/2402.16153 |
| `outline_to_story` | false | true | partial | n/a | true | https://arxiv.org/abs/2101.00822 |
| `poetmt` | partial | partial | partial | true | false | https://arxiv.org/abs/2408.09945 |
| `pollux_creativity` | partial | true | false | false | true | https://arxiv.org/abs/2505.24616 |
| `pron_vs_prompt` | true | true | partial | partial | partial | https://arxiv.org/abs/2407.01119 |
| `pun_eval` | true | true | partial | partial | true | https://arxiv.org/abs/2404.13599 |
| `research_idea_execution` | true | true | false | false | true | https://arxiv.org/abs/2506.20803 |
| `scar` | true | true | partial | n/a | partial | https://aclanthology.org/2023.findings-emnlp.160 |
| `science_analogies` | true | true | false | n/a | partial | https://aclanthology.org/2022.inlg-main.25/ |
| `showerthoughts` | partial | true | partial | partial | false | https://aclanthology.org/2024.starsem-1.23/ |
| `slang_generation` | true | true | false | partial | partial | https://arxiv.org/abs/2502.XXXXX |
| `sonnet_or_not_bot` | unknown | unknown | unknown | unknown | unknown | https://arxiv.org/abs/2406.18906 |
| `speak_to_structure` | partial | true | false | partial | true | https://arxiv.org/abs/2412.14642 |
| `ss_gen` | true | true | partial | partial | true | https://arxiv.org/abs/2406.15695 |
| `story_generation_rocstories` | partial | false | partial | n/a | false | https://arxiv.org/abs/2303.08991 |
| `thenextchapter` | partial | true | false | n/a | true | https://arxiv.org/abs/2301.09790 |
| `tinyfabulist` | true | true | partial | partial | true | https://arxiv.org/abs/2504.20605 |
| `tinystories` | true | true | partial | partial | true | https://arxiv.org/abs/2305.07759 |
| `writingbench` | true | partial | false | false | true | https://arxiv.org/abs/2503.05244 |

### `aidanbench`
_Scenario and run_spec faithfully preserve AidanBench's 66-question set, intent (diverse+coherent idea generation), and judge model choice. Primary deviation is the documented single-turn batch adaptation plus the missing novelty/combined-score metric, which means reported scores proxy diversity+coherence rather than the paper's stopping-count. Safe to run as a creativity diversity/coherence probe, not as a reproduction of AidanBench Score. Paper PDF access was blocked (OpenReview 403); cross-check leaned on GitHub README and annotator notes._

**Gaps:**
- Protocol adaptation: scenario uses single-turn batch prompting (N=30 responses in one shot) rather than the paper's iterative multi-turn loop that stops on coherence<=15 or novelty<=0.15 (scenarios/aidanbench_scenario.py:19-22, 130-136). This is explicitly documented as an adaptation but changes the evaluation semantics.
- Primary metric drift: paper's headline metric is AidanBench Score = count of responses passing both coherence and novelty thresholds. run_specs/aidanbench_run_specs.py:49-52 only registers Self-BLEU (diversity proxy) and a per-response coherence_score via LLM judge. No novelty/embedding-similarity metric and no threshold-based AidanBench Score is computed, despite metric_notes/aidanbench_eval_metrics_notes.md:61-75 describing it.
- Novelty metric missing: paper uses embedding cosine similarity (text-embedding-ada-002) with tau=0.15; no embedding metric is wired up in run_specs/aidanbench_run_specs.py.
- Judge prompt wording differs slightly from the paper/GitHub template in metric_notes/aidanbench_eval_metrics_notes.md:47-57 (paper template embeds the question; run_specs rubric in aidanbench_run_specs.py:15-24 does not reference the question and uses a 5-anchor Likert description). Same 0-100 scale and coherence intent, so task is preserved.
- Judge model: run_specs/aidanbench_run_specs.py:58 uses openai/o1-mini matching paper, but judge_max_new_tokens=64 may truncate reasoning-model outputs; temperature=0.0 is fine (o1 ignores temperature).
- Output shape: display_predictions.json shows numbered markdown lists with 15-30 items and coherence_score annotations populated (0-100, e.g. 72, 78, 82). Visually consistent with intended batch-mode output.

### `amuse_chord_generation`
_Prompt text and dataset (254 suno.wiki keywords, 4 bars, 30 progressions, paper-listed keys/modes) faithfully mirror the Amuse paper's Appendix A.2. Two config issues are blocking: max_tokens=512 truncates outputs mid-progression in the sample predictions, and num_outputs=30 double-counts against the batch-prompt asking for 30 progressions in one completion. Metric args patch (self_bleu + JSD n=2) aligns with paper's bigram JSD but misses unigram JSD; temperature and model choice deviate for documented reasons._

**Gaps:**
- run_specs/amuse_chord_generation_run_specs.py:28 num_outputs=30 conflicts with paper's batch-prompting semantics: the paper generates 30 progressions in ONE completion (the prompt itself asks for {num_progressions} diverse progressions), but the adapter additionally samples 30 completions per instance, inflating calls 30x and confusing Self-BLEU granularity.
- run_specs/amuse_chord_generation_run_specs.py:29 max_tokens=512 is too low for 30 four-bar progressions; sample predictions in display_predictions.json (e.g., id158, id83, id101, id118, id228) are visibly truncated mid-progression, which will bias Self-BLEU/JSD downward by dropping late outputs.
- run_specs/amuse_chord_generation_run_specs.py:30 temperature=0.7 diverges from the paper's temperature=1.0 ('to promote creativity and diversity').
- run_specs/amuse_chord_generation_run_specs.py:36 JSDMetric only configured with n=2 (bigram); paper reports BOTH unigram (Amuse 0.27) and bigram (Amuse 0.46) JSD. Unigram JSD is missing from the metric set.
- Paper evaluates against Hooktheory reference distribution for JSD; metric_notes.md acknowledges this but run_spec does not pass a hooktheory_path arg — confirm metrics/jsd_metric.py loads it internally before the full run.
- Paper LLM is gpt-4o-2024-05-13; this run uses anthropic/claude-haiku-4.5 via OpenRouter (documented constraint due to Gemini's num_outputs<=8 cap). Results will not be directly comparable to paper's 0.30/0.61 Self-BLEU numbers.
- metric_notes/amuse_chord_generation_annotator_notes.md is empty (1 line, effectively blank); no annotator guidance is defined, though this task has no human-judge component so it is acceptable.

### `analobench`
_Prompt template faithfully mirrors code/t1.py and the dataset/subset load is consistent with the paper. Main drift is adapter-level: generation-mode decoding at temp=0.7 plus a misaligned metric bundle, compounded by models occasionally answering with digits instead of A-D, which will understate exact_match. Valid to run tomorrow but results should be interpreted on exact_match/quasi_exact_match only._

**Gaps:**
- run_specs/analobench_run_specs.py:30 temperature=0.7 for a deterministic MC task; paper implies greedy decoding (temp=0)
- run_specs/analobench_run_specs.py:35 metric bundle includes f1_score/rouge_l/bleu_1/bleu_4 which are not meaningful for single-letter MC answers; exact_match/quasi_exact_match are the correct signal
- run_specs/analobench_run_specs.py:21 uses ADAPT_GENERATION rather than ADAPT_MULTIPLE_CHOICE_JOINT; predictions include digits ('3','4') alongside letters (A-D), e.g. gemini_lite display_predictions.json ids id299/id76/id278, producing false exact_match misses vs. gold 'A'-'D'
- metric_notes/analobench_annotator_notes.md and metric_notes/analobench_eval_metrics_notes.md are empty (0 bytes); paper's intended eval setup/metrics were not transcribed for cross-check
- scenarios/analobench_scenario.py:37 loads T1S1-Subset split='train'; verify paper uses this subset as its test set (name collision risk)

### `arastories`
_Scenario loads dataset correctly and uses dataset prompts verbatim; judge model and 5 dimensions match paper. Drift is in judge-call structure (per-dimension vs single-call) and custom rubrics not present in paper. Sample predictions show well-formed Arabic stories with all 5 integer 1-5 annotations populated — safe to run at full scale._

**Gaps:**
- Judge is split into 5 separate per-dimension annotator calls with custom rubrics (run_specs/arastories_run_specs.py:88-147), but paper uses a single GPT-4 call scoring all 5 dimensions at once with the exact template in metric_notes/arastories_annotator_notes.md:13-23.
- Custom 1-5 rubrics per dimension (run_specs/arastories_run_specs.py:15-63) are not from the paper; paper gives dimension definitions only and asks for 'scores directly without explanations'.
- Full dataset is ~2,996 instances across 3 dialects (scenarios/arastories_scenario.py:20-22); paper's test subset is 20 prompts/dialect — running full set is a superset but may inflate judge cost.
- metric_notes/arastories_eval_metrics_notes.md is empty (0 lines) — no eval-metrics spec documented.
- Adapter max_tokens=512, temp=0.7 (run_specs/arastories_run_specs.py:83-84) not verified against paper generation config.

### `arn`
_Prompt text and four-way subset partitioning (near/far x high/low) faithfully match paper Appendix E.2 and the 1,095-triple dataset. Critical risk: reference outputs are '1'/'2' but model emits full templated sentences (confirmed in gemini_pro and trial_10inst display_predictions), so exact_match likely fails without a choice-extraction step. Judge n/a since task is binary multiple choice._

**Gaps:**
- run_specs/arn_run_specs.py:30 sets temperature=0.7; paper's binary-accuracy eval typically uses greedy/low-temp decoding, introducing noise
- run_specs/arn_run_specs.py:35 emits extra generative metrics (f1, rouge_l, bleu) that are meaningless for a 2-choice task; only exact_match is task-appropriate
- metric_notes/arn_annotator_notes.md and metric_notes/arn_eval_metrics_notes.md are empty (0 lines) - no documented parsing/scoring rationale
- scenarios/arn_scenario.py:144-150 references are literal '1'/'2', but predictions are free-form template '{{narrative_X, because...}}' (see display_predictions.json) - exact_match will score 0 unless HELM's quasi_exact_match or a post-parser extracts the choice

### `c3_crosstalk`
_Data loading and 10-context/10-gold split are faithful to the C3 repo layout. Core faithfulness issue is metric selection: BLEU/ROUGE without Chinese tokenization and inclusion of EM/F1 will produce misleading numbers. Prompt is a reasonable original construction since paper tested seq2seq, not LLMs; documented honestly in the docstring._

**Gaps:**
- Paper used seq2seq models, not LLMs; no canonical LLM prompt exists. Chinese instruction at c3_crosstalk_scenario.py:120-124 is a reasonable custom zero-shot prompt (documented as such at lines 26-35).
- run_specs (c3_crosstalk_run_specs.py:35) adds exact_match / quasi_exact_match / f1_score which are inappropriate for open-ended Chinese dialogue; paper reports BLEU, ROUGE, distinct-1/2. bleu_1/bleu_4/rouge_l are included (good) but distinct-1/2 (lexical diversity, emphasized in paper) is absent.
- BLEU/ROUGE in HELM BasicGenerationMetric are whitespace-tokenized and not Chinese-aware; scores will be near-zero/noisy against Chinese gold references.
- metric_notes/c3_crosstalk_annotator_notes.md and c3_crosstalk_eval_metrics_notes.md are empty (1 line each) — human-rating rubric (quality/humor/coherence) is undocumented despite scenario docstring referencing it.
- 50 test instances confirmed in display_predictions.json (id2, id4, id10, id11, id22, id27, id28, id31, id33, id38, id41 visible); split logic at scenario.py:99-114 correctly parses 20-utterance dialogues, takes lines 10-19 as gold.

### `cap_aut`
_Scenario prompt, 5-item list, and V7 judge rubric line up with Beaty's CAP pilot protocol; this is not backed by an arxiv paper but by the in-house UVA Study 3 pipeline (noted in scenarios/cap_aut_scenario.py:1-21). Quality-only scoring is an intentional HELM-time choice with novelty deferred to a post-hoc script — faithful to the protocol spec but the first-trial outputs are incomplete until that script runs. The trial_cap run also used max_eval_instances=3, clipping the 5-item battery._

**Gaps:**
- metric_notes/cap_aut_annotator_notes.md and metric_notes/cap_aut_eval_metrics_notes.md do not exist (only CAP-family notes missing from metric_notes/)
- scenarios/cap_aut_scenario.py:20 references human/uva_pilot/scripts/run_all_models_study3.py but that path does not exist in this repo, so the verbatim-item claim cannot be cross-checked locally
- run_specs/cap_run_specs.py:10-11 notes novelty is NOT scored inside HELM; only the V7 quality judge (1-7 'does it make sense') is run — novelty requires the post-hoc scripts/score_cap_novelty.py pipeline, which the first-trial run does not invoke
- benchmark_output/runs/trial_cap/cap_aut_model=*/ contains per_instance_stats.json + scenario_state.json but no display_predictions.json (HELM summarization step not run); only 3 instances (max_eval_instances=3) scored — not the full 5-item battery
- run_specs/cap_run_specs.py:39-40 admits the judge rates the whole semicolon-joined completion rather than per-idea as the human pipeline does, so AUT quality is coarser than the human V7 scoring

### `cap_design`
_Scenario and run spec are internally consistent with the documented UVA pilot Study 3 protocol: 5 design prompts, 3-6 semicolon-separated solutions, V7 quality judge (Gemini-3-flash, temp 0, 1-7 scale) shared across the CAP battery. Faithfulness is capped by missing metric_notes, absent pilot-run outputs, and the source study3 script not being available in this working tree for verbatim cross-check._

**Gaps:**
- No metric_notes/cap_design_annotator_notes.md or cap_design_eval_metrics_notes.md exist (metric_notes/ has no cap_* files).
- No benchmark_output/runs/*/cap_design_model=*/ directories exist — benchmark has not been executed, so output shape is unverified.
- Novelty metric is deferred to post-HELM aggregation (scripts/score_cap_novelty.py); only V7 quality judge runs in HELM — run_specs/cap_run_specs.py:88-105.
- Judge sees whole semicolon-joined completion, not per-idea (run_specs/cap_run_specs.py:36-40), diverging slightly from human pipeline which splits on ';'.
- Pool centroid for novelty uses UVA pilot proxy, not the 200-model ABC corpus (run_specs/cap_run_specs.py:13-15).
- No paper / pilot protocol doc could be fetched; item list traceable only to human/uva_pilot/scripts/run_all_models_study3.py which is not present in this checkout.

### `cap_metaphor`
_Scenario and run spec are internally consistent with the CAP battery design and the shared V7 quality rubric, judge model, and temp=0 settings match the other four CAP tasks. Main risks are documentation gaps (missing metric_notes, missing referenced human-pipeline scripts) and prompt framing that augments the bare Study 3 stem with task instructions. Output shape cannot be verified since no cap_metaphor run outputs exist yet._

**Gaps:**
- No dedicated metric_notes files — cap_metaphor_annotator_notes.md and cap_metaphor_eval_metrics_notes.md do not exist in metric_notes/; rubric lives only in run_specs/cap_run_specs.py:42-58 (shared V7)
- Referenced source-of-truth files not in tree: human/uva_pilot/scripts/run_all_models_study3.py (scenarios/cap_metaphor_scenario.py:13) and human/uva_pilot/scripts/old/score_quality_v7_pilot.py (run_specs/cap_run_specs.py:4) — cannot verify stems are verbatim or rubric matches human pipeline
- No display_predictions.json for cap_metaphor under benchmark_output/runs/*/ — benchmark has not been run yet, so output shape unverified
- Scenario wraps each stem with an added instruction block (scenarios/cap_metaphor_scenario.py:50-58) — departs from the bare-stem human administration; can inflate/suppress quality relative to human baseline
- Novelty (core CAP metric per run_specs/cap_run_specs.py:10-15) deferred to post-hoc scripts/score_cap_novelty.py — only quality is scored in-run; pool centroid still the UVA pilot proxy, not the 200-model ABC corpus
- CAP is an internal Beaty-lab pilot; no published paper URL is cited in source

### `cap_sctt`
_Items are reproduced verbatim from the UVA Study 3 script, prompt scaffold (3-6 ideas, semicolon-separated, novel+scientifically-possible framing) matches the human protocol, and the V7 quality rubric + Gemini judge are intentionally shared with the human scoring pipeline. Main divergences are whole-completion vs per-idea judging and the proxy novelty pool — both acknowledged in comments and non-blocking for a pilot run._

**Gaps:**
- scenarios/cap_sctt_scenario.py:2 docstring says 'Scientific Creative Thinking Task' but the task brief calls SCTT 'story/scenario creative thinking test' — verify canonical name with Beaty (items are the scientific-hypothesis stems either way).
- metric_notes/cap_sctt_annotator_notes.md and metric_notes/cap_sctt_eval_metrics_notes.md do not exist; only the shared run_specs/cap_run_specs.py V7 rubric is documented.
- run_specs/cap_run_specs.py:38-40 judges the whole semicolon-joined completion as a single V7-quality rating, whereas human score_quality_v7_pilot.py judges each idea individually — comment acknowledges this, but per-idea re-judge is not yet implemented.
- scripts/score_cap_novelty.py:24-28 uses the current UVA pilot pool as a proxy centroid rather than the promised 200-model ABC corpus, so novelty values are provisional.
- benchmark_output/runs/trial_cap/cap_sctt_model=google_gemini-2.5-flash-lite/ contains scenario_state.json but no display_predictions.json (HELM summarize likely not run yet).

### `cap_story`
_Prompt wording, adapter (temp=0.7, 512 tokens, single-response), and V7 judge (gemini-3-flash-preview, temp=0, 1-7 makes-sense rubric) cleanly match the documented CAP pipeline. Main faithfulness risk is the unverifiable triad list and the fact that the headline CAP creativity metric (novelty vs. pool centroid) is intentionally out-of-band; quality-only scoring understates the construct CAP-Story is designed to measure._

**Gaps:**
- Triads in scenarios/cap_story_scenario.py:21-27 are hand-written placeholders (pen/paper/story, key/door/lock, bridge/river/cross, mirror/face/reflection, shoe/path/walk) — docstring claims they are 'reproduced verbatim from human/uva_pilot/scripts/run_all_models_study3.py' but that file does not exist in the repo, so the exact Study 3 item set cannot be verified.
- Only the V7 'quality/makes-sense' rubric is scored in-run (run_specs/cap_run_specs.py:42-58, 88-104). Novelty — the primary CAP creativity DV — is deferred to post-hoc scripts/score_cap_novelty.py and is not part of this benchmark's reported metric.
- No metric_notes/cap_story_annotator_notes.md or cap_story_eval_metrics_notes.md exist; CAP battery is documented only in the cap_run_specs.py module docstring.
- display_predictions.json not produced for the trial run (trial_cap/cap_story_model=google_gemini-2.5-flash-lite/ contains scenario_state.json, per_instance_stats.json, stats.json only); shape verified via scenario_state.json instead.
- Trial run used max_eval_instances=1 (scenario_state.json:18); only 1 of 5 triads executed.

### `chinese_homophonic_puns`
_Prompt faithfully adapted from DuanzAI prompt.py zero-shot task_1 template; dataset loads full 1000 examples from the official GitHub raw task_1.json. Outputs across Gemini/Gemma runs produce clean 2-4 char Chinese predictions matching expected shape. Primary gap is metric: paper's fuzzy-similarity scoring is not implemented and standard HELM metrics are known-inappropriate (noted by the annotator notes themselves)._

**Gaps:**
- run_specs/chinese_homophonic_puns_run_specs.py:35 uses BasicGenerationMetric (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) instead of paper's custom SequenceMatcher+fuzzywuzzy fuzzy-similarity metric
- metric_notes/chinese_homophonic_puns_eval_metrics_notes.md is empty (0 lines) — metric spec undocumented
- run_specs/chinese_homophonic_puns_run_specs.py:30 temperature=0.7 is noisy for a deterministic short-extraction task (paper implies greedy/0-shot extraction)
- annotator_notes.md (line 49-53) explicitly warns standard exact_match and BLEU/ROUGE are inappropriate for Chinese 2-4 char extraction, yet run_spec uses exactly those

### `creativemath`
_Scenario faithfully reproduces CreativeMath's prompt, data, and instance expansion. The run_spec, however, collapses the paper's novelty-focused 3-stage jury evaluation into a single 1-5 correctness judge — losing the benchmark's headline Novelty Ratio and Novel-Unknown Ratio. Combined with a 512-token generation cap that truncates most solutions, the current configuration cannot reproduce paper numbers and even the correctness signal is unreliable. Runs in benchmark_output confirm truncation in predicted_text. Paper PDF/GitHub fetch returned only partial info; cross-check relied on annotator_notes.md._

**Gaps:**
- Generation prompt (scenarios/creativemath_scenario.py:48-62) matches the paper's load_novel_solution_generation_prompt verbatim including all 5 novelty criteria. Instance expansion (400 problems x k=1..n -> 605) follows the paper.
- max_tokens=512 (run_specs/creativemath_run_specs.py:44) is severely insufficient for competition math solutions (USAMO/IMO proofs). Predictions in benchmark_output/runs/trial_10inst/creativemath_model=google_gemini-2.5-flash-lite/display_predictions.json are visibly truncated mid-sentence at ~500 tokens (e.g. USAMO_1_k1, AMC_8_8_k1, AMC_8_1_k3 cut off before final answer), biasing both correctness and novelty judgements.
- temperature=0.7 is a reasonable default but the paper's per-model generation config is not explicitly matched; no canonical temp is set.
- Metric pipeline drift (major): paper prescribes a 3-stage pipeline — Stage 1 correctness (unanimous YES across 3 judges: Claude-3-Opus, Gemini-1.5-Pro, GPT-4), Stage 2 coarse novelty vs shown refs (majority YES), Stage 3 fine novelty vs withheld refs — producing Correctness Ratio, Novelty Ratio, Novel-Unknown Ratio, and two conditional ratios (metric_notes/creativemath_annotator_notes.md). run_specs/creativemath_run_specs.py only wires a single GenericLLMJudgeMetric with a 1-5 Likert correctness rubric and judge_model_name=openai/gpt-4. No novelty stage, no jury/voting, no YES/NO extraction, no withheld-solution comparison.
- Judge rubric shape differs: paper uses YES/NO binary per stage; HELM run uses 1-5 Likert for correctness only, so downstream Correctness Ratio cannot be recovered without a threshold decision.
- Paper's pre-eval cleanup instruction (remove transition sentences/justifications from model output before judging) is not implemented.
- metric_notes/creativemath_eval_metrics_notes.md is effectively empty (1 line); annotator_notes.md carries all the protocol detail.
- Output shape: predictions contain the expected long CoT solutions with instance_ids of form {competition}_{problem_id}_k{k} matching the scenario. Annotations show generic_llm_judge_llm_judge_correctness scores 1-5 — structurally sane but semantically mismatched to paper metric.

### `critics_story`
_Scenario is well-structured and dataset loading is correct; problem is the run_spec silently drops the judgment subset (where the real human-annotated ground truth lives) and replaces the CritiCS pairwise 3-dimension evaluation with a generic 1-5 literary-quality LLM-judge rubric. Output shape is sane (long story plans, integer scores 1-5) but the headline number does not measure what the CritiCS paper measures. Adapter settings (temp=0.7, 512 tokens, zero-shot) are reasonable for story generation._

**Gaps:**
- Scenario defines two subsets (generation, judgment) at scenarios/critics_story_scenario.py:95, but run_specs/critics_story_run_specs.py:30-33 instantiates with args={} and therefore only ever runs the 'generation' subset — the paired-judgment task with human ground truth from doc-storygen-v2 (Q1 label) is never executed, so the exact_match-style evaluation against human annotations is dead code.
- Generation instruction in scenarios/critics_story_scenario.py:52-58 is hand-authored ('Given the following story premise, write a detailed story plan…'). The scenario docstring admits the CritiCS paper uses a multi-agent pipeline (Leading/Revising Critics) rather than a single prompt, and no exact single-prompt equivalent exists in the paper — so this prompt is an adaptation, not a reproduction.
- The judgment prompt template (scenarios/critics_story_scenario.py:62-81) deviates from the CritiCS persona_comparision template documented in metric_notes/critics_story_annotator_notes.md:14-35: the repo template collapses the 4 questions (interesting/coherent/creative/premise-closeness) into a single 'which do you prefer' question, and asks for [[A]]/[[B]]/[[C]] verdicts rather than the multi-dimension A/B/C answer set.
- Metric is GenericLLMJudgeMetric with metric_name='llm_judge_quality' and a generic 1-5 literary-quality rubric (run_specs/critics_story_run_specs.py:15-24, 50). The CritiCS paper's actual evaluation is pairwise (Storyline A vs B) across Interesting/Coherent/Creative — a single-output 1-5 quality score does not reproduce that construct.
- Judge model is hardcoded to 'openai/gpt-4' (run_specs/critics_story_run_specs.py:57) matching the paper, but with judge_temperature=0.0 and max_tokens=512 — consistent with paper defaults. However the rubric itself ignores the paper's three CritiCS dimensions and the detailed creativity rubric (Originality/Structure/Ending/Text-quality) described in metric_notes/critics_story_annotator_notes.md:36-59.
- metric_notes/critics_story_eval_metrics_notes.md is empty (1 line) — no documentation of how llm_judge_quality was chosen over the CritiCS-native pairwise-3-dimension metric.
- Generation subset deduplicates by premise (scenarios/critics_story_scenario.py:112-118) yielding up to 6,932 instances from llm-aes/doc-storygen-v2; CritiCS paper used 300 premises from the DOC pipeline (doc-storygen-v2 is a superset), so there is no alignment with the paper's specific premise set.
- display_predictions.json eyeballed for gemini-2.5-flash and gemma-3-27b: instance_ids 'critics_gen_<id>' match the generation-subset format, predicted_text is long-form story plans (setting/characters/plot outline as promised), annotations carry single 'llm_judge_quality' integers in [1..5] — output shape internally consistent but tells us nothing about CritiCS dimensions.

### `crowd_vote`
_Prompt wording and instance count faithfully mirror the paper (300 = 100 brands x 3 task types, exact system + task templates). Key deviations: curated brand list (original proprietary), single-response LLM-as-judge replacing pairwise crowd Bradley-Terry ranking, and -- most importantly -- the implemented rubric is a generic 1-5 quality scale rather than the documented 4-dimension creativity rubric, so current outputs do not measure originality/brand_relevance/creative_potential/conciseness as intended. Output shape sane across inspected runs (gemma_3_1b, gemini_2.5_flash). Paper WebFetch not attempted (deferred tool); cross-check leaned on internal scenario/notes and arXiv id 2509.09702 references._

**Gaps:**
- Dataset substitution: paper uses 100 proprietary brand challenges from Springboards that are not publicly released. scenarios/crowd_vote_scenario.py:29-36 substitutes a curated 100-brand list covering the paper's 12 categories. Task types and counts (100 brands x 3 = 300) match, but specific stimuli differ from original study, so results are not directly comparable.
- Evaluation protocol replaced: paper's primary evaluation is pairwise crowd voting by 678 ad professionals with 11,012 comparisons ranked via Bradley-Terry (metric_notes/crowd_vote_annotator_notes.md:5-11). Scenario uses single-response LLM-as-judge scoring (run_specs/crowd_vote_run_specs.py:49-51). This is a documented adaptation but fundamentally changes the metric from relative preference to absolute quality.
- Judge rubric mismatch: metric_notes/crowd_vote_annotator_notes.md:18-60 specifies a 4-dimension rubric (originality, brand_relevance, creative_potential, conciseness, each 1-5, with GPT-4o as judge). run_specs/crowd_vote_run_specs.py:15-24 actually wires up a single generic 'llm_judge_quality' 1-5 rubric referencing clarity/helpfulness/accuracy/appeal - none of the four intended creativity dimensions are scored.
- Judge model mismatch: annotator notes specify GPT-4o; run_specs/crowd_vote_run_specs.py:57 uses openai/gpt-4 (non-o). Minor but not what notes prescribe.
- metric_notes/crowd_vote_eval_metrics_notes.md is empty (0-1 lines), so eval metric expectations are under-documented beyond annotator_notes.md.
- Prompts verbatim from paper: system prompt and three task templates in scenarios/crowd_vote_scenario.py:111-134 match paper wording (insights <=10 words, ideas/wild_ideas <=50 words).
- Output shape sane: display_predictions.json across gemma_3_1b and gemini_2.5_flash runs shows plausible brand-specific marketing copy with llm_judge_quality scores 3-5; 300 instances per model as expected.

### `cue_word_story`
_System + user prompts in scenarios/cue_word_story_scenario.py:70-83 reproduce the repository prompts from github.com/mismayil/creative-story-gen (src/prompts.py) verbatim, including the boring_theme anti-prompts per triad. Adapter is zero-shot generation, temp=0.7, max_tokens=512, num_outputs=1, which is reasonable for creative generation. Sample outputs across gemini-2.5-pro and gemma-3-1b include all three cue words and stay within the ~5-sentence cap, and llm_judge_creativity=5 for all inspected predictions (suggesting ceiling effect / weak judge discrimination on this rubric). Output shape matches other GenericLLMJudge tasks in this repo. Paper abstract (arxiv.org/abs/2411.02316) confirms the 5-sentence cue-word task design but the abstract does not expose the full methodology; deeper verification would require the PDF._

**Gaps:**
- Only 4 cue-word triads are exercised (scenarios/cue_word_story_scenario.py:109-130): 'stamp-letter-send', 'petrol-diesel-pump' (low distance) and 'gloom-payment-exist', 'organ-empire-comply' (high distance). The HuggingFace dataset (mismayil/creative_story_generation_dataset) contains 479 pre-generated human+AI stories grouped by these same 4 item_ids, so item coverage matches the paper's pilot set, but the run yields only 4 generations per model (one per triad) — no repetition/sampling for diversity metrics the paper reports.
- Paper (Section 4) rates stories on 4 dimensions (creativity, originality, surprise, effectiveness/value) with 1-5 scales from expert and non-expert raters. run_specs/cue_word_story_run_specs.py:15-23,50-63 implements only a single generic creativity rubric (llm_judge_creativity, 1-5) via GenericLLMJudgeAnnotator — originality/surprise/value dimensions and the paper's specific per-dimension prompt template (metric_notes/cue_word_story_eval_metrics_notes.md:28-39) are not scored.
- Judge model is openai/gpt-4 at temperature 0.0 (run_specs/cue_word_story_run_specs.py:57-58). Paper used human experts + non-experts as the ground truth; GPT-4 LLM-as-judge is a reasonable proxy acknowledged in metric_notes, but no calibration/correlation to human ratings is reported and the rubric text is a generic 1-5 scale, not the paper's dimension-specific anchor language.
- Automated metrics from the paper (n-gram diversity, inverse homogenization, semantic distance of dominant terms, sentence-to-sentence semantic distance, Flesch-Kincaid, POS/dep/constituency complexity) are not implemented — only HELM's basic BLEU-1/BLEU-4/ROUGE-L/F1/exact-match/quasi-exact-match against human reference stories (run_specs/cue_word_story_run_specs.py:49). metric_notes/cue_word_story_eval_metrics_notes.md:47-65 explicitly defers these as future work.
- metric_notes/cue_word_story_annotator_notes.md is empty (0 lines) — annotator configuration is undocumented outside the run_spec itself.
- Reference stories used for BLEU/ROUGE/F1 are filtered to human-authored only (scenarios/cue_word_story_scenario.py:142-146); this is a defensible choice but is not stated in the paper as the intended comparison target — the paper treats human vs. LLM stories symmetrically.

### `dat`
_dat_creative_writing near-duplication (per NOTES_FOR_VIJETA.md:43, §1): CONFIRMED at the 'dat' subset level.
dat_creative_writing_scenario.py includes a 'dat' subset (SUBSETS[0]) whose DAT_PROMPT (lines 45-53, from
Bellemare et al. 2024 Nature Human Behaviour / DAT_GPT repo) is a slightly longer variant of the same task
with the same semantics (10 single-word English nouns, no proper nouns, no specialised vocab) as the
standalone dat scenario's Chen & Ding 2023 prompt. The two run_specs also differ on embedding model
(all-MiniLM-L6-v2 vs all-mpnet-base-v2) and dat_creative_writing adds an LLM-judge creativity rubric.
Recommendation: keep dat_creative_writing for its synopsis/flash_fiction/haiku/dat_strategies subsets
(which are genuinely distinct — Bellemare 2024's strategy-guided DAT and open-ended writing tasks)
but DROP the bare 'dat' subset from dat_creative_writing to avoid counting the same benchmark twice
under different names with different embedding models. Alternatively, make dat_creative_writing re-use
the same GloVe-based DAT scorer on its 'dat' subset for internal consistency.
Paper cross-check: Olson et al. 2021 PNAS (primary methodology source) and Chen & Ding 2023 EMNLP Findings
(LLM-adaptation) both consistently use GloVe 840B 300d with the first-7-valid-unique-words rule; current
implementation deviates from both._

**Gaps:**
- Embedding model drift (primary issue): metric_notes/dat_annotator_notes.md:32-36 and dat_eval_metrics_notes.md both specify GloVe 840B 300d as the required embedding — this is the canonical choice from Olson et al. 2021 PNAS and replicated by Chen & Ding 2023 EMNLP. run_specs/dat_run_specs.py:37 instead uses sentence-transformers all-MiniLM-L6-v2 (384d). SemanticDiversityMetric (metrics/semantic_diversity_metric.py:44-54) is not GloVe-based. Resulting scores are not on the same scale as the published human baseline (mean ~78, SD ~10) or LLM benchmarks (GPT-4 ~90) quoted in the annotator notes.
- Missing Olson-spec validation: metric_notes specifies 'keep first 7 valid unique words' (min threshold) with GloVe-vocabulary OOV-skip before computing pairwise distances. _dat_response_to_words (metrics/creativity_score_metric.py:45-63) does no vocabulary check, no min-7 enforcement, no duplicate dedup beyond list semantics, and no 'return None/NaN' for <7 valid words — it will happily score truncated outputs.
- Segmentation fragility on observed outputs: display_predictions.json shows markdown-bold numbered lists like '1.  **Quantum**\n2.  **Symphony**...'. _dat_response_to_words splits on '. ' after the first '1.', which parses double-star-wrapped items as 'quantum', 'symphony' etc. reasonably, but the leading preamble ('Here are 10 single-word nouns...') is dropped before '1.' which is correct. Fallback path (comma-split, last-token-per-chunk) would mis-handle the same outputs. Worth unit-testing on the exact observed markdown form.
- Prompt phrasing: scenarios/dat_scenario.py:77-83 uses Chen & Ding 2023's condensed prompt, NOT the fuller Olson 2021 prompt that dat_creative_writing uses. Both are published DAT prompts; the run_spec doc comment correctly attributes it. Acceptable for the Chen & Ding LLM-adaptation but diverges from the canonical human-baseline instruction set.
- Temperature/decoding: run_specs/dat_run_specs.py:29-30 sets temperature=0.7, num_outputs=1, max_tokens=512, matching Chen & Ding 2023 §6 (t=0.7, top_p=0.9 — top_p not set explicitly). num_trials=100 (default in scenario) matches paper protocol.
- Observed determinism across trials: display_predictions.json for gemini-2.5-flash-lite shows byte-identical predictions across multiple instance IDs (id2, id16, id26, id54, id55, id73, id75, id86, id93...) despite temperature=0.7. This is almost certainly a caching/deduplication artifact of identical prompts with a cache-warm client, not a faithfulness bug in scenarios/run_specs — but it defeats DAT's repeated-trial design (each trial should be an independent sample). Verify HELM's request cache is either salted per-instance or disabled for this benchmark.
- Human baseline not loaded: scenario docstring (dat_scenario.py:12) references 'Human baseline data from probing_creativity repository (8,572 responses)' and dataset/human.json, but get_instances does not load or surface any human reference — references=[] everywhere. That's fine for the generation-only design, but percentile comparisons against humans are not computable from the current output shape.

### `dat_creative_writing`
_Near-duplication with `dat` (per NOTES_FOR_VIJETA.md:43, §1): CONFIRMED and currently worse than the
abstract concern. Because run_specs/dat_creative_writing_run_specs.py hardcodes args={} on the ScenarioSpec,
only subset='dat' executes — making this scenario, as wired today, a strict redundant sibling of the
standalone dat scenario with (a) a different sentence-embedding model (all-mpnet-base-v2 vs
all-MiniLM-L6-v2), (b) an added LLM-judge annotation whose rubric is miscalibrated for bare DAT output,
and (c) the same identical-predictions caching bug observed in dat.
The scenario file itself has real additive value over dat: the five strategy-DAT prompts and the three
writing prompts (synopsis / flash_fiction / haiku) are drawn verbatim from Bellemare et al. 2024
(Nature Human Behaviour) and the DAT_GPT repo and are not covered anywhere else. But none of that value
reaches the evaluator because the run_spec never instructs the scenario to emit those subsets.
Recommendation: expand into nine run_spec variants (or a single parameterised one), wire in
GloVe-DAT for the six DAT variants and DSI+Lziv for the three writing variants, drop the redundant
'dat' subset (or ensure it uses the exact same scorer as standalone dat), and fix the request-cache
issue before treating this benchmark's numbers as meaningful. As currently configured, this run reports
a sentence-embedding diversity proxy over 100 byte-identical Gemini outputs on one subset out of nine —
it is not the Bellemare 2024 benchmark in any substantive sense._

**Gaps:**
- Scenario is multi-subset (SUBSETS = dat, dat_thesaurus, dat_etymology, dat_opposites, dat_random, dat_young, synopsis, flash_fiction, haiku) but run_specs/dat_creative_writing_run_specs.py:30-33 instantiates ScenarioSpec with args={}, so only the default subset='dat' is ever executed. All five strategy-DAT subsets and all three writing subsets (synopsis/flash_fiction/haiku) — the whole reason this scenario exists in addition to bare `dat` — are dead code in the current run.
- Prompts: DAT_PROMPT (scenarios/dat_creative_writing_scenario.py:45-53) is verbatim from Bellemare et al. 2024 DAT_GPT (scripts/api_call_dat_gpt4.py). The five DAT_STRATEGY_PROMPTS (lines 55-106) are verbatim from the same source. Writing prompts (lines 108-114) match api_call_gpt4_stories.py. Prompts themselves are faithful.
- Metric drift from spec: metric_notes/dat_creative_writing_eval_metrics_notes.md:5-50 mandates GloVe 840B.300d cosine-distance DAT scoring with first-7-valid-unique-words rule for DAT subsets, and DSI (BERT-large-uncased layers 6-7) + Lziv (antropy.lziv_complexity) for the writing subsets. Actual metric wired in run_spec is SemanticDiversityMetric with all-mpnet-base-v2 — a single sentence-embedding diversity proxy that is neither GloVe-DAT nor DSI-BERT nor Lziv. None of the three canonical metrics (DAT, DSI, Lziv) are implemented for this scenario.
- Cross-check vs dat scenario: the bare 'dat' subset here and the standalone dat scenario (reviewed in dat.yaml) both run on the same DAT task, and neither uses the GloVe-based scorer the metric_notes specify. dat uses all-MiniLM-L6-v2 (384d); dat_creative_writing uses all-mpnet-base-v2 (768d). Not even internally consistent with each other, let alone with the paper. Results from the two runs cannot be pooled.
- Judge config: _RUBRIC_LLM_JUDGE_CREATIVITY (run_specs:15-24) is a generic divergent-thinking rubric but its wording ('connects the given words or concepts') presumes a DAT-style input — it is semantically incoherent when applied to bare 10-word DAT outputs (nothing to 'connect') and would be reasonable only for strategy-DAT or writing subsets. Judge is openai/gpt-4, temp 0, max 256 tokens — matches the project-wide judge convention but paper's flash-fiction rating used GPT-4 at default temperature per metric_notes:92-94. No per-subset rubric differentiation for synopsis/flash_fiction/haiku quality.
- Observed output pathology: benchmark_output/runs/trial_10inst/dat_creative_writing_model=google_gemini-2.5-flash-lite/display_predictions.json contains byte-identical predicted_text for every single dat_* instance_id inspected (dat_2, dat_16, dat_26, dat_54, dat_55, dat_73, dat_75, dat_86, dat_93, dat_95 all emit the same 'Sun, Silence, Feather, Justice, Ocean, Dream, Stone, Music, Shadow, Kindness' list). Same caching/determinism artifact flagged in dat.yaml — identical prompt across 100 trials plus a cache-warm client defeats the num_trials=100 diversity-measurement design.
- Judge score floor: all inspected annotations return llm_judge_creativity=1 (the minimum). This is plausible since the rubric asks about 'connecting concepts' and bare DAT outputs have no connective prose — confirms the rubric mismatch above.
- Near-duplication with dat: CONFIRMED. The 'dat' subset of this scenario (SUBSETS[0]) is semantically identical to the standalone dat scenario at the task level; the only genuinely distinct subsets here are the strategy-DAT (×5) and the writing tasks (×3). Since run_spec executes only the default 'dat' subset, the current configuration is a strict near-duplicate of dat with a different embedding model and an added judge. See notes.

### `data_narrative`
_BORDERLINE CREATIVITY - flagged by NOTES_FOR_VIJETA.md as multimodal/design-adjacent and a known always-fail needing investigation. Scenario construction (dataset download, splits, few-shot) is clean and documented, but run_spec drops the paper's entire evaluation methodology (GPT-4 judge on 4 dimensions) in favor of n-gram overlap + BertScore. For a data-to-text faithfulness task this is the wrong metric family - models can hallucinate statistics undetected. Recommend either (a) excluding from creativity suite per borderline flag, or (b) adding the 4-dimension LLM judge before relying on results. WebFetch of arxiv/ACL pages returned only abstracts (full-text PDFs exceeded size limit); cross-check leaned on annotator_notes + scenario docstring which already codify the paper's 4 dimensions._

**Gaps:**
- Protocol adaptation: paper defines a 6-stage multi-agent iterative refinement pipeline (reflection -> outline -> narration, each with critic/revise) per scenarios/data_narrative_scenario.py:16-22 and metric_notes/data_narrative_annotator_notes.md:12-27. HELM scenario collapses this to single-turn generation (scenarios/data_narrative_scenario.py:24-25, 121-133). Documented adaptation but materially changes task.
- Prompt is a simplified paraphrase (scenarios/data_narrative_scenario.py:126-133), not the verbatim Figure 24 narration-generation prompt from the paper. Loses visualization placeholders, outline-conditioning, and header structure that the paper's narration stage prescribes.
- Primary metric drift: paper uses GPT-4 judge across 4 dimensions (factual accuracy, coherence, comprehensiveness, theme consistency) per metric_notes/data_narrative_annotator_notes.md:34-38. run_specs/data_narrative_run_specs.py:34-37 registers ONLY n-gram overlap metrics (exact_match, quasi_exact_match, f1, rouge_l, bleu_1, bleu_4) plus BertScore. No LLM judge, no factual-accuracy metric, annotators=None (line 45).
- Factual accuracy is the paper's central concern (annotator notes mark it 'Critical', line 58); unmeasured here. Free-form narratives will pass BLEU/ROUGE while hallucinating numbers - display_predictions.json shows exactly this risk (model invents specific figures like '6.59 in 1950', 'peak around the 2080s') with no factual verifier.
- Output shape issue: display_predictions.json at benchmark_output/runs/trial_10inst/data_narrative_model=google_gemini-2.5-flash-lite shows heavy prompt-conditioned duplication - 5 of 10 instances for story_3 return byte-identical paragraphs because the prompt varies only by segment index while topic/intent/table repeat. Suggests either dataset has redundant segments or scenario is not differentiating seg_idx in the user-visible prompt.
- temperature=0.7 with num_outputs=1 (run_specs/data_narrative_run_specs.py:30) is reasonable for generation but not matched to any paper setting; max_tokens=512 may truncate multi-paragraph narratives the paper expects.
- few-shot max_train_instances=5 from Tableau train (run_specs/data_narrative_run_specs.py:27); paper pipeline is zero-shot agentic, not few-shot single-turn, so few-shot is an unjustified addition.
- Borderline creativity fit: NOTES_FOR_VIJETA.md lines 37, 58 flags data_narrative as borderline (multimodal/design category) and one of the 7 'always-fails' pending investigation. Task is data-to-text faithfulness, closer to factual grounding than open-ended creativity.

### `dpt`
_Paper cross-check: arXiv:2502.03253 abstract confirms facets 'remoteness', 'uncommonness',
'cleverness' with originality as aggregate; annotator_notes.md:100-101 correctly maps
paper 'remoteness' -> rubric 'uncommonness'. Paper PDF could not be text-extracted via
WebFetch (returned binary stream); notebook scrape from the repo confirmed the 4-facet +
effectiveness structure matches annotator_notes.md, but the run_spec rubric ignores all
of this and uses a generic divergent-thinking Likert, which is the dominant faithfulness
bug here. Scenario dataset construction and prompt adaptation are reasonable; the LLM-judge
wiring is the part that has drifted from both the paper and the scenario's own
annotator_notes.md._

**Gaps:**
- Rubric dimension mismatch (primary issue): metric_notes/dpt_annotator_notes.md:17, 27-58 specifies a 5-dimension rubric (originality, cleverness, uncommonness, effectiveness, conciseness) with explicit per-dimension anchors, matching the paper's human-study dimensions (originality, cleverness, uncommonness, remoteness, plus effectiveness). run_specs/dpt_run_specs.py:15-24 instead uses a generic 1-5 'CREATIVITY' rubric copy-pasted from a divergent-thinking judge (referencing 'range' of 'ideas produced', which is wrong for a single-response generation task). The annotator_notes rubric is never actually wired to the annotator.
- Judge model mismatch: annotator_notes.md:15 recommends GPT-4o (per paper, outperformed humans in consistency). run_specs/dpt_run_specs.py:57 configures 'openai/gpt-4' (legacy GPT-4, not GPT-4o). Paper baselines (annotator_notes.md:88-91) are for gpt-4o-mini and claude-3.5-haiku — neither is the judge actually used.
- Dimension collapse not surfaced: paper's key finding (annotator_notes.md:93-96) is that LLM judges collapse originality/cleverness/uncommonness/remoteness to r≈0.99. Using a single 1-5 'creativity' score in the run_spec guarantees collapse by construction — the benchmark cannot replicate the paper's multi-dimension analysis, nor distinguish originality from effectiveness (which the paper treats as a separately rescaled secondary metric, annotator_notes.md:102-103).
- Prompt wording adapted, not verbatim: scenarios/dpt_scenario.py:87-92 uses an adapted instruction ('Think of a new, creative way... original... practically feasible... 2-4 sentences'). Paper Section 2.1 wording could not be recovered from arXiv abstract or PDF extract; cross-check vs supplemental_materials.pdf in the GitHub repo is recommended before locking this as faithful. The 2-4 sentence length cap is a scenario-level imposition — paper's human study likely had no strict length cap.
- 16 design problems: scenarios/dpt_scenario.py:53-76 lists 16 problems split 5/3/8 across accessibility/transportation/environment. Notebook scrape confirms ~14 distinct problem_ids in the public CSV (possibly filtered) with the paper title referencing 16; problem texts look plausible and domain-consistent but exact verbatim match to paper's study instrument could not be verified from the arXiv page.
- Calibration not performed: annotator_notes.md:60-82 describes 830 human-rated responses in cleaned_data_explanations_gold.csv for judge calibration. No calibration code is present in the run_spec; the judge is used cold against a rubric that doesn't match the human-rater dimensions. Reported scores have no known correlation with the paper's human baseline.
- Decoding params: run_specs/dpt_run_specs.py:44-46 uses temperature=0.7, max_tokens=512, num_outputs=1. Paper's LLM-generation protocol not recovered; 0.7 is a conventional creative-writing default and likely acceptable, but should be confirmed.
- Output shape sane: 16 instances × 1 completion = 16 predictions per model. display_predictions.json for gemini-2.5-flash and gemma-3-12b show well-formed 2-4 sentence solutions appropriately targeted to each design problem; judge scores collapsing to '4' across all 16 instances for gemini-2.5-flash is consistent with the dimension-collapse finding but also indicates very low judge discrimination on this rubric.

### `eqbench_creative_writing_v3`
_Scenario faithfully loads the 32 v3 prompts from the official repo and preserves the zero-shot generation task. Core faithfulness issues are in the evaluation layer: the so-called 'elo_rating' metric is not Elo (no pairwise, no Glicko-2), seed_modifiers are ignored, and the num_outputs=3 x 3-iteration duplication concatenates 3 stories into one scored artifact per instance. Treat outputs as a creative-writing Likert rubric probe, not as a reproduction of the EQBench v3 leaderboard. Paper (arXiv:2509.02534) references EQBench as a downstream eval but does not specify implementation; cross-check leaned on the EQ-bench/creative-writing-bench GitHub README._

**Gaps:**
- Iteration shape drift: scenario creates 3 Instances per prompt (96 total) with num_outputs=1 semantics, but run_specs/eqbench_creative_writing_v3_run_specs.py:54 sets num_outputs=3, meaning each of the 96 instances yields 3 completions (288 completions). Predictions confirm this — display_predictions.json shows single instance_id 'eqbench_cw_v3_9_iter3' with predicted_text containing 3 concatenated stories. The iteration multiplier is applied twice.
- Predicted text concatenation: display_predictions.json 'predicted_text' is a single string containing multiple stories run together with no separator, because num_outputs=3 collapses into one field. This breaks per-iteration scoring — judge sees 3 stories as one artifact, inflating/garbling rubric and elo scores (annotations show one rubric_score per merged blob).
- Seed modifiers unused: scenario loads 'seed_modifiers' (scenarios/eqbench_creative_writing_v3_scenario.py:111) but does not apply them to the prompt (line 116: prompt_text = writing_prompt). Paper/GitHub methodology varies prompts across iterations via seed modifiers to promote diversity; here all 3 iterations receive identical input, reducing variance.
- min_p=0.1 not wired: metric_notes and scenario docstring both specify min_p=0.1 as a required generation param, but AdapterSpec in run_specs/eqbench_creative_writing_v3_run_specs.py:46-58 does not pass min_p (HELM AdapterSpec has no min_p field; it is silently dropped). Paper parity is compromised for samplers that rely on it.
- Metric scheme mismatch: canonical EQBench v3 uses pairwise Glicko-2 Elo (with margin weighting) anchored to reference models, plus rubric aggregation across 10 criteria on a 0–20 scale per criterion (per the GitHub pipeline). run_specs here uses a single-response LLM-as-judge on a 1–5 Likert for both 'elo_rating' and 'rubric_score' metrics (run_specs/eqbench_creative_writing_v3_run_specs.py:15-35, 61-62). The metric named 'elo_rating' is not an Elo rating — it is a Likert quality score. No pairwise comparisons, no Glicko-2, no normalization.
- Judge rubric abbreviated: canonical rubric enumerates ~10 criteria (character authenticity, originality, coherence, emotional engagement, prose quality, plus slop detection, voice, etc.) with detailed anchors. run_specs rubric strings are 5 generic Likert anchors without per-criterion breakdown or slop penalty (annotator_notes.md:109-113 describes slop detection as part of the benchmark, but it is not implemented).
- max_tokens=2048 likely truncates: target length is ~1000 words (~1300-1500 tokens), and with num_outputs=3 sharing the same max_tokens budget per completion, outputs at budget limit are truncated — display_predictions.json shows stories ending mid-sentence ('something old' with no terminator).
- Output shape: 3 models x 96 instances x 3 concatenated completions visible; annotations populated with integer 1–5 scores for both 'elo_rating' and 'rubric_score'. Values in reviewed sample (3, 4, 3) are plausible mid-range Likerts.

### `future_ideas`
_Paper cross-check: https://arxiv.org/abs/2409.06185 (EMNLP 2025). Abstract confirms IAScore + IDI as
automated metrics and novelty/relevance/feasibility as the three human-eval axes. Exact prompt text is
NOT disclosed in the paper or GitHub README (verified via WebFetch on both); scenario's invented prompt
is therefore an unavoidable reconstruction but should be flagged in run reports.
Dataset is loaded correctly from the official RealF xlsx files on GitHub (scenario lines 66-77) with the
right columns (full_text_WF as input, Future_work as gold), skipping data-leakage columns (full_text,
Response_Chat). 458-paper / 5-domain split matches paper's Table 1.
Cross-benchmark note: the paper's own Response_Chat column holds GPT-4/Claude-2/Gemini outputs from the
original experiments — these could be used to calibrate expected IAScore/IDI ranges once the metrics are
implemented (metric_notes/future_ideas_eval_metrics_notes.md:47-50 already flags this)._

**Gaps:**
- Primary metrics missing: metric_notes/future_ideas_eval_metrics_notes.md:8-31 designates IAScore (SentenceBERT cosine similarity of generated ideas vs author Future_work) and IDI (1 - mean pairwise cosine sim across generated ideas) as the paper's authoritative metrics. run_specs/future_ideas_run_specs.py:60-63 instead registers two GenericLLMJudgeMetric instances (llm_judge_novelty, llm_judge_relevance). Neither IAScore nor IDI is computed. BLEU/ROUGE/F1 (mentioned in eval_metrics_notes.md:33-37 as soft proxies) are also not wired into metric_specs.
- Human-eval dimension coverage: paper uses novelty + relevance + FEASIBILITY as the three human-judgment axes (metric_notes/future_ideas_eval_metrics_notes.md:41-42, confirmed by paper abstract on arxiv.org/abs/2409.06185). run_specs only implements novelty and relevance; feasibility is silently omitted.
- Prompt source transparency: scenarios/future_ideas_scenario.py:34-48 acknowledges 'Standard instruction format (paper does not publish exact prompts)' and uses an invented instruction. This is honest, but it means the prompt is NOT the paper's prompt — the benchmark measures performance under a reconstructed prompt. Neither arxiv abstract, PDF extract, nor the GitHub README discloses the canonical prompt text, so this is a known unavoidable deviation.
- Paper truncation: scenarios/future_ideas_scenario.py:82 hard-caps paper_text at 2000 words via leading-truncation (keeps first 2000 words). The paper's setup feeds the ENTIRE paper body (full_text_WF) to the LLM — the GitHub data column name itself signals this design. 2000-word cap will drop later sections (methods/results tail, discussion) for long papers, changing what the model conditions on. Trailing truncation would also drop the discussion that most directly cues future work; leading truncation drops intro/related-work which may weaken grounding. Either way, this is a material deviation from the paper's full-text protocol that should be documented and ideally lifted for long-context models.
- Metric name mismatch: run_specs/future_ideas_run_specs.py:39-41 imports ADAPT_GENERATION, sets temperature=0.7, max_tokens=512, num_outputs=1. Paper does not specify decoding params; 0.7/512 is a reasonable default but max_tokens=512 systematically truncates outputs — display_predictions.json across gemini-2.5-flash and gemini-2.5-pro shows every prediction truncated mid-sentence (e.g., 'Investigating the degradation of I' — id48, flash; 'caspase-3, -' — id48, pro). Ideas-per-response is thereby capped by format rather than by model reasoning. Raise max_tokens (paper uses full generation) or accept that IAScore/IDI-style idea counts are confounded.
- Judge config: run_specs pins judge_model_name='openai/gpt-4' (not gpt-4o/gpt-4-turbo), temperature=0.0, max_new_tokens=256. The paper itself does not use an LLM-judge rubric for these dimensions — human annotators score 30 sampled outputs/model/domain on 1-5 scales. So the LLM-judge path in run_specs is a HELM-local substitution, not a paper-faithful reimplementation. The 1-5 rubrics (run_specs lines 15-35) are LLM-adaptations of the paper's human-eval dimensions; rubric wording is reasonable but unsourced from the paper.
- Idea segmentation absent: metric_notes/future_ideas_eval_metrics_notes.md:44-46 explicitly notes 'metric implementation should segment generated output into individual idea sentences before computing IAScore/IDI'. No such segmentation exists in the current pipeline. The LLM-judge sees the whole blob and returns one novelty+one relevance score, so per-idea granularity (central to both IAScore and IDI) is lost.
- Output shape observed (flash, pro, both share): long markdown-structured bulleted lists grouped under Roman-numeral categories, 5-10 idea families each. Reasonable content, but (a) truncated by max_tokens in every sample inspected, (b) not parseable into discrete idea units without a segmenter.

### `fuxibench`
_Only ci_gen is actually exercised in the trial run (display_predictions.json filenames all fuxi_CiG_*), and it is scored by a generic 1–5 quality rubric rather than pacc. The scenario code itself is faithful (exact instruction+input prompt, correct fields, valid GitHub raw data source, TEST split); faithfulness failures are concentrated in run_specs/fuxibench_run_specs.py (single spec, wrong metric, wrong judge). Paper PDF and GitHub README did not yield exact decoding params via WebFetch; cross-check relied on scenario docstring, metric_notes, and annotator_notes which cite paper Section 4.2 and evaluate.py._

**Gaps:**
- Subset collapse: scenarios/fuxibench_scenario.py declares 5 creative subtasks (ci_gen, couplet_gen, poem_gen, poem_nmt_inv, poem_appre) but run_specs/fuxibench_run_specs.py registers a single run spec with args={} (defaults to ci_gen only). Other four subsets never instantiated; the display_predictions.json output files all show fuxi_CiG_* ids confirming only ci_gen is executed.
- Metric drift (ci_gen, all subsets): paper/notes specify subset-specific metrics — pacc (rule-based cipai template match via FormatEvaluator/cipai2info.json) for ci_gen, cacc for couplet_gen, lacc for poem_gen/poem_nmt_inv, BLEU for poem_appre (metric_notes/fuxibench_eval_metrics_notes.md). Run spec instead applies a single generic 1–5 Likert LLM judge (_RUBRIC_LLM_JUDGE_QUALITY) to whatever subset runs. None of pacc/cacc/lacc/BLEU is implemented.
- Judge model mismatch: annotator_notes specify fine-tuned Qwen2-7B-Instruct (Y/N binary, 89.8% acc, kappa=0.764) as the validated judge for lacc. run_specs uses openai/gpt-4o with a generic 1–5 rubric — different model, different output space, different rubric content (no 包含要点/与标准答案一致/符合事实 criteria).
- Temperature: adapter sets temperature=0.7 for generation; paper/repo evaluate zero-shot with deterministic decoding (no explicit temperature documented but standard FuxiBench eval is zero-shot CG with default HF generate). 0.7 adds sampling noise not in the reference protocol.
- Few-shot: repo exposes both zero-shot and 5-shot CG variants; scenario/run_spec pick zero-shot (max_train_instances=0). Acceptable but the 5-shot variant is not offered.
- Output shape: display_predictions.json shows Chinese ci poetry outputs with llm_judge_quality integers in 2–4 range — consistent with the generic judge that was actually wired up, but not with paper's binary pacc pass/fail.

### `gauss`
_BORDERLINE inclusion flagged in NOTES_FOR_VIJETA.md §1 — only 3/41 problems are creativity-tagged; scenario default correctly filters to those 3, but n=3 is too small for a standalone creativity score and each of the 3 is a different sub-task (open exploration / multi-solution IMO / digit-move puzzle), so scores are not comparable across problems. Scenario loading and prompt construction are faithful to the dataset; the evaluation layer (run_spec) is where faithfulness breaks: a generic 0-7 Likert rubric replaces the paper's per-problem variable-point rubrics with total_score caps, and max_tokens=512 truncates 12a/12b in every inspected run (gemini_pro, gemma_3_27b). Observed rubric scores (e.g. 4 on a problem with total_score=3) exceed the problem's maximum, confirming the judge is not honoring the per-problem bounded scale. WebFetch of the arXiv abstract confirmed 12 dimensions grouped into 3 domains (knowledge/understanding, problem solving/communication, meta-skills/creativity) but the 120-page paper was not fully fetched — scenario docstring and metric_notes carry the protocol detail used for cross-check._

**Gaps:**
- Contamination-borderline inclusion (NOTES_FOR_VIJETA.md section 1): GAUSS has 41 graduate-level math problems across 12 skill dimensions; only 3 (categories 12a/12b/12c) are creativity-tagged. The scenario default dimension='12' (scenarios/gauss_scenario.py:88) filters to just those 3, which is the right call for a creativity suite, but leaves n=3 — too small to be statistically meaningful as a standalone creativity benchmark.
- Scenario faithful: dataset loaded from GaussMath/GAUSS (HF), problem_statement passed as prompt with no system preamble, standard_solution stored as CORRECT_TAG reference, rubric + total_score preserved in extra_data (scenarios/gauss_scenario.py:107-156). Matches dataset schema described in paper.
- max_tokens=512 (run_specs/gauss_run_specs.py:44) is far too low for graduate math / SLE-style open exploration. Inspection of benchmark_output/runs/gemini_pro/gauss_model=google_gemini-2.5-pro/display_predictions.json shows 12a and 12b truncated mid-sentence (~500 tokens, visibly cut off before final answer / second solution). Same truncation visible in gemma_3_27b run for 12b. Biases rubric_score downward, especially for 12b (rubric requires 'up to 2 points' for multiple solutions — truncation prevents reaching the second solution).
- Judge rubric shape mismatch: paper/metric_notes prescribe per-problem variable-point rubrics tied to total_score (1-3 pts; 12a=3, 12b=2, 12c=1) with criterion-specific award rules (e.g. 12c requires the exact equation 2^6 - 63 = 1). run_specs/gauss_run_specs.py:15-24 uses a single generic 0-7+ Likert rubric ignoring total_score and the per-problem rubric text from extra_data. The annotator (GenericLLMJudgeAnnotator) is not configured to thread problem_statement, standard_solution, rubric, or total_score into the judge prompt per metric_notes/gauss_eval_metrics_notes.md.
- Observed rubric_scores in display_predictions.json go up to 4 (gemini_pro 12a) and 3 (gemma_3_27b 12b), exceeding the per-problem total_score (3 and 2 respectively) — confirms the judge is scoring on the generic 0-7 scale rather than the paper's bounded per-problem scale.
- Judge model hardcoded to openai/gpt-4 (run_specs/gauss_run_specs.py:58); paper uses human expert graders following rubrics (metric_notes/gauss_eval_metrics_notes.md §'Human Expert Evaluation'). LLM-as-judge is a documented adaptation, not the paper method.
- BLEU/ROUGE/F1 metrics (run_specs:50) are uninformative for proof-style free-form math solutions and are not used by the paper; harmless but noisy.
- metric_notes/gauss_annotator_notes.md is empty (0 lines) — all protocol detail lives in gauss_eval_metrics_notes.md.
- temperature=0.7 is unusual for math proof generation; paper does not specify a canonical decoding config for third-party evaluation but 0.0 would be more standard for reproducibility.

### `geo_story`
_Scenario correctly downloads GeoNames cities1000 and countryInfo.txt, joins on country_code, and deterministically samples 25 cities per country with seed=100. Zero-shot, num_outputs=1, temp=0.7 adapter is sensible for creative generation. Inspected display_predictions.json across gemini-2.5-flash, gemma-3-1b, gemma-3-27b show city+country correctly interpolated into prompts, stories ~500 tokens, llm_judge_creativity scores span 3-4 with reasonable variance. Output shape matches sibling GenericLLMJudge tasks in this repo. Paper URL (arxiv.org/abs/2411.07320) PDF was not parseable via WebFetch; verification was cross-checked against the paper's public repo github.com/FLAIR-IISc/richer-countries-have-richer-output (prompt_template.json, measure_uniqueness.py, measure_informativeness.py, extract_emotions.py fetched directly). The faithfulness gap is primarily that HELM implements a generic creativity judge instead of the paper's three domain-specific analyses._

**Gaps:**
- Paper metrics (lexical uniqueness via corpus-wide IDF from src/measure_uniqueness.py; informativeness via spaCy en_core_web_trf NER count of LOC/FAC/GPE entities from src/measure_informativeness.py; emotion classification via GPT-4 across 5 categories Joy/Hardships/Fear/Sadness/Serenity from src/extract_emotions.py) are NOT implemented. run_specs/geo_story_run_specs.py:49-52 only runs BLEU/ROUGE/F1/exact_match (meaningless here — no references; scenario sets output='' at geo_story_scenario.py:232) plus a single generic LLM-judge creativity rubric. metric_notes/geo_story_eval_metrics_notes.md:55-62 explicitly acknowledges this gap.
- Prompt template verbatim deviations vs. src/prompt_template.json: scenario de-duplicates 'Write a story of a a couple from {}' to 'Write a story of a couple from {}' (geo_story_scenario.py:64), strips trailing space from 'Write a story of a social worker from {} ' (line 68), and adds a trailing period to the travel template 'Tell me some important sites to incorporate into my travel plans to {}' (line 96). Minor — likely typo fixes in the paper's JSON — but not literally identical.
- Judge rubric is a generic 1-5 creativity Likert (run_specs/geo_story_run_specs.py:15-24). The paper does NOT use LLM-as-judge for creativity — its three analyses are IDF uniqueness, NER count, and GPT-4 emotion tagging. The rubric added here is orthogonal to the paper's framework and cannot reproduce the paper's GDP-per-capita correlations (metric_notes/geo_story_eval_metrics_notes.md:50-52).
- Judge model pinned to openai/gpt-4 at temp=0.0 (run_specs/geo_story_run_specs.py:58-59). Paper uses GPT-4 only for emotion classification (temp=0.7, max_tokens=50 per extract_emotions.py); the repo's gpt-4 choice aligns with the paper's judge model but is applied to a different task than the paper specifies.
- Scenario samples 25 locations per country with seed=100 (geo_story_scenario.py:121,210-213), matching the paper's ~4000 locations × 245 countries scale; sampling is deterministic and reproducible. However, each location is assigned a SINGLE randomly-chosen prompt (line 227) rather than all 21 story prompts ×  all locations, so the ~200K story generations scale reported in the paper abstract is not reproduced — only ~4K generations per subset.
- Default subset='story' pools all 21 story templates together (geo_story_scenario.py:141-144); the paper reports metrics broken down by prompt type (childhood/character/profession/daily_life). The per-subset variants exist (SUBSETS line 116-119) but are not invoked by the default @run_spec_function.
- Adapter uses max_tokens=512, temperature=0.7 (run_specs/geo_story_run_specs.py:44-45). Paper/code do not state an exact max_tokens budget; inspected outputs (e.g. benchmark_output/runs/gemini_flash/.../display_predictions.json) are truncated mid-sentence at ~512 tokens, which could bias uniqueness/informativeness metrics if ever computed downstream.

### `grapheval_iclr`
_Paper cross-check: https://arxiv.org/abs/2503.12600 (GraphEval; Feng, Sun, You, 2025).
Verified via arxiv HTML v2 experimental-setup section and GitHub README:
  - 50-paper ICLR test set (matches scenario)
  - temperature=0.1 for LLM calls (scenario uses 0.7 — MISMATCH)
  - metrics = accuracy + macro-P/R/F1 (scenario uses LLM-judge Likert — MISMATCH)
  - basic_prompt.txt verbatim reproduced in scenario SYSTEM_PROMPT (MATCH)
Sister-scenario concern: this is the third of three GraphEval-family scenarios that independently
picked temperature=0.7 and an LLM-judge-quality metric in place of the paper's F1/accuracy. The
pattern suggests a shared template was copy-pasted across grapheval_ai_researcher /
grapheval_iclr / grapheval_review_advisor without per-scenario metric adaptation. A single fix
to the classification-metric path in this file can be ported to the other two.
Dataset fields used (paper_id, title, abstract, decision) and skipped (keywords, ratings, year)
are correct for the basic-prompt baseline. Labels observed in test JSONL match paper's 4-way
decision taxonomy._

**Gaps:**
- Primary metrics missing: paper evaluates the ICLR Papers task with classification metrics — accuracy, macro precision, macro recall, and macro F1 over the four decision classes (Reject / Accept Poster / Accept Oral / Accept Spotlight), confirmed via arxiv.org/html/2503.12600v2 experimental setup and GitHub README ('accuracy, precision, recall, and F1 Score'). run_specs/grapheval_iclr_run_specs.py:49-51 registers only GenericLLMJudgeMetric (llm_judge_quality, 1-5 Likert over 'review quality'). No accuracy/F1 against the gold 'decision' label is computed despite scenarios/grapheval_iclr_scenario.py:110 attaching the correct decision as a CORRECT_TAG reference.
- Decoding temperature mismatch: paper's hyperparameter config uses temperature=0.1 ('conservative, deterministic') for the LLM baseline call. run_specs line 46 hard-codes temperature=0.7, which substantially increases variance on a 4-class decision task and will degrade accuracy vs paper-reported numbers. This is the same triple-variant concern noted for the sister grapheval_* scenarios: each run_spec picked 0.7 independently without consulting the shared GraphEval hyperparameter table.
- Judge rubric irrelevant to task: _RUBRIC_LLM_JUDGE_QUALITY (run_specs lines 15-24) scores 'accuracy of evaluation, depth of technical understanding, quality of review reasoning' on 1-5. But the model only emits 'Overall Score= N\n<Decision>' per the scenario prompt (lines 79-86) — there is no review prose to judge for 'depth' or 'reasoning'. Every sampled prediction across gemini-2.5-pro, gemini-2.5-flash, gemma-3-27b display_predictions.json is exactly two lines: score line + decision line. The judge is scoring an artifact that does not contain the content its rubric evaluates, which explains the degenerate 1-4 score spread observed in annotations.
- Judge model legacy: run_specs pins judge_model_name='openai/gpt-4' (deprecated legacy 8k). Same issue as future_ideas / grapheval_ai_researcher — upgrade to gpt-4o or gpt-4-turbo.
- Output shape observed: clean and compliant. predicted_text across all inspected runs is of the form 'Overall Score (0-100)= NN\n<one of 4 decisions>'. This IS the paper's required format (confirmed via basic_prompt.txt content). Predictions ARE parseable into (score, decision); scenario simply never parses them. gemma-3-27b shows heavy mode-collapse onto 'Accept (Poster)' (matches sister-benchmark pattern); gemini-2.5-pro shows wider spread across all four classes. This heterogeneity would be visible in macro-F1 but is invisible under the current llm_judge_quality metric.
- Data split / prompt: scenario loads the official ICLR_test_set.jsonl (50 papers) from the ulab-uiuc/GraphEval repo (scenario lines 38, 90), matches paper's 50-paper test split. SYSTEM_PROMPT (scenario lines 41-86) is a verbatim copy of Baselines/Prompt/basic_prompt.txt — confirmed by WebFetch. max_train_instances=0 is correct (paper uses the same zero-shot prompt with 4 in-prompt examples; no external few-shot demos). This part is faithful.
- Minor: max_tokens=512 is more than adequate — observed outputs are <20 tokens. No truncation risk here, unlike future_ideas.

### `graphrag_bench`
_Scenario correctly identifies the right paper (arXiv:2506.05690), correctly filters the Creative Generation subset, and correctly documents its departures. The critical faithfulness gap is on the metric side: only surface n-gram metrics are registered, while the paper's LLM-as-judge answer_correctness/coverage_score are described in annotator notes but not implemented (annotators=None). Safe to run as a creative-writing capability probe, but headline scores will not reproduce paper results. eval_metrics_notes.md is an empty file._

**Gaps:**
- Retrieval context omitted: paper evaluates GraphRAG pipelines where retrieved graph/text context accompanies the question; scenarios/graphrag_bench_scenario.py:18-21,35-37 deliberately drops evidence/evidence_triple and runs zero-shot from question alone. This is explicitly documented but changes the task from graph-grounded generation to open-ended generative capability probe.
- Primary metric mismatch: paper reports answer_correctness + coverage_score via RAGAS LLM-as-judge (metric_notes/graphrag_bench_annotator_notes.md:10-18, 25-29). run_specs/graphrag_bench_run_specs.py:34-36 registers only BasicGenerationMetric (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4). None of these are the paper's metrics; open-ended creative outputs will score near-zero on exact_match/quasi_exact_match. annotators=None (line 44) — no LLM judge wired.
- Judge model absent: paper uses gpt-4-turbo + bge-large-en-v1.5 embeddings; nothing configured in run_spec.
- Dataset counts verified: scenario filters question_type=='Creative Generation' from raw JSON; annotator notes list novel=67, medical=166, total=233. HF dataset card lists ~2k rows per domain across 4 question types, consistent with ~233 Creative Generation items after filtering. Default domain='novel' (67 items) loads unless args override; run_spec passes args={} so only novel subset runs.
- Prompt: question used directly with empty instructions (input_suffix='\n'); paper provides no explicit template so this is a reasonable default, but temperature=0.7 + max_tokens=512 will truncate long creative outputs (display_predictions show mid-sentence cutoffs at ~512 tokens across id27, id56, id35, id48, id42, id46).
- Output shape: display_predictions.json across gemini_pro, gemma_3_27b, and others show coherent long-form creative generations (diary entries, press releases, letters) matching the Creative Generation task intent; outputs truncated by max_tokens=512.

### `historical_analogy`
_Scenario correctly pulls the 20-item popular_analogy.jsonl and uses the right 3 fields. Key faithfulness issues are: (1) rubric scale changed from 1-4 to 1-5 with different anchors, (2) Pass@1/Wikipedia metric and Jaccard metric are not implemented, (3) the one-shot prompt plus no stop_sequences is causing runaway outputs that auto-score 1 and pollute the metric, and (4) judge rubric measures a different construct than the paper. Full-paper PDF fetch returned binary; cross-check relied on ACL Anthology abstract, GitHub repo landing page, and the local metric_notes which paraphrase evaluation.py. Safe as an exploratory analogy-quality probe; not a faithful reproduction of the paper's scores._

**Gaps:**
- Prompt adaptation: scenarios/historical_analogy_scenario.py:88-101 uses a one-shot template with COVID-19 -> Spanish flu as the exemplar. The paper/direct_generation.py repo page could not be fetched to verify exact wording (GitHub blob 404 on raw and HTML views), so one-shot vs. zero-shot fidelity is unverified. Annotator notes describe a zero-shot 'You are a historical analogy bot...' framing; the embedded one-shot example may leak Spanish flu (which is the id0 ground-truth-adjacent COVID-19 target) to the model for that specific instance.
- Metric scale drift: run_specs/historical_analogy_run_specs.py:15-24 defines a 1-5 Likert rubric (judge_score_analogy), but the paper's abstract-similarity judge uses a 1-4 scale with specific anchor semantics (same-topic/same-situation distinctions) documented in metric_notes/historical_analogy_annotator_notes.md:36-46. Observed judge_score_analogy values include 5 (benchmark_output/runs/gemini_pro/.../display_predictions.json:10), confirming the 1-5 scale is in use and not reproducing the paper's 1-4 scoring.
- Rubric content differs: paper's judge prompt emphasizes abstract-level similarity (not surface), warns against self-analogy, and scores the 'Process' component of structured Wikipedia-derived summaries (metric_notes/historical_analogy_annotator_notes.md:26-76, 83-90). run_specs rubric asks for 'historical accuracy, appropriateness, insightfulness, persuasiveness' of the analogy in general, a different construct than paper's abstract-process similarity.
- Missing Pass@1 metric: paper's headline metric is Pass@1 via Wikipedia variant matching (metric_notes/historical_analogy_annotator_notes.md:14-18); run_specs/historical_analogy_run_specs.py:49-51 only registers the LLM-judge score. No Jaccard keyword similarity is computed either.
- Missing pre-processing pipeline: paper scores on Wikipedia-summary 4-part features (Summary/Background/Process/Result) extracted by GPT-4 before judging (metric_notes:83-90). The scenario feeds only the raw predicted_text (often the event name alone or runaway generation) and the target_event name as Reference; no feature extraction step.
- Judge model: run_specs uses openai/gpt-4 with temperature=0.0 matching the paper's GPT-4 at ~0 temperature; judge_max_new_tokens=256 is adequate for a single integer score.
- Output shape issue: display_predictions show frequent runaway generations where the model echoes the next case's Input Event verbatim instead of emitting an analogy (e.g. id2/id3 in both gemini_pro and gemma_3_27b runs: predicted_text is the Russian Revolution or Russian interference intro text, scored 1). No stop_sequences are set (run_specs:46), and max_tokens=512 with temperature=0.7 lets the one-shot format drift. The judge then correctly scores these non-answers as 1, but the benchmark is measuring prompt-format failure rather than analogy quality for those instances.
- Dataset size/fields match paper: 20 popular analogies, fields event_name/event_intro/target_event (scenarios:67-78); annotator notes claim an event_type field that does not exist in the repo JSONL (benign documentation drift).

### `idrbench`
_Scenario code structure (prompts, subject list, IDR definition, standards) lines up with the arXiv:2507.15736 task design and the display_predictions outputs show models complying with the template. However, the run_spec wires wrong metrics to free-text outputs, only exercises 1 of 5 splits, has a broken I2R label heuristic, and ships annotator notes that belong to a different paper with the same name. Not safe to run as a faithful IDRBench reproduction until metrics parsing and I2R labels are fixed; as-is, IPI results would be near-zero exact_match noise._

**Gaps:**
- Paper identity mismatch: scenarios/idrbench_scenario.py:4 cites arXiv:2507.15736 (Shen et al., 'Understanding LLMs' Ability on Interdisciplinary Research', three-task classification benchmark). metric_notes/idrbench_annotator_notes.md:3 instead cites arXiv:2601.06676 (Feng et al., 'IDRBench: Interactive Deep Research Benchmark', a long-form report-generation benchmark). The annotator notes describe a completely different benchmark (15 questions, Accuracy/Completeness/Coherence/Citations 1-5 Likert, GPT-4 judge) that is irrelevant to the implemented IPI/I3/I2R classification scenario.
- Default run spec (run_specs/idrbench_run_specs.py:17) invokes IDRBenchScenario() with no args, so only task='IPI', level='level_1' runs. The I3 (level_1, level_2) and I2R (level_1, level_2) tasks defined in scenarios/idrbench_scenario.py:122-127 are never executed — 4 of 5 dataset splits are silently skipped.
- Metrics drift: run_specs/idrbench_run_specs.py:35 uses BasicGenerationMetric with exact_match/quasi_exact_match/f1_score/rouge_l/bleu_1/bleu_4. Paper's IPI task is a Yes/No classification (accuracy / macro F1), plus a multi-label subject tag and confidence. Because the scenario emits a templated 3-line string ('Your verdict: Yes\nConfidence score: 95\nSubject: [...]') and the Reference is bare 'Yes'/'No' (scenario line 354-356), exact_match will be ~0 across the board; ROUGE-L/BLEU are meaningless for a classification label. No verdict-extraction parser is wired up.
- I2R gold-label logic is explicitly flagged as a placeholder: scenarios/idrbench_scenario.py:443-450 always marks 'Paper 1' as correct when any target_paper id exists, with the comment 'Simplified matching logic' and 'would need refinement'. This yields a systematically wrong label distribution; any I2R accuracy is uninterpretable.
- No LLM judge is configured (annotators=None in run_spec), yet metric_notes/idrbench_annotator_notes.md documents a 4-dimension GPT-4 jury. The notes and the run spec are disconnected; even setting aside that the notes describe the wrong benchmark, no annotator is wired in.
- Adapter temperature=0.7 (run_specs/idrbench_run_specs.py:30) for a deterministic Yes/No classification task introduces unnecessary sampling noise; paper uses standard decoding for classification.
- Output shape sanity: display_predictions.json (gemini_flash, gemma_3_27b) show well-formed outputs strictly following the 'Your verdict: / Confidence score: / Subject: [...]' template, so prompt formatting is being respected by models. This is the strongest signal the IPI prompt is faithful.
- Could not fully verify verbatim prompt wording against Appendix A.2.3 (arxiv PDF fetch failed); docstring claims 'exact from paper' but template lines around output format ('Use the template... with no markdown...') in scenario lines 250-256 are scenario-authored scaffolding that may deviate from the paper.

### `liveideabench`
_Input pipeline (keyword source, prompt wording, zero-shot framing, 100-word constraint) is faithful. Evaluation pipeline is substantially re-specified: a generic 1-5 creativity judge replaces the paper's five-dimension 1-10 jury plus pairwise fluency plus percentile-based flexibility. Current run therefore measures 'a gpt-4 creativity rating of LiveIdeaBench prompts' rather than LiveIdeaBench scores; headline numbers will not be leaderboard-comparable. Paper full text beyond the abstract was not retrievable via WebFetch; cross-check leaned on the annotator notes, utils/prompts.json excerpt, and abstract._

**Gaps:**
- Idea-generation prompt in scenarios/liveideabench_scenario.py:39-48 matches the exact idea_prompt.description from utils/prompts.json (reward framing, keyword slot, 100-word cap, originality/feasibility/clarity hint). Input assembly uses input_suffix='\n' (run_specs/liveideabench_run_specs.py:40) which appends a trailing newline; harmless.
- Keyword set loaded directly from the official keywordsEverywhere20241216.xlsx (scenarios/liveideabench_scenario.py:36, 56-67). Column-B extraction with header skip is correct; expected ~1,180 keywords across 22 domains (matches paper).
- Metric drift (major): paper evaluates five dimensions (originality, feasibility, fluency, flexibility, clarity) on a 1-10 scale with a dedicated critic prompt plus pairwise-fluency and 30th-percentile-flexibility post-hoc computation (metric_notes/liveideabench_annotator_notes.md:10-66). run_specs/liveideabench_run_specs.py:49-64 instead wires a single generic llm_judge_creativity score on a 1-5 scale with a custom rubric that collapses originality+feasibility+novelty into one ordinal. None of the paper's five dimensions are produced.
- Judge-model drift: paper uses a dynamic top-10 LiveBench panel (claude-3.5-sonnet, gpt-4o, qwen-2.5-72b, deepseek-chat, gemini-2.0-flash-thinking as the disclosed CRITIC_MODELS per metric_notes/liveideabench_annotator_notes.md:7). run_specs uses a single openai/gpt-4 judge at temperature 0.0 (run_specs/liveideabench_run_specs.py:57-58). Single-judge + gpt-4 (non-thinking, non-panel) substantially changes scoring dynamics the paper designed against bias.
- Judge rubric text (run_specs/liveideabench_run_specs.py:15-24) does not match the paper's Nature/Science reviewer persona critic prompt with JSON score block (metric_notes/liveideabench_annotator_notes.md:14-32). No JSON parsing for dimension scores; no pairwise-fluency comparator prompt wired up; flexibility percentile not computed.
- Adapter: max_tokens=512 (run_specs/liveideabench_run_specs.py:44) is ample for the <=100-word outputs; temperature=0.7 is a reasonable creative default but the paper does not pin a single temperature across models. No stop_sequences — acceptable.
- Output shape: display_predictions.json for gemini-2.5-pro shows on-topic, ~100-word Background+Idea responses; annotation key 'generic_llm_judge_llm_judge_creativity.llm_judge_creativity' yields integer 1-5 scores (mostly 4s in sample). Consistent with the collapsed single-score metric, not with the paper's 5-dim 1-10 scores.
- metric_notes/liveideabench_eval_metrics_notes.md is empty (0 bytes); the paper's scoring formulas (flexibility = 30th percentile of other dims, fluency A/B/C/D -> 10/7/4/1 mapping) are only documented in annotator_notes.md:62-66.

### `llm_discussion`
_Scenario faithfully pulls prompts verbatim from the lawraa/LLM-Discussion GitHub JSON files for all four tests, which is the right data source. The faithfulness problem is on the run-spec/metric side: only AUT is selected by default and no LLM judge is attached, so the first-trial run measures token-level Self-BLEU/Distinct-n on AUT-only completions — not the paper's four creativity dimensions across four tests. Predictions are well-formed numbered lists but consistently truncate at 512 tokens. arXiv HTML had no methodology detail; PDF fetch returned binary and could not be parsed, so judge-model choice (GPT-4 vs GPT-3.5) and temperature were not cross-checked against the paper directly — relying on annotator notes as the spec of record._

**Gaps:**
- scenarios/llm_discussion_scenario.py:52 defaults test='aut' and run_specs/llm_discussion_run_specs.py:15-18 passes args={}, so only AUT (30 items) is evaluated by default — the paper runs all four tests (AUT, Similarities, Instances, Scientific; 120 items total). The scenario supports 'all' but the run_spec never selects it.
- No LLM-as-judge annotator is wired: run_specs/llm_discussion_run_specs.py:34-38 registers only SelfBleuMetric and DistinctNMetric (n=1, n=2), and annotators=None (line 46). The paper's four canonical dimensions (Fluency, Flexibility, Originality, Elaboration) scored by GPT-4/GPT-3.5 are not computed — the metric_notes/llm_discussion_annotator_notes.md:17-19 spec is documented but unimplemented.
- metric_notes/llm_discussion_eval_metrics_notes.md exists but is empty (0 content lines) — eval-side metric documentation is missing.
- Instances are created with references=[] (scenarios/llm_discussion_scenario.py:98, 117, 136, 158), which is correct for open-ended generation, but combined with the missing judge this means there is no ground truth and no rubric-based scoring in the first-trial run.
- AdapterSpec uses max_tokens=512, temperature=0.7, num_outputs=1 (run_specs/llm_discussion_run_specs.py:29-31). Paper-reported generation settings are not verifiable (PDF fetch blocked, GitHub overview uninformative), but 512 tokens can truncate long AUT completions — display_predictions.json from trial_10inst shows AUT outputs cut mid-sentence around items 13-17 in every sampled instance (e.g. id13 rope list ends 'create unique and', id11 calculator ends 'The Musical & Rhythmic'). Fluency counts (if ever judged) will be biased low.
- benchmark_output/runs/trial_10inst/llm_discussion_model=google_gemini-2.5-flash-lite/display_predictions.json contains 10 instances, all AUT objects (umbrella/key/scissors/etc.). Confirms default AUT-only scope and confirms only n=10 of 30 items were run in the trial (max_eval_instances clipping).
- Self-BLEU and Distinct-n are computed across tokens within a single model response rather than across the set of generated ideas per prompt — appropriate as a diversity proxy but not what the paper's Flexibility dimension measures (distinct conceptual categories judged by LLM).

### `mixassist`
_Scenario is internally coherent and honestly documents its assumptions (no paper, invented prompt, text-only despite audio source). Primary concern is measurement validity rather than scenario incorrectness: surface-overlap metrics against a single human reference are a weak signal for open-ended mixing-session dialogue, and the adapter settings (temperature 0.7, empty stop_sequences, max_tokens 512) combine to produce noisy, often truncated outputs. Safe to ship as an exploratory creativity signal; results should not be interpreted as a clean MixAssist benchmark because none exists in the literature._

**Gaps:**
- No paper exists for MixAssist (dataset-only release by Michael Clemens on HuggingFace mclemcrew/MixAssist). Scenario docstring correctly acknowledges this (scenarios/mixassist_scenario.py:3-4). Faithfulness is judged against the dataset README rather than a paper spec, so 'prompt_matches' is inherently an ASSUMPTION — the prompt template (scenarios/mixassist_scenario.py:93-108) was invented by the scenario author.
- Empty metric notes: metric_notes/mixassist_annotator_notes.md and metric_notes/mixassist_eval_metrics_notes.md are both 0 bytes. No recorded rationale for the chosen metric set or for declining to use an LLM judge. Cross-check relied solely on scenario/run_spec code and the HF dataset card.
- Metric set over-broad: run_specs/mixassist_run_specs.py:35 requests exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4 against a single reference human response. Scenario docstring line 45 declares 'Evaluation: open_ended (ROUGE-L, BLEU)'. For open-ended multi-turn dialogue continuation where there is no single correct answer (responses like 'Yeah, yeah, yeah, right there...' vs. 'Bring it down. And now bring in the other one.'), exact_match and quasi_exact_match will be ~0 and uninformative. ROUGE/BLEU against one reference are also known to be weak for dialogue; a semantic-similarity or LLM-judge metric would be more defensible but is not wired.
- Prompt deviation from typical MixAssist usage: the dataset includes audio_file references that are dropped (scenarios/mixassist_scenario.py:36 acknowledges this). The real mixing sessions were audio-grounded — text-only replay loses the acoustic context the human assistant actually had. References like 'Based on the audio segment, the overall mix has...' in gemini_pro display_predictions.json (id107, id112) show models hallucinating specific frequency observations they have no way to verify. This is a fundamental limitation of the text-only framing, not a scenario bug, but should be noted.
- Adapter temperature=0.7 with num_outputs=1 (run_specs/mixassist_run_specs.py:30): reasonable for open-ended generation but introduces run-to-run variance that BLEU/ROUGE against a single reference will amplify. No seed pinning visible.
- max_tokens=512 causes visible truncation mid-sentence on verbose models (gemini_pro id107 ends '...starting around '; gemma_3_1b id37 ends 'I need more'). For reference targets that are typically short conversational turns (1–3 sentences) 512 is overkill; models with verbose priors produce over-long outputs that then truncate, artificially depressing surface-overlap metrics.
- Stop sequences empty (run_specs/mixassist_run_specs.py:31) while the prompt uses 'User:'/'Assistant:' role tags: some models (visible in gemma_3_1b id7) continue the dialogue past the requested turn and emit follow-up 'User:'/'Assistant:' lines, inflating length and degrading overlap metrics. Should set stop_sequences=['\nUser:', '\n\nUser:'].
- has_content filter is sensible and matches scenario docstring (156 test instances retained from 250). Verified in scenario code lines 75-77.
- max_train_instances=0 (zero-shot) is consistent with docstring but means the model never sees the conversational style of the dataset; baseline models answer in meta-analysis mode (gemma_3_1b id26/id37/id24 produce 'Okay, here is a breakdown of the conversation...' rather than in-character assistant turns). A few-shot exemplar or stronger system prompt would better elicit the target register. Current zero-shot framing is faithful to the 'no paper prompt' decision but systematically disadvantages smaller models.

### `music_theory_bench`
_Dataset (m-a-p/MusicTheoryBench, 269 knowledge + 98 reasoning test, 5 dev), prompt text, and zero-shot protocol match the paper/config. Core faithfulness failure is the generation+parsing shape: temperature=0.7 plus stop_sequences=['\n'] plus single-letter references means models producing CoT answers (as seen in both display_predictions.json files — every sampled completion is verbose prose, often ending with '**B**' or 'the answer is B') will be graded incorrect by exact_match. Scenario also silently drops abc_score, potentially breaking questions that depend on a rendered musical example._

**Gaps:**
- run_specs/music_theory_bench_run_specs.py:30 temperature=0.7 diverges from paper's zero-shot deterministic MC protocol (paper uses greedy/temperature=0 for ChatMusician/GPT-3.5/LLaMA2 accuracy reporting); MC benchmarks should be temperature=0.
- run_specs/music_theory_bench_run_specs.py:31 stop_sequences=['\n'] combined with adapter output_prefix='Answer: ' is incompatible with models that emit chain-of-thought before the letter. display_predictions.json shows predictions like 'Let's analyze each option...' running 300+ tokens with the final letter buried at the end (e.g., id303 ends '**Therefore, the answer is B.**'); exact_match against 'A'/'B'/'C'/'D' references will score these as incorrect even when the model is right.
- run_specs/music_theory_bench_run_specs.py:29 max_tokens=512 plus the above causes many completions to exhaust tokens mid-reasoning without ever emitting a final letter (e.g., id141 truncates inside ABC notation analysis with no answer).
- run_specs/music_theory_bench_run_specs.py:34-36 MetricSpec uses BasicGenerationMetric with names=['exact_match']; paper reports accuracy on MC. rajkumar diagnosis note flags that the registry expects compute_reference_metrics for exact_match and BasicMetric for accuracy — current single spec may not emit the 'accuracy' metric the aggregator expects.
- scenarios/music_theory_bench_scenario.py:97 prompt matches ChatMusician eval/configs/datasets/music_theory_bench_ppl_zero_shot.py verbatim; however scenario drops the abc_score field entirely (line 33 comment concedes it is 'sometimes empty, sometimes contains notation'). Questions where abc_score carries the musical example and the stem only references it will be unanswerable.
- run_specs/music_theory_bench_run_specs.py:27 max_train_instances=0 matches paper's zero-shot setting — faithful.
- metric_notes/music_theory_bench_annotator_notes.md and metric_notes/music_theory_bench_eval_metrics_notes.md are both empty (1 line each); no documented metric rationale.

### `outline_to_story`
_Scenario loads WritingPrompts cleanly and generation is coherent, but the core task framing deviates from the O2S paper: raw Reddit prompts are used where cascaded per-paragraph event outlines are expected. Combined with a 512-token generation cap against ~1000-word references and reference-overlap metrics tuned for short answers (EM/QEM), the reported numbers will proxy short-form prompt-conditioned generation rather than the paper's fine-grained controllable long-story task. Paper PDF text could not be extracted by WebFetch; cross-check relied on arxiv abstract and GitHub repo README._

**Gaps:**
- Input construction drift: The O2S paper (arxiv:2101.00822) defines the task as generating multi-paragraph stories from *cascaded events* - sequences of outline events/keywords extracted per paragraph from the reference story (e.g., via keyword extraction). scenarios/outline_to_story_scenario.py:91,98 instead uses the raw Reddit writing prompt as the 'outline'. Raw WritingPrompts prompts are brief premises (e.g., '[WP] Leonardo DiCaprio...torpedoes his career'), not cascaded per-paragraph event outlines. This changes the task from fine-grained controllable generation to standard prompt-conditioned story generation.
- Dataset scope: Paper uses both WritingPrompts AND WikiPlots with paper-specific outline preprocessing (see GitHub repo fangleai/Outline2Story). scenario uses only euclaise/writingprompts with no outline preprocessing (scenarios/outline_to_story_scenario.py:82).
- Metric mismatch vs. paper intent: run_specs/outline_to_story_run_specs.py:35 uses exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4. For open-ended multi-paragraph creative fiction against a single reference, exact_match/quasi_exact_match will be ~0 and uninformative. The O2S paper reports BLEU plus diversity metrics (distinct-n) and perplexity; no distinct-n/diversity or embedding-based metric is registered. No controllability metric (e.g., outline-coverage) which is central to the paper's fine-grained-control claim.
- Missing metric_notes: metric_notes/outline_to_story_annotator_notes.md and outline_to_story_eval_metrics_notes.md exist but are empty (0/1 lines). No documented rationale for metric choices or deviations from paper.
- Generation length: adapter_spec max_tokens=512 (run_specs/outline_to_story_run_specs.py:29). Paper targets 'multi-paragraph stories in thousands of words'; reference stories in dataset are 500-2000 words per scenario docstring. 512 tokens will truncate outputs well below reference length, biasing BLEU/ROUGE/F1 downward and making length-sensitive metrics unreliable. Observed trial outputs confirm mid-sentence truncation (e.g., o2s_test_10689, o2s_test_12053 cut off mid-paragraph).
- Split/size: split='test' (15,138 examples) matches HF dataset; max_instances=0 means full eval - reasonable default but expensive. No max in the run_spec args.
- Output shape (trial_10inst): 10 stories generated, all coherent multi-paragraph creative fiction ~450-500 tokens each, ending mid-sentence due to max_tokens cap. Content quality looks fine; shape is sane modulo truncation.

### `poetmt`
_Scenario cleanly clones andongBlue/PoetMT and loads tang/song/yuan jsonl files with src/ref fields. Prompt template follows paper's Appendix B.2 shape. Judge config (gpt-4, 1-5 scale, temperature 0) matches paper. Three primary failure modes: (1) models emit long English commentary + multiple translations + pinyin analysis instead of a single clean translation, and the benchmark ingests the whole blob; (2) max_tokens=512 truncates discourse-level poems mid-verse; (3) judge rubrics see only the candidate text with no source Chinese or reference, undermining BS/BF/BM scoring validity. Reference-overlap metric set is broader (ROUGE-L/F1/exact_match) and narrower (no COMET/BLEURT) than the paper. RAG pathway intentionally omitted; this is baseline-only reproduction. metric_notes/poetmt_annotator_notes.md is empty. Paper PDF extraction failed (binary); cross-check used arxiv html v3 and andongBlue/PoetMT README._

**Gaps:**
- Prompt template awkward: scenarios/poetmt_scenario.py:167-171 uses translate_type=f'{dynasty_name.capitalize()} poetry' (e.g., 'Tang poetry'), producing 'Please translate this classical Chinese poem Tang poetry into an English poem Tang poetry: Poem:{chinese_poem}'. The paper (Appendix B.2, confirmed via arxiv html) uses the same slot-template but 'translate_type' is intended as a poetry-type label (e.g., 'Lüshi', 'Jueju', 'Ci', 'Sanqu'), not dynasty. Sampled outputs show the judged model interpreting 'Tang poetry / Song poetry / Yuan poetry' correctly as dynasty style, so the prompt is readable but lexically non-faithful to the paper's placeholder semantics, and doubles the phrase awkwardly.
- RAG 'Explanation:{rag_context}' field is intentionally omitted (scenario docstring lines 20-21, 166). Paper's headline method is RAT (retrieval-augmented translation) using the 30K-poem knowledge base; baseline-without-RAG is a legitimate configuration but the scenario does not expose the RAG pathway at all, so only the weakest paper condition is reproduced.
- Dataset size mismatch: scenario reports 790 poems (295 Tang + 196 Song + 299 Yuan, lines 30-33). Paper's discourse-level translation set is 608 poems (197 Tang / 189 Song / 222 Yuan) per arxiv html; 758 is the sentence-level adequacy split. The 790 figure appears to be raw jsonl rowcount from andongBlue/PoetMT `all_poems/*.jsonl` which includes entries beyond the discourse eval set. Not a faithfulness break per se, but the evaluated population is larger and differently composed than any table in the paper.
- Reference-overlap metrics: run_specs/poetmt_run_specs.py:72 registers exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4 via BasicGenerationMetric. Paper uses BLEU/BLEU-1..4, COMET (wmt22), and BLEURT (confirmed arxiv html). ROUGE-L and F1 are additions not in the paper; COMET and BLEURT are omitted. exact_match / quasi_exact_match are meaningless on free-form poetic translation and will be ~0.
- Judge rubric wording drift: run_specs/poetmt_run_specs.py:15-46 rubrics use generic phrasing (phonetic beauty, rhythm, structural fidelity, semantic depth) rather than paper's stated criteria (rhyme/meter/rhythm-harmony for BS; line alignment / balanced phrasing / structural consistency for BF; thematic depth / emotional resonance / vivid imagery for BM). Scale (1-5) and judge model (gpt-4) match paper. metric_notes/poetmt_eval_metrics_notes.md:62-81 documents a different prompt template (with source poem + reference translation in the judge context) which is NOT what the rubric-only annotator sends; judge sees only the candidate translation, not the source Chinese or reference, which materially weakens BM/BF/BS grading.
- Output shape bad: sampled display_predictions.json from gemini_pro and gemma_3_27b show predictions dominated by English meta-commentary ('Of course. This is a beautiful...', 'Here are a few English translations...', multi-option analyses, pinyin breakdowns, explanations of choices). The actual translation is buried in ~10-20% of max_tokens=512 output and often truncated mid-poem (e.g., instance poetmt_tang_16 cut mid-line, poetmt_song_415 cut at 'look to heaven's'). There is no extraction step to isolate the English translation before judging, so rubric scores reflect the whole blob including commentary; BLEU/ROUGE will be computed against the commentary+poem mixture vs reference translation and will be severely deflated.
- max_tokens=512 is too small for discourse-level Tang/Song regulated verse plus the commentary models like to emit; this is a generation-config issue interacting with the missing output-extraction step.
- metric_notes/poetmt_annotator_notes.md is empty (0 bytes); all annotator documentation is effectively in the run_specs file.

### `pollux_creativity`
_Scenario correctly pulls ai-forever/POLLUX test split, filters to seven creative task types, and preserves Russian-language instructions verbatim with all expert-annotation metadata attached to each Instance. However, the run_spec does not use any of that metadata: it wires up two generic English Likert rubrics scored 1-5 by GPT-4, which is effectively a different evaluation than POLLUX. The dataset itself is faithful; the evaluation protocol is not. display_predictions.json shows judge outputs clustered at 2-4 (plausible but on the wrong scale and wrong criteria). WebFetch of arXiv abstract confirmed 2,100 prompts, 35 task types, 7B/32B judge release, criteria-driven scoring — consistent with annotator notes but cannot verify exact 0-4 rubric text without full-paper access._

**Gaps:**
- Scoring scale mismatch: paper uses 0-4 integer scale per POLLUX rubric (metric_notes/pollux_creativity_annotator_notes.md:17-23), but run_specs/pollux_creativity_run_specs.py:19-34 defines generic 1-5 Likert rubrics for 'creativity' and 'originality'. display_predictions.json confirms outputs in 2/3/4 range, numerically overlapping but semantically misaligned with the paper's 0-4 anchors.
- Criteria mismatch: POLLUX defines ~8 task-specific creativity criteria (Креативность, Драматургия, Выразительность диалога, Качество рифмы, Литературные акценты, Соблюдение образа, Размер стиха, Попадание в жанр) per scenarios/pollux_creativity_scenario.py:80-89 and annotator_notes. run_specs collapses all of this into two generic English rubrics (creativity, originality), ignoring the dataset's per-instance criteria_name/rubrics/criteria_description fields exposed in extra_data (scenarios/pollux_creativity_scenario.py:169-178).
- Judge model mismatch: paper releases ai-forever/pollux-judge-7b-r and 32b-r as the canonical judges (annotator_notes:25-30); run_specs uses openai/gpt-4 with English rubrics instead of the Russian-native POLLUX judges or GPT-4 with the exact paper prompt (annotator_notes:190-215).
- Judge prompt language mismatch: POLLUX judge prompt is in Russian and follows the [FEEDBACK]...[RESULT] N [END] format (annotator_notes:190-220). Generic English rubrics used here do not pass instruction/reference_answer/criteria_name/rubrics to the judge, despite those fields being populated in extra_data.
- Missing reference/rubric injection: scenario surfaces ex['criteria_name'], ex['rubrics'], ex['criteria_description'], ex['reference_answer'], and ex['criteria_score'] (expert gold) in extra_data (scenarios/pollux_creativity_scenario.py:169-178), but the generic_llm_judge_annotator is not wired to consume them; no metric computes agreement with the expert criteria_score baseline.
- Deduplication by instruction (scenarios/pollux_creativity_scenario.py:138-144) discards per-criterion annotation multiplicity. Paper's 161,076 samples cover (instruction x model x criterion); keeping only one ex per instruction drops most criteria_name variants for that instruction, so whichever criterion happens to be first is the one retained.
- Generation settings: max_tokens=512 (run_specs:55) likely truncates literary text / story generation tasks; display_predictions.json shows several outputs cut mid-sentence. temperature=0.7 is reasonable for creative tasks.
- Adapter input_suffix='\n' with empty instructions/prefixes is fine for self-contained Russian instructions.
- Task filter: scenario includes 7 Russian task types (scenarios/pollux_creativity_scenario.py:69-77); docstring lists 6 but code splits 'ИИ как персонаж' into expert + informal variants. Minor doc mismatch, not a faithfulness issue.

### `pron_vs_prompt`
_Prompt and data pipeline are a faithful port of the paper's English condition (60 titles, verbatim system+user text). Evaluation layer diverges substantially: rubric scale (1-5 vs 0-3), rubric framing ('advertising' vs literary synopsis), and dropped dimensions (relevance, literary quality) mean current scores are not comparable to the paper's human-expert results. Compounded by max_tokens=512 truncation, runs cannot validly measure literary quality of ~600-word synopses. Safe as a generation probe once token cap and rubric are fixed; not a reproduction of the paper's findings. Paper PDF was not directly retrievable; cross-check used arXiv abstract, GitHub rubric.json summary in annotator_notes, and two display_predictions.json samples (gemini-2.5-pro, gemma-3-27b)._

**Gaps:**
- Rubric scale mismatch vs paper: paper/annotator_notes use 0-3 per dimension (relevance 0-4, literary 0/1) grounded in Boden's novelty/surprise/value. run_specs/pron_vs_prompt_run_specs.py:15-46 redefines each dimension on a 1-5 scale, changing both range and anchor semantics. metric_notes/pron_vs_prompt_eval_metrics_notes.md is empty (0 bytes), so no cross-reference documents this drift.
- Rubric content drift: run_spec rubric text frames output as 'promotional or advertising text' (pron_vs_prompt_run_specs.py:16, 27, 38) rather than a ~600-word literary movie-synopsis — wrong task framing for the judge.
- Missing rubric dimensions: paper uses 5 dimensions (attractiveness, originality, relevance, creativity, literary/critic). run_specs registers only 3 (attractiveness, originality, creativity); relevance and literary-quality/anthology/own-voice are dropped (pron_vs_prompt_run_specs.py:71-107).
- Judge model: run_spec uses openai/gpt-4 at temperature 0.0 (pron_vs_prompt_run_specs.py:81-83); annotator_notes.md:15 recommends GPT-4o. Paper itself used 3 human expert critics, not any LLM — LLM-as-judge is an acknowledged adaptation.
- max_tokens=512 (pron_vs_prompt_run_specs.py:66) truncates ~600-word target: observed outputs in display_predictions.json end mid-sentence (e.g., id26 'dissolving into', id35 'the', id11 'He') for both gemini-2.5-pro and gemma-3-27b — no synopsis reaches the required length.
- Prompt fidelity: scenarios/pron_vs_prompt_scenario.py:66-82 reproduces the GPT-4 English system+user prompt verbatim from 0_GPT4_prompts.ipynb; 60 titles loaded from canonical CSV with language='en' / title_origin='all' defaults — faithful.
- Output shape: judge returned integer scores in {4,5} with near-zero variance across instances in both sampled runs, consistent with a 1-5 scale but providing little discrimination — may reflect both truncation and rubric mismatch.

### `pun_eval`
_Generation prompt (scenario lines 75-89) and explanation prompt (lines 92-105) copied verbatim from Notebooks 5 (Method 1) and 2 (CoT recognition, side='pun') of github.com/Zhijun-Xu/PunEval. Dataset loaded from official GitHub raw URLs and correctly filtered to 1,457 fully ExPun-annotated entries from the 2,589 SemEval+ExPun pool (hom=1443, het=1146). Predictions in display_predictions.json (gemini-2.5-pro, gemma-3-27b) show clean JSON {"Sentence": "..."} outputs matching the paper's expected format; llm_judge scores span 1-4 reasonably. Main faithfulness issue is that the wired judge rubric is a generic 5-point quality scale rather than the paper's binary pun-detection classifier, and the explanation task is defined but not exposed via a run_spec._

**Gaps:**
- run_specs/pun_eval_run_specs.py:27-33 registers only task='generation' (default); task='explanation' declared in scenario but no run_spec exposes it, so only one of the two implemented tasks runs
- run_specs/pun_eval_run_specs.py:51-64 uses a 5-point generic 'llm_judge_quality' rubric; paper's Notebook 6 evaluation is a BINARY pun-detection judge (Choice: pun / non-pun), as documented in metric_notes/pun_eval_annotator_notes.md:27-47. Scores therefore do not correspond to paper's Pun Detection Rate.
- run_specs/pun_eval_run_specs.py:58 judge_model='openai/gpt-4' but annotator_notes.md:20 specifies GPT-4o to match paper lineup
- run_specs/pun_eval_run_specs.py:50 BasicGenerationMetric (exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4) is inappropriate for generation task — scenario comments (line 172) state many valid puns exist per keyword, so BLEU/ROUGE vs single human_text is a weak signal
- Paper's four core generation metrics (Ambiguity, Distinctiveness, Surprise, Unusualness) are documented in metric_notes/pun_eval_eval_metrics_notes.md but NOT implemented — only pun-detection (and even that as wrong rubric) is wired up
- run_specs/pun_eval_run_specs.py:45 temperature=0.7 diverges from paper's deterministic judge usage; acceptable for generation but should be 0.0 for judge (judge_temperature=0.0 is set correctly at line 59)
- No recognition task — correctly skipped per scenario docstring (not a creativity task)

### `research_idea_execution`
_Input pipeline (5-dim 1-10 peer-review prompt, structured-idea field extraction, Google Drive + GitHub fallback loaders) is a faithful operationalisation of the paper's Appendix A rubric — though adapted: paper never specifies an LLM prompt. Evaluation pipeline is substantially misaligned: annotators ask an LLM judge to grade an 'execution plan' (artifact the task does not produce) on three 1-5 dims that do not exist in the paper. Scenario also conflates the ideation study (data loaded) with the execution study (paper title). Headline scores measure 'gpt-4o quality/novelty/feasibility rating of a peer review mis-labeled as an execution plan', not the paper's execution-gap metric. Paper full text beyond the abstract was not retrievable via WebFetch; cross-check leaned on the scenario docstring, annotator notes, and abstract._

**Gaps:**
- Task re-specification (major): paper (arXiv:2506.20803) is an *execution* study — 43 expert researchers spent 100+ hours each executing assigned ideas into 4-page papers, which were then blind-reviewed. The scenario (scenarios/research_idea_execution_scenario.py:1-53) instead asks the model to *peer-review a structured idea* (pre-execution). The bench therefore measures review-writing on ideation-stage ideas, not execution-gap effects. Name 'research_idea_execution' is misleading.
- Data source drift: paper's execution-study reviews live in Execution_Study_Data.zip (Google Drive) and cover executed papers on the 5-dim rubric. Scenario instead loads Ideation_Study_Human_Ideas.zip (file id 1Z2Nd7WNNks-eCoqUgPzx1_ovYqU8OiPx) plus a 10-idea GitHub fallback (ai_researcher/prompts/idea_examples_method.json), which are *ideation-stage* idea specs. Expert reviews referenced in annotator notes (data_points_all_anonymized.json, 398 records) only contain novelty_score + novelty_rationale, not the full 5-dim data (metric_notes/research_idea_execution_annotator_notes.md:68-82).
- Prompt faithfulness: the review prompt in scenarios/research_idea_execution_scenario.py:97-126 reproduces the Appendix A 5-dimension rubric (Novelty, Excitement, Feasibility, Expected Effectiveness, Overall on 1-10) with per-dimension rationale. The scenario docstring explicitly notes 'no LLM prompt specified in the paper; this scenario uses the evaluation rubric as the prompt basis' (scenario.py:33-34) — reasonable adaptation, but a fabrication-by-adaptation flag.
- Metric drift (blocking): run_specs/research_idea_execution_run_specs.py:71-75 wires three generic llm_judge annotators (novelty, feasibility, quality) each on a 1-5 scale with custom rubrics (lines 15-46) that talk about 'research idea execution plan' quality. This has no correspondence to the paper's 5-dim 1-10 reviewer rubric and drops excitement, expected_effectiveness, overall entirely. Metric names ('llm_judge_quality') do not map to the paper.
- Judge-rubric mismatch: the judge is asked to evaluate 'the generated research idea execution plan' (run_specs:15-46) but the model's output is a *peer review of an idea*, not an execution plan. The judge rubric therefore grades the wrong artifact class; this likely drives the low scores seen in display_predictions (mostly 3/5 novelty, 5/5 feasibility, 2-5/5 quality in gemma_4_26b output).
- Judge-config drift: single judge openai/gpt-4o at temperature 0.0, max_tokens 512 (run_specs:81-106). Paper used 58 human expert reviewers (avg h-index 7, 15 pubs) — no LLM judge disclosed. Using a single LLM judge is a necessary adaptation but not a jury; the richer annotator_notes judge prompt (metric_notes:23-64 with calibration/reasoning/specificity/coherence on 1-5) is *not* the rubric wired into run_specs.
- Annotator-notes / run-specs divergence: metric_notes/research_idea_execution_annotator_notes.md:14-64 specifies a 4-dim review-quality judge (calibration, reasoning_quality, specificity, coherence on 1-5) referencing expert calibration data. run_specs instead implements three unrelated dims (novelty/feasibility/quality of an 'execution plan'). The documented annotator is not the one actually run.
- metric_notes/research_idea_execution_eval_metrics_notes.md is effectively empty (1 line / 0 usable content) — no eval-metric documentation for the headline scores.
- Adapter: max_tokens=512 (run_specs:66) is tight for a 5-dim review with rationales (~250-400 tokens observed in gemma_4_26b display_predictions); temperature=0.7 reasonable. No stop_sequences; input_suffix='\n' appends a benign newline.
- Output shape: gemma_4_26b display_predictions.json shows well-formatted 5-dim reviews with per-dim rationales matching the prompt spec; annotations resolve to 'generic_llm_judge_llm_judge_{novelty,feasibility,quality}' ints in 1-5. Shape is internally consistent with the (collapsed) metric wiring.
- Dedup logic keeps only first occurrence of each idea by title (scenario.py:264-276); zip + 10 fallback ideas combined yields an unspecified instance count. No explicit max_instances gate — instance count will vary by Google Drive availability.

### `scar`
_Prompt template faithfully reproduces the paper's Instruction 1 format and the full 400-instance test set is used. Main drift is metric plumbing: the registry expects stat 'f1' but run_spec wires BasicGenerationMetric which emits 'f1_score', so one required stat is missing at aggregation (a fix is already drafted). Secondary drift is evaluation shape: exact_match over str(list)-formatted gold will be brittle vs. model formatting and order variation, and the paper's set-based P/R/F1 over mapping pairs is not implemented. Valid to run tomorrow after wiring F1Metric; results should be interpreted on exact_match with the caveat that it undercounts mappings that are semantically correct but formatted differently._

**Gaps:**
- run_specs/scar_run_specs.py:30 temperature=0.7 for a deterministic structured-output task; paper framing (structure abduction with fixed gold mappings) implies greedy decoding (temp=0)
- run_specs/scar_run_specs.py:35 emits stat 'f1_score' via BasicGenerationMetric, but registry_metrics.yaml:1856 expects stat name 'f1' (token-level F1 from metrics/f1_metric.F1Metric); custom metric class exists but is not wired in -> missing metric at aggregation
- run_specs/scar_run_specs.py:29 max_tokens=512 with temperature=0.7 lets models produce long CoT/thinking before the list (see gemma_4_26b scar_133 'thinking_text' rambling); no stop_sequences and no post-processing to extract the final [[...]] mapping, so exact_match is brittle against free-form preambles
- scenarios/scar_scenario.py:153 gold reference is Python str(mappings) which yields single-quoted form (e.g. "[['Newton', 'Faraday'], ...]"); model may produce double quotes, different whitespace, or set-equivalent reordering -> set-based P/R/F1 from paper not implemented; exact_match understates performance
- scenarios/scar_scenario.py:128 items_a/items_b derived from sorted(set(m[0] for m in mappings)) leaks the mapping structure into the prompt via alphabetical ordering of items rather than the paper's original item list; could mildly bias models that align by list position
- run_specs/scar_run_specs.py:27 max_train_instances=0 (zero-shot); paper's Instruction 1 template is zero-shot-compatible so this is defensible, but paper also reports few-shot/CoT conditions not covered here
- metric_notes/scar_annotator_notes.md is empty (0 bytes); only scar_eval_metrics_notes.md is populated

### `science_analogies`
_Prompt templates and dataset parsing are faithful to plm_generator.py. The scenario and adapter correctly implement zero-shot single-prompt generation on the nosrc subset. Major faithfulness drift is at the metric layer: the paper's core evaluation is four-dimension human rating plus automatic precision over short reference Explanations, but the run registers only exact/quasi-exact/f1/rouge/bleu — all poor fits for the 500-2000 char essay-style outputs models actually produce. Empty metric_notes files confirm paper metrics were never transcribed. Secondary issues: temperature=0.7 vs paper's preferred temp=0, max_tokens=512 truncating outputs, and wsrc subset never run. Safe to execute as an open-ended generation probe; NOT a faithful reproduction of paper's human-eval or precision scores._

**Gaps:**
- Prompt templates match paper code verbatim: scenarios/science_analogies_scenario.py:47-48 uses prompts_nosrc[0]='Explain {target} using an analogy.' and prompts_wsrc[0]='Explain {target} using an analogy involving {source}.' from plm_generator.py (verified via raw.githubusercontent.com/Bhaavya/InstructGPT-Analogies/main/plm_generator.py)
- Dataset fields/size match: saqa.txt parsed into (Source, Target, Explanation) triples; nosrc subset dedupes to 109 targets, wsrc keeps 148 rows, consistent with scenario docstring and paper data description
- Subset selection: run_specs/science_analogies_run_specs.py:17 passes args={} so ScienceAnalogiesScenario defaults to subset='nosrc' (109 instances); wsrc subset (148 instances) is never run despite being implemented
- Temperature drift: run_specs:30 temperature=0.7 but paper reports best prompts at LOW temperature; plm_generator.py supports temperature=0 (deterministic) and temperature=0.85 (high-variability) modes, not 0.7. Paper abstract explicitly notes 'best prompts tend to be precise imperative statements especially with a low temperature setting'
- Model mismatch vs paper: paper evaluates text-davinci-001/002 InstructGPT variants; this benchmark evaluates Gemini/Gemma families (expected modernization, but no InstructGPT baseline run)
- Metric bundle is wrong for open-ended analogy generation: run_specs:35 uses exact_match/quasi_exact_match/f1_score/rouge_l/bleu_1/bleu_4. Paper's actual evaluation is HUMAN rating on four dimensions (meaningfulness, novelty, soundness, comprehensibility) collected via AMT (amt_res.xlsx, amt_src_res dir), plus an automatic 'precision' score over short Explanation snippets. Exact/quasi-exact match on 500+-token free-form analogies will be ~0 for every model; BLEU/ROUGE against a 1-sentence Explanation reference is a poor proxy for the paper's human-eval construct
- No LLM judge / no human-eval pipeline / no precision-over-explanation metric implemented; the metric_notes/science_analogies_annotator_notes.md and science_analogies_eval_metrics_notes.md files are EMPTY (0 bytes), so paper metrics were never transcribed
- Output shape: display_predictions show models emitting 500-2000 char multi-analogy essays with headings, bullet tables, and truncation at max_tokens=512 (e.g. gemini_pro id 'saqa_nosrc_chloroplast', 'saqa_nosrc_dna' end mid-sentence). Reference Explanations in saqa.txt are 1-3 sentence chegg/study.com snippets. Mismatch in length/format guarantees BLEU/ROUGE noise floor
- max_tokens=512 in run_specs:29 is well below plm_generator.py's max_tokens=939; causes mid-analogy truncation visible across display_predictions.json for all 14 models
- stop_sequences=[] combined with zero-shot single-turn generation is fine (no few-shot bleed issue), but no newline/period stop means verbose models run to the max_tokens cap

### `showerthoughts`
_Scenario correctly downloads the aiintelligentsystems/showerthoughts-dataset test file and filters to genuine entries. References are unused by the task since generation is unconditional — paper used human Showerthoughts only as ground-truth baseline in human survey, not as per-instance reference. Main faithfulness issues: (1) 3/5 evaluation dimensions missing, (2) 5-point rubric diverges from paper's 6-point scale, (3) rubric text is custom not paper-verbatim, (4) unconditional single-shot generation produces duplicate predictions (verified in gemini-2.5-pro display_predictions.json — first 5 instances all identical), making the 300-instance run effectively a single judgment repeated. Paper prompt was for batch-of-100 with diversity instruction; single-generation adaptation breaks the diversity mechanism._

**Gaps:**
- scenarios/showerthoughts_scenario.py:91-96 adapts paper's prompt: original asked for '100 Showerthoughts' with sentence-structure variation (Section 4.1); here modified to single-generation. Documented in docstring lines 37-39, but is a deviation — paper never evaluates one-at-a-time zero-shot outputs.
- Unconditional generation with identical prompt across all 300 instances yields identical outputs per model (display_predictions.json: every instance for gemini-2.5-pro returns the same 'Cleaning your house is just a side quest...' text). References (genuine showerthoughts) serve no functional role — scenario acknowledges this at lines 101-103 but still emits 300 duplicate instances.
- run_specs/showerthoughts_run_specs.py:71-75 implements only 3 of paper's 5 evaluation dimensions (creativity, humor, cleverness); Logical Validity and General Score are documented in metric_notes/showerthoughts_annotator_notes.md:17-21 but NOT wired up as MetricSpec/AnnotatorSpec.
- run_specs/showerthoughts_run_specs.py:15-46 uses 5-point Likert rubrics (1-5), but paper (annotator_notes.md:22) and human baselines (annotator_notes.md:109-116) use a 6-point scale. Scores will not be comparable to paper's Table 2 (Genuine=3.71, ChatGPT=3.23, etc.).
- run_specs/showerthoughts_run_specs.py:15-46 rubrics are custom-written anchors ('mind-bending', 'comedic quality') rather than paper's verbatim Likert prompts from annotator_notes.md:44-93 ('Rate the following Showerthought on creativity - is it creative, original, and novel?...').
- run_specs/showerthoughts_run_specs.py:81 judge_model='openai/gpt-4' matches annotator_notes.md:12 but paper itself predates GPT-4o; acceptable.
- scenarios/showerthoughts_scenario.py:59 loads 'roberta_test_data_mixed.ndjson' (test split of detector data with 50/50 genuine/generated); correctly filters to label=='genuine' (line 84). Data handling is faithful.
- metric_notes/showerthoughts_eval_metrics_notes.md exists but is empty (1-line file).
- adapter_spec temperature=0.7 with num_outputs=1 and identical prompt across instances is why outputs collapse; paper's ChatGPT runs requested 100 diverse showerthoughts in one completion, relying on in-completion diversity that this single-shot setup cannot produce.

### `slang_generation`
_Scenario data loading is correct (666 conv_slang.txt entries parsed from the upstream repo, test-only split) and the prompt adapts build_prompt_general() reasonably. The faithfulness gap is in evaluation: run_specs configures two generic LLM-judge rubrics (creativity, relevance) that have no connection to the paper's SBERT-based semantic-novelty or Morfessor-based morphological-coherence metrics. metric_notes.md documents the paper's metrics but the actual MetricSpec list does not implement them, so what will be reported is a pair of generic LLM-judge scores unrelated to Wu & Sun's framework. Outputs across Gemini/Gemma variants are consistently verbose (rationale, multiple examples, markdown headers) — acceptable for an LLM judge but would break any string-parsing metric._

**Gaps:**
- Paper's primary metric is Semantic Novelty (SBERT mean Euclidean distance between generated slang definition and standard dictionary definitions of the generated word) via novelty.py; scenario's metric_notes (metric_notes/slang_generation_eval_metrics_notes.md:7-20) documents this but run_specs (run_specs/slang_generation_run_specs.py:60-63) configures only generic LLM-judge creativity+relevance rubrics — no SBERT novelty, no Morfessor morphological coherence, no BLEU/ROUGE vs gold term
- metric_notes/slang_generation_annotator_notes.md is empty (0/1 lines) — no documentation of the LLM-judge annotator protocol actually configured in run_specs
- LLM-judge uses openai/gpt-4 (run_specs:69,79), a legacy model; other creativity tasks in this suite tend to use gpt-4o / stronger judges — the generic rubric also makes no reference to slang-specific criteria (e.g., coinage vs reuse, morphological plausibility, semantic distance from standard sense) from the paper
- Scenario implements only Freeform mode (scenarios/slang_generation_scenario.py:16-22,91-96); paper systematically compares three modes (freeform/coinage/reuse) — single-mode implementation means we can't reproduce the paper's central human-vs-machine comparison across modes
- Model outputs are multi-field verbose markdown (predicted_text in benchmark_output/runs/gemini_flash/.../display_predictions.json and gemma_3_27b/.../display_predictions.json include headers, rationale, multiple usage examples, emphasis formatting); paper evaluates on a clean extracted slang word + definition. No output parser is configured to isolate fields (1)/(2)/(3), so any downstream novelty/coherence metric (if added) would fail to find the target word reliably
- References stored are the gold human slang term with CORRECT_TAG (scenario:99-101) but no surface-overlap metrics (BLEU/ROUGE) are actually in metric_specs — the reference is loaded but never consumed; metric_notes:36-40 says BLEU/ROUGE 'are computed' but run_specs has no such MetricSpec

### `sonnet_or_not_bot`
_Scenario implementation closely mirrors paper's eval script structure (form_group
bucketing, 5000-char truncation, expert form label as gold). Primary risk is
adapter/generation-config mismatch (MC-joint + T=0.7 + \n stop) rather than
scenario logic. Accuracy metric itself is faithful in spirit._

**Gaps:**
- Prompt passes possible_forms as Python list repr (e.g. "['sonnet', 'ballad']") rather than a natural-language list — likely paper uses comma-joined form.
- temperature=0.7 inappropriate for deterministic classification; adds noise to accuracy.
- max_tokens=512 + stop="\n" conflict with verbose CoT outputs observed; some instances truncated.
- MC-joint letter scoring may disagree with paper's name-string accuracy when model answers with form name only (no letter).
- metric_notes/*.md files are empty (0 lines) — no documented rubric beyond code.

### `speak_to_structure`
_The scenario correctly downloads TOMG-Bench CSVs, preserves the Instruction field, and emits parseable SMILES outputs. However, the evaluation pipeline is a Tier-1 approximation plus an unrelated LLM-judge correctness score, not the paper's RDKit-based Validity/Success Rate/Novelty/Similarity protocol (metric_notes/speak_to_structure_eval_metrics_notes.md:6-15 describes the correct metrics but they are not implemented). Running as-is will not produce numbers comparable to arXiv:2412.14642 Table 2. Treat current outputs as a SMILES-plausibility probe only. Paper access via arXiv abstract only; leaned on annotator notes for metric specifics._

**Gaps:**
- Prompt augmentation: scenarios/speak_to_structure_scenario.py:77-85 prepends a custom `_SYSTEM_CONTEXT` ("You are an expert chemist...") to every Instruction. Paper (arXiv:2412.14642) evaluates LLMs on the raw Instruction field; this added system preamble changes the elicitation and is not part of the canonical TOMG-Bench prompt.
- Metric drift (critical): paper's headline metrics are Validity (RDKit parseability), Success Rate (constraint satisfaction via RDKit atom/bond counts, SMARTS matching, property calculations), Novelty (vs. ZINC/ChEMBL, MolCustom only), and Similarity (Tanimoto, MolEdit/MolOpt). run_specs/speak_to_structure_run_specs.py:49-52 wires only a generic `ValidityMetric` plus a 1-5 Likert `llm_judge_correctness`. No RDKit-based success rate, no Tanimoto, no novelty, no property-improvement check.
- Validity approximation: `metrics.validity_metric.ValidityMetric` likely implements the Tier-1 regex heuristic from metric_notes (balanced parens/brackets + atom-letter presence) rather than RDKit `Chem.MolFromSmiles`, so Validity numbers are not directly comparable to paper Table 2.
- LLM judge is unsuitable for this task: a 5-point correctness rubric over a SMILES string cannot reliably verify atom counts, functional groups, bond counts, Tanimoto similarity, or logP/QED/MR improvements. The judge in run_specs/speak_to_structure_run_specs.py:58 (openai/gpt-4) substitutes chemistry-aware scoring with a subjective judgment and will produce scores poorly correlated with paper's Success Rate.
- Judge model drift: paper does not use GPT-4-as-judge (success is deterministic via RDKit). Any scalar correctness score from a judge is an orthogonal signal, not a reproduction.
- Subtask-specific evaluation is collapsed: paper reports per-subtask Validity/Success separately across 10 subtasks; run_spec emits a single `llm_judge_correctness` averaged over all subtasks, losing MolCustom vs MolEdit vs MolOpt breakdown.
- Sampling: `max_instances_per_subtask=500` yields 5,000 instances vs paper's full 50,000; acceptable sampling adaptation, but should be logged.
- max_tokens=512 and temperature=0.7 — paper baselines generally use temperature=0.0/1.0 deterministic decoding for reproducibility; 0.7 introduces stochastic SMILES that complicate success-rate comparisons.
- Output shape (display_predictions.json) looks reasonable: single SMILES strings like `CCCCCCCCCCCCCCC`, `CC(C)([NH3+])CS[C@@H]1CCOC2(CCSCC2)C1`; the system preamble successfully suppresses explanatory text.
- Judge scoring suspicious: `CCCCCCCCCCCCCCC` (pentadecane) and `CCCC` (butane) both scored 5 and 3 respectively by the judge without the judge knowing the source constraint; confirms the judge is not reliably checking constraint satisfaction.

### `ss_gen`
_Generation prompt verbatim matches paper Section 4 (scenarios/ss_gen_scenario.py:85-89). Dataset URLs and JSONL loader correct; 5085 total instances load into TRAIN/VALID/TEST as paper specifies. Sample predictions show well-formed 200-300 word social stories scoring 5/5 on coherence across Gemini Pro and Flash-Lite runs. Primary drift: only 1/5 judge dimensions implemented and 5-shot vs zero-shot mismatch._

**Gaps:**
- Judge evaluates only 1 of 5 paper dimensions — only Coherence is scored (run_specs/ss_gen_run_specs.py:15-24,49-65). Paper evaluates 5 dimensions: Coherence, Descriptiveness, Empathy, Grammaticality, Relevance (metric_notes/ss_gen_annotator_notes.md:9-14).
- Judge rubric is a custom 1-5 anchored scale (run_specs/ss_gen_run_specs.py:15-24) not from the paper; paper prompt is not published, so fidelity cannot be verified.
- Missing paper's human-evaluation constraints: Structural Clarity (1-5), Descriptive Orientation / GR-Eight 2:1 ratio (Y/N), Situational Safety (Y/N) — not implemented as auto-checks (metric_notes/ss_gen_annotator_notes.md:35-52).
- Paper evaluates 200 randomly sampled test stories; scenario loads full 508-instance test split (scenarios/ss_gen_scenario.py:36-38,107-143) — superset, inflates judge cost but does not invalidate.
- metric_notes/ss_gen_eval_metrics_notes.md is empty (0 lines) — no eval-metrics spec documented.
- BasicGenerationMetric includes exact_match/quasi_exact_match/f1_score (run_specs/ss_gen_run_specs.py:50) which are not meaningful for long open-ended story generation; paper reports BLEU-4, ROUGE-1/2/L (scenarios/ss_gen_scenario.py:29). Missing ROUGE-1, ROUGE-2.
- max_train_instances=5 enables 5-shot few-shot from TRAIN_SPLIT (run_specs/ss_gen_run_specs.py:42); paper uses zero-shot title-only prompting per the instruction in Section 4.
- Adapter max_tokens=512 may truncate 200-300 word stories near boundary; temp=0.7 not verified against paper config (run_specs/ss_gen_run_specs.py:44-45).

### `story_generation_rocstories`
_Scenario downloads DeltaScore data correctly and produces syntactically valid prompts, but: (1) the scenario name says 'rocstories' yet loads both ROC and WritingPrompts; (2) max_eval_instances=20 on a 40-pool yields a mixed random subset; (3) all observed Gemini-3 Pro completions truncated to ~20 tokens with finish_reason='length' despite max_tokens=512, suggesting reasoning-token budget exhaustion; (4) metrics are BLEU/ROUGE/EM/F1 which neither match the DeltaScore paper's quality-dimension framing nor are informative on truncated one-sentence outputs. Faithfulness to the DeltaScore paper is weak - the paper is about metric validation on existing human-rated stories, not about benchmarking LLM story generation with n-gram metrics._

**Gaps:**
- Scope mismatch vs. name: scenario is named 'story_generation_rocstories' and the run_spec/groups use the same label, but scenarios/story_generation_rocstories_scenario.py:72-83 defaults dataset='both' and loads BOTH roc.jsonl AND wp.jsonl (20+20=40 prompts). run_spec passes args={} so the default 'both' is used. Concretely, benchmark_output/runs/gemini_3_pro/.../instances.json contains 8 roc_* and 12 wp_* instances - a mixed ROC+WritingPrompts sample, not ROCStories-only. Either rename the scenario to 'story_generation' or set args={'dataset':'roc'}.
- Sampling under max_eval_instances=20: run_spec.json shows max_eval_instances=20 is applied by the orchestrator to a 40-instance pool, yielding a random 20-sample mix. This silently halves the intended workload and produces a non-reproducible roc/wp ratio (here 8/12, not 10/10).
- Paper framing drift: DeltaScore (arxiv:2303.08991) is a *story-evaluation metric* paper, not a story-generation benchmark. It repurposes ROCStories and WritingPrompts human-rated stories to validate perturbation-based scoring against 5 quality dimensions (fluency, coherence, relatedness, logicality, interestingness). The scenario correctly acknowledges this in its docstring (lines 38-43) but then evaluates LLM-generated stories with surface-overlap reference metrics (BLEU/ROUGE/F1) that the paper itself does not endorse for open-ended creative writing - the paper's whole motivation is that such n-gram metrics correlate poorly with human judgment on these datasets.
- Metric mismatch: run_specs/story_generation_rocstories_run_specs.py:35 registers exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4. For open-ended story generation against a single human reference, exact_match/quasi_exact_match are vacuous (observed ~0.0 across all 20 instances). No diversity metric (distinct-n), no embedding-based similarity, no LLM-judge on the paper's 5 dimensions, and no DeltaScore-style perturbation metric. The registered metrics do not operationalize the paper's fine-grained quality dimensions.
- Missing metric_notes content: metric_notes/story_generation_rocstories_annotator_notes.md and story_generation_rocstories_eval_metrics_notes.md both exist but are empty (1 line / 0 bytes of content). No documented rationale for metric choices or deviations from paper.
- Output shape broken for Gemini-3 Pro trial: display_predictions.json shows 19/20 outputs are 60-130 characters (20-30 tokens) with finish_reason='length' despite max_tokens=512. Only 1 instance produced a full ~420-char story. num_completion_tokens reports 0.0 in stats.json. This looks like reasoning-token budget exhaustion: with Gemini 3 reasoning models, max_tokens=512 is consumed by internal thinking before any visible output is emitted, so the 'length' finish reason fires on invisible tokens. Net result: the model is being scored on one- or two-sentence openings, not stories. BLEU/ROUGE/F1 on these truncations are uninterpretable.
- Adapter prompt assembly: scenario wraps 'Write a story based on the following prompt:\n\n{prompt}\n\nStory:' (lines 121-127) AND adapter_spec adds input_suffix='\n' + output_prefix=''. Harmless but the 'Story:' inside the instance text plus the adapter's '\n' seam means the model sees 'Story:\n' - fine, but the prompt pattern is internal to scenario rather than the HELM adapter convention.
- Reference quality: references are the DeltaScore 'reference' fields which come from the original ROCStories/WritingPrompts corpora pre-tokenized (spaces around punctuation, lowercase, `` quotes, placeholders like [FEMALE]). Using these directly as BLEU/ROUGE references penalizes models that produce natural-cased prose with proper punctuation. Observed rouge_l ~0.07, bleu_4 ~3e-309 partly reflect this tokenization mismatch, not generation quality.
- Dataset size relative to paper: DeltaScore's human-rated set contains hundreds of (prompt, generated-story) pairs across multiple systems per prompt; the scenario deduplicates to 20 ROC + 20 WP unique prompts (scenarios/story_generation_rocstories_scenario.py:100-119). This is a reasonable reduction for a generation benchmark but leaves only 20 evaluation items per subset, which is small for reliable system comparison.

### `thenextchapter`
_Data loading is clean and subset sizes match the paper's sample_human.txt. Core deviations are (a) added instruction wrappers not used in the paper and (b) metric suite dominated by reference-overlap metrics (plus vacuous EM/QEM) while the paper's headline evaluation was 5-dimension human rating. A complete judge prompt is already drafted in annotator_notes.md but is not wired up (annotators=None). WebFetch of the arxiv abstract was thin; cross-check relied on scenario docstring, annotator_notes.md, and the paper's HumanEvaluation directory structure on GitHub (ZhuohanX/TheNextChapter)._

**Gaps:**
- Prompt drift: The paper (Xie et al., INLG 2023, arxiv:2301.09790) passes conditions directly to models without an instruction wrapper for the LLM baselines (e.g., GPT-3 Davinci is fed the raw condition). scenarios/thenextchapter_scenario.py:67-80,129 prepends subset-specific English instructions ('Continue the following story in a few sentences:', 'Write a short story based on the following prompt:', 'Continue the following news article:'). This is a reasonable modern adaptation but is an added instruction not present in the paper. Docstring acknowledges this (lines 32-33, 'no prompt specified in paper').
- Metric mismatch vs. paper intent: run_specs/thenextchapter_run_specs.py:35 registers exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4. The paper's headline evaluation is human rating on five 1-5 dimensions (fluency, coherence, relatedness, logicality, interestingness) plus automatic BLEU/ROUGE/METEOR/diversity. EM and QEM are vacuous for open-ended story generation (will be ~0). No LLM-as-judge annotator is wired up despite metric_notes/thenextchapter_annotator_notes.md specifying a complete 5-dimension judge prompt + calibration plan.
- Annotator not registered: annotators=None in run_specs/thenextchapter_run_specs.py:44, yet metric_notes/thenextchapter_annotator_notes.md documents a full GPT-4 judge protocol with 5 dimensions and pre-scored calibration items in HumanEvaluation/inhouse and HumanEvaluation/crowdsource. The judge plan is specified but not executed.
- Empty eval_metrics_notes: metric_notes/thenextchapter_eval_metrics_notes.md exists but is empty (1 line / empty file). No documented rationale for the chosen automatic metrics or deviation from paper.
- Subset coverage and sizes: scenario exposes roc (800), wp (1000), cnn (600) which match the paper's sample_human.txt sizes. No max_instances cap in the run_spec; full sweep is ~2400 instances per subset combination. Reasonable, but not sampled.
- [MALE]/[FEMALE]/[NEUTRAL] placeholders: roc subset conditions contain gender placeholders passed verbatim to the model. Paper also passed these as-is; faithful. Worth noting for judge prompts (annotator notes mention this).
- Generation config: max_tokens=512, temperature=0.7, num_outputs=1, zero-shot (max_train_instances=0). Temperature unspecified in paper; 0.7 is a standard default. ROC targets ~5-sentence continuations, WP/CNN are longer; 512 is adequate for roc but may truncate longer wp/cnn continuations and bias length-sensitive metrics.
- Output shape (trial_1inst, gemini-2.5-flash-lite): single instance generated a coherent ~3-sentence continuation (~45 words), consistent with roc-style short continuation. Shape sane.

### `tinyfabulist`
_Scenario correctly downloads the 100 benchmark prompts from the klusai/tinyfabulist GitHub repo (scenarios/tinyfabulist_scenario.py:49-123), matching the paper's Table 1 evaluation set. Judge model (openai/o3-mini-2025-01-31, temp 0.0, 350 tokens) matches paper exactly. Sample predictions show well-formed age-appropriate fables with populated grammar_score and creativity_score annotations. Main drift: only 2 of 4 paper dimensions scored, so results will not be directly comparable to paper baselines — safe to run but with documented caveat._

**Gaps:**
- Paper judge scores 4 dimensions in a single JSON-returning call (grammar, creativity, moral_clarity, adherence_to_prompt) plus best_age_group (metric_notes/tinyfabulist_annotator_notes.md:21-75); implementation runs only 2 separate per-dimension annotator calls for grammar_score and creativity_score (run_specs/tinyfabulist_run_specs.py:56-82) — moral_clarity, adherence_to_prompt, and best_age_group are dropped.
- Custom rubrics in run_specs/tinyfabulist_run_specs.py:15-31 are close to but not verbatim with the evaluator.yaml rubric; missing expert-literary-critic system prompt and JSON-only output instruction (metric_notes/tinyfabulist_annotator_notes.md:14-16).
- Corpus-level metrics (Self-BLEU, Distinct-1, Flesch Reading Ease) from paper (metric_notes/tinyfabulist_annotator_notes.md:99-106) are not implemented in metric_specs.
- Adapter max_tokens=512 (run_specs/tinyfabulist_run_specs.py:51) may truncate ~250-word fables; observed outputs in display_predictions.json fit but paper generation config is not explicitly verified.
- Scenario prepends SYSTEM_MESSAGE into the user input (scenarios/tinyfabulist_scenario.py:149) rather than using a true system role — behavior-equivalent but not a structural match.

### `tinystories`
_Scenario downloads the official 44 Evaluation_prompts.yaml correctly and runs zero-shot over the full test set. Sample outputs across gemini_flash and gemma_3_27b are well-formed story completions with all three integer 1-5 annotations populated (grammar/creativity/consistency). Drift is in judge scale (1-5 vs 1-10), per-dimension vs single-call structure, missing plot-coherence and age-group, custom rubrics, and judge-model override — all non-blocking for ranking models relatively._

**Gaps:**
- Judge scale is 1-5 per dimension in custom rubrics (run_specs/tinystories_run_specs.py:15-46), but paper uses 1-10 per dimension (metric_notes/tinystories_annotator_notes.md:9, 38); scores are not directly comparable to paper numbers.
- Judge is split into 3 separate per-dimension calls with custom rubrics (run_specs/tinystories_run_specs.py:77-107), but paper uses a single GPT-4 call grading Grammar, Creativity, Consistency, (plot coherence) plus age-group in one pass (metric_notes/tinystories_annotator_notes.md:26-39).
- Plot coherence and age-group estimation from the paper (metric_notes/tinystories_annotator_notes.md:31, 38) are not implemented — only 3 of 4 scored dimensions plus no age bucket.
- Custom rubric text in run_specs/tinystories_run_specs.py:15-46 is not from the paper; paper uses the student/teacher framing with no fixed rubric anchors (metric_notes/tinystories_annotator_notes.md:14, 21-23).
- Hardcoded judge_model_name='openai/gpt-4' (run_specs/tinystories_run_specs.py:81,90,99) hits HELM openai_responses_client pydantic bug; session used CREATIVITY_JUDGE_OVERRIDE=anthropic/claude-haiku-4.5 — judge identity differs from paper's GPT-4.
- metric_notes/tinystories_eval_metrics_notes.md is empty — no aggregation spec documented.
- Adapter max_tokens=512, temp=0.7 (run_specs/tinystories_run_specs.py:66-67) not verified against paper generation config.
- Scenario prepends a custom 'Complete the following story...' wrapper (scenarios/tinystories_scenario.py:69-74); paper's format just feeds the beginning — model outputs show leading 'Story beginning:/Story completion:' echo indicating the wrapper leaks into outputs.

### `writingbench`
_BORDERLINE benchmark per NOTES_FOR_VIJETA.md: only 18.3% (183/1000) of queries are in the Literature & Arts creative subset; the remaining 817 are academic, business, legal, education, or marketing writing — professional/technical tasks rather than creative generation. Whether this belongs in a creativity benchmark hinges on a team decision.
Separate from the scope question, the metric implementation is significantly unfaithful: the paper's core contribution is instance-specific rubric-based scoring using 5 criteria × 10-point scale loaded from the dataset's `checklist` field, and the run_spec discards all of this in favor of a single generic 1-5 rubric via GenericLLMJudgeAnnotator. Scores are therefore neither leaderboard-comparable nor methodologically representative of WritingBench. The scenario does carry `checklist` through extra_data, so retrofitting a faithful annotator is plausible without re-building the data pipeline.
Judge is gpt-4o rather than Claude (paper) and is pinned to deterministic temp; max_new_tokens=1024 vs paper 2048.
display_predictions for gemma_4_26b id993 (a physics lesson plan — Education domain, not creative) scored llm_judge_quality=5 (rubric ceiling), a likely symptom of the 1-5 scale compressing score resolution._

**Gaps:**
- Scope / contamination: scenario runs all 1,000 queries across 6 domains (Academic & Engineering 167, Finance & Business 210, Politics & Law 201, Literature & Arts 183, Education 111, Advertising & Marketing 128) — only the Literature & Arts subset (183) is unambiguously creative. The other 5 domains are professional/technical writing (contracts, financial reports, lesson plans, ad copy) and drift from the project's creativity criterion. Scenario constructor supports a `domain='literature'` filter (scenarios/writingbench_scenario.py:86-115) but run_specs/writingbench_run_specs.py:33 passes `args={}` and runs the full mixed set. BORDERLINE inclusion per NOTES_FOR_VIJETA.md section 1.
- Metric drift (major): paper uses 5 INSTANCE-SPECIFIC criteria per query stored in the `checklist` field, each with its own name, description, and 5 detailed scoring rubrics covering 1-10 integer scores (metric_notes/writingbench_annotator_notes.md:22-59). run_specs/writingbench_run_specs.py:15-24 replaces this with a SINGLE GENERIC 5-point rubric ("Score 1..Score 5") assessing overall quality. This collapses 5 dimensions × 1-10 scale → 1 dimension × 1-5 scale and completely discards the per-query `checklist` metadata carried in extra_data. Resulting scores are not comparable to WritingBench leaderboard numbers and lose the instance-specific rubric design that is the paper's main methodological contribution.
- Checklist is loaded but unused: scenario stores `checklist` in extra_data (scenarios/writingbench_scenario.py:164) but the GenericLLMJudgeAnnotator rubric is a static string not templated with per-instance criteria, so the 5 rubrics per query are silently dropped.
- Judge model mismatch: paper specifies Claude models (or a finetuned critic model) as the judge; run_specs/writingbench_run_specs.py:57 uses openai/gpt-4o. Judge temperature 0.0 vs paper's 1.0 (annotator_notes.md:178-183 reports scoring params top_p=0.95, temperature=1.0, max_length=2048); judge_max_new_tokens=1024 matches paper's 2048 order of magnitude but is lower.
- Output format mismatch: paper expects JSON `{score: int 1-10, reason: string}` per criterion; generic judge collects a single score on the 1-5 scale from the rubric (confirmed in display_predictions: `llm_judge_quality: 5` for gemma_4_26b instance id993 — ceiling effect suggests the 5-point ceiling is being hit).  This compresses score resolution and may inflate apparent performance.
- Aggregation: paper aggregates via mean across 5 criteria per instance; here there is one scalar per instance so no intra-instance aggregation happens, and instance-level heterogeneity across 6 domains is averaged uniformly, weighting non-creative domains 4x the literature subset.
- Generation adapter parameters: temperature=0.7, max_tokens=16000 match the paper's generation params (top_p=0.8, top_k=20 not plumbed through AdapterSpec but HELM rarely exposes these).
- Dataset fetch is correct: downloads benchmark_all.jsonl directly from X-PLUG/WritingBench GitHub raw (scenarios/writingbench_scenario.py:80-127). All 1,000 instances flow to TEST_SPLIT with empty references — appropriate for open-ended generation.
- Output shape sane: display_predictions for gemma_4_26b shows long multi-section writing outputs (lesson plans, papers) consistent with a 16k-token writing benchmark; annotation field populated with llm_judge_quality integer score.

