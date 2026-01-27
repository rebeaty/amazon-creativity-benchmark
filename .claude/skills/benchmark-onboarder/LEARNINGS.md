# Benchmark Onboarding Learnings

Add issues and patterns here as you discover them. Everyone on the team benefits.

## Dataset Quirks

| Benchmark | Issue | Solution |
|-----------|-------|----------|
| RiddleSense | Test split has no labels (empty `answerKey`) | Use validation split instead |
| ANALOBENCH | Field is `Sentence` not `Story` | Check actual dataset keys before coding |
| BRAINTEASER | No official HF dataset; many unofficial versions | Use `tasksource/brainteasers` with SP/WP configs; use `choice_order` and `label` fields for correct shuffling |
| Sudoku-Bench | Three subsets (challenge_100, nikoli_100, ctc); requires config name | Pass subset as parameter to scenario; visual_elements is JSON string needing parsing |
| Pun2Pun | Requires git clone; data in multiple JSON files; 4 sub-tasks | Clone repo, combine graphic.json+phonic.json, implement translation task only (primary creativity task) |
| SPLAT | Data in Excel (.xlsx); requires openpyxl; paper proposes interactive multi-turn framework | Install openpyxl, simplify to single-turn inference for HELM compatibility |
| NEOCODER | Requires git clone; `codes` field contains model outputs (NOT ground truth); 6 constraint levels per problem | Clone repo; use `problem_statements` for prompts and `test_cases_annotated.json` for evaluation; each problem has 1 original + 5 constrained variants |
| ProPara-Logy | Requires git clone; multiple CSV files with different distractor levels | Use `gold_set_analogies_w_challenging_distractors_w_randoms.csv` for full test set; 3-choice MC format |

## Common Patterns

- **Datasets requiring `trust_remote_code`**: Add `trust_remote_code=True` to `load_dataset()`
- **Suspected model output fields**: Check the paper to confirm field purpose before skipping

## Benchmarks That Don't Qualify

Some papers/repos don't meet the criteria for benchmark onboarding:

| Name | Issue | Reason |
|------|-------|--------|
| ANALOGYKB | Training data resource, not evaluation benchmark | No test set or evaluation task; authors evaluated on other benchmarks |
| MEMECAP | Multimodal (vision-based) | Requires meme images as input; HELM is text-only |
| CreataSet | Non-English (Chinese); meta-evaluation benchmark | Entire test set (3196 examples) is in Chinese; evaluates creativity evaluators not generators |
| DeepMath-Creative | Non-English (Chinese) | 78 proof problems + 101 counterexample problems; advanced mathematics; entirely in Chinese |
| Pron vs Prompt | Research study, not benchmark | One-time comparison of fixed model outputs (Pron vs GPT-4 vs Claude); no test set for new models |
| PACE | Evaluation metric, not benchmark | 110 seed words for generating association chains; no ground truth; measures creativity via semantic distance; pre-computed results |
| Sonnet Generation (PoetryDiffusion) | Model training repository, not benchmark | Diffusion model for poetry generation; 335 test sonnets are for model validation, not benchmarking |
| TIGeR-Bench | Multimodal (text-to-image) | Unified text-to-image generation and retrieval benchmark; HELM is text-only |
| LayoutSAM-Eval | Multimodal (layout-to-image) | Dataset HuiZhang0812/LayoutSAM-eval; evaluates image generation from bounding boxes + region captions; HELM is text-only |
| Random Number Generation | Research study, no benchmark dataset | Paper arXiv:2505.00047 analyzes aligned vs base models on randomness tasks; no public test set or evaluation framework |
| Fann or Flop | Non-English (Arabic) | Dataset omkarthawakar/FannOrFlop; 6,984 Arabic poetry explanations across 12 eras; EMNLP 2025 |
| HumorousAI Benchmark | Multimodal (cartoon caption generation) | Dataset yguooo/newyorker_caption_ranking; 250M ratings on 2.2M captions for New Yorker cartoons; requires cartoon images; HELM is text-only |
| HumorDB | Multimodal (visual humor understanding) | Dataset kreimanlab/HumorDB; 3,545 images (photos, cartoons, sketches); binary classification and funniness rating tasks; HELM is text-only |
| PopBlends Evaluation Framework | No public dataset | arXiv:2111.04920; system for suggesting conceptual blends; user study only, no benchmark dataset available |
| Pun Understanding Evaluation | Complex multi-task requiring LLM-as-judge | GitHub Zhijun-Xu/PunEval; 3 tasks (recognition, explanation, generation); primary creativity tasks need LLM-as-judge and custom metrics not in HELM; recognition is just binary classification |
| RPGBENCH | Multi-turn interactive; requires LLM-as-judge | Dataset DongmingShenDS/RPGBench; 2 tasks (Game Creation, Game Simulation); multi-turn interactive gameplay simulation not compatible with HELM's single-turn architecture; arXiv:2502.00595 |
| QUDsim | Evaluation methodology, not benchmark | GitHub AlliteraryAlligator/QUDsim; similarity metric for measuring discourse structure similarities; analyzes LLM text reuse patterns; no ground truth evaluation task; arXiv:2504.09373 |
| MIXASSIST | Multimodal (audio); training dataset, not benchmark | Dataset mclemcrew/MixAssist; 640 conversational turns about music mixing; requires audio files; no evaluation task or ground truth |
| LLM-MA Balderdash | Simulation framework; no public dataset | GitHub ParsaHejabi/Simulation-Framework-for-Multi-Agent-Balderdash; multi-agent Balderdash game; datasets require contacting authors |
| Speak-to-Structure (S2-Bench/TOMG-Bench) | Domain-specific chemistry benchmark | GitHub phenixace/TOMG-Bench; molecule generation from natural language; 3 tasks (MolEdit, MolOpt, MolCustom) with 15K samples; requires molecular structure validation; arXiv:2412.14642 |
| ArenaHard v2.0 | General LLM benchmark, not creativity-specific | Dataset lmarena-ai/arena-hard-auto; 500 diverse challenging prompts (software engineering, math, creative writing, etc.); evaluates overall model capabilities, not creativity; arXiv:2406.11939 |
| Design Problems Task (DPT) | Research study, not benchmark | GitHub Beaty-Lab/CogSci-2025-Scientific-Creativity; comparative analysis of human vs LLM creativity reasoning; human ratings/explanations of design problem responses; not a model evaluation task; arXiv:2502.03253 |
| ROBOTOOLBENCH | Multimodal (images + code); domain-specific robotics | VLMgineer benchmark; 12 robotic tool design and manipulation tasks; requires environment images and source code; robotics-focused, not general text creativity; HELM is text-only |
| CoMPAS3D | Multimodal (3D motion, video, audio); dance/motion generation | Dataset Rosie-Lab/compas3d; salsa dance motion capture; 3+ hours of improvised dance; motion generation and style transfer tasks; HELM is text-only |
| Comparative Artistic Generation from LDM Prompts | No public information available | Paper ID 2a9db8d5445cf3f078ec608966cac6e09597ce74; no URL provided; no dataset or paper found in search results |

**Disqualification criteria:**
- No evaluation method (no accuracy/metrics/human ratings)
- Training data only (no test set)
- Not a creativity task
- Dataset unavailable

## Style Conventions

- Scenario class names: `{Benchmark}Scenario` (e.g., `RiddlesenseScenario`)
- `name` field: lowercase with underscores (e.g., `riddlesense`)
- `description` field: data source reference, not task description
- Always include `tags = ["creativity", ...]`

## Evaluation Types Encountered

| Benchmark | Eval Type | HELM Pattern | Notes |
|-----------|-----------|--------------|-------|
| Sudoku-Bench | exact_match | `get_exact_match_metric_specs()` | Match 81-digit solution string; open-ended generation format |
| Pun2Pun | llm_judge | Custom Annotator needed | Hit metric (binary pun detection) + Overlap metric (semantic similarity via embeddings) |
| SPLAT | open_ended | `get_open_ended_generation_metric_specs()` | BLEU, ROUGE, semantic similarity; could also use LLM-as-judge for reasoning quality |
| NEOCODER | custom | External evaluation required | NeoGauge@T metric: correctness (code execution) + technique detection (constraint violation checking); requires code execution sandbox |

**HELM RunSpec patterns:**
- `exact_match` → `get_exact_match_metric_specs()`
- `open_ended` → `get_open_ended_generation_metric_specs()` (BLEU, ROUGE, F1)
- `summarization` → `get_summarization_metric_specs()`
- `llm_judge` → Custom Annotator required
- `custom` → New metric implementation needed

## LLM-as-Judge Benchmarks

| Benchmark | Judge Model | Rubric Location | Dimensions | Annotator Notes |
|-----------|-------------|-----------------|------------|-----------------|
| Pun2Pun | GPT-4 or similar | eval/aacc_pun.py | Hit (binary: pun preserved?), Overlap (cosine similarity) | scenarios/pun2pun/annotator_notes.md |

**Workflow:**
1. Create Scenario as normal (Scenario stays pure—no eval info)
2. Extract judge config per Step 3b → `scenarios/benchmark_name/annotator_notes.md`
3. Set `eval_type: llm_judge` in benchmarks.json
4. Common dimensions: novelty, usefulness, fluency, coherence, surprise
