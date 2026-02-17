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
| HumorDB | Multimodal; images embedded as PIL objects in HuggingFace dataset | Images must be saved to temp files for MediaObject; use `output_path` for temp storage; binary classification with balanced 352/354 split |
| YesBut | Multimodal (PIL images); 9 examples have no difficulty label (None values) | Filter out None difficulty values; save PIL images to temp files; 1,075 valid examples (EASY: 954, MEDIUM: 107, HARD: 14); open-ended explanation task |
| Rebus Puzzle | HuggingFace dataset only has images; ground truth in GitHub repo | HF dataset (Kyunnilee/visual-puzzles) has 432 images; answers.json in GitHub repo (Kyunnilee/visual_puzzles) contains ground truth and skill annotations; download answers.json via urllib or use git clone |
| New Yorker Caption | 3 subsets: cartoons (parquet with images as bytes), ratings (captions/rankings), descriptions (GPT-4 scene descriptions); ~6K captions per contest | Load cartoons from parquet using hf_hub_download + pandas; images stored as `{'bytes': b'...'}` dict; contest_number maps to index via `contest_number - 530`; sample top caption + distractors from ranks 10-50; 47 test contests |
| MEMECAP | Images hosted separately from JSON annotations; previously marked "not_suitable" due to outdated assumption | JSON data in GitHub repo; images available on Kaggle or Google Drive (must download separately); 559 test, 5,823 train+val; meme interpretation task; successfully onboarded as multimodal scenario using MediaObject |
| HUMMUS | Multimodal benchmark (New Yorker cartoons + captions) | Use text-only ablation: `image_descriptions.csv` provides scene descriptions. Data in GitHub repo, not HuggingFace. |
| HUMMUS | Label "WIDLII" means "While In Doubt, Leave It In" | Treat as positive class (metaphorical) along with "Yes"; "Discard" items excluded from test_set.json |
| HUMMUS | Paper has 6 tasks; 4 are text-compatible | Tasks 3-4 (ImageBbox, ImageLabel) require visual grounding. Implemented: classification (940), naming (568), caption_highlight (568), explanation (568) |
| MACGYVER | Data in xlsx format on GitHub, not HuggingFace | Download xlsx via `?raw=true` URL suffix; load with pandas. File: `data/MacGyver/problem_solution_pair.xlsx` |
| Meta4XNLI | Multiple configs with language-specific splits | Use `int_eval` config with `xnli_test_met` split; filter by `language` field for EN/ES. Detection uses `det_en_finetune` config. |
| Meta4XNLI | Exact prompts in Appendix Table 29 | Zero-shot: "Say which is the inference relationship...answer only with one word between 'entailment', 'neutral' or 'contradiction'." Format: `{Premise} -> {Hypothesis}:` |
| Simile Generation | Data in CSV on GitHub, not HuggingFace | Download SimileEMNLP.csv from repo. Input: literal sentence, Output: simile. Human1/Human2 columns are references. |
| MACGYVER | Includes both solvable and unsolvable problems | For unsolvable problems (377 of 1683), expected response is to identify infeasibility. Include all in eval. |
| MACGYVER | Human-annotated evaluation | Paper uses fine-grained categories (efficient, inefficient, infeasible, etc.). See `annotator_notes.md` for judge setup. |
| ARN | Data on Google Drive (xlsx), not HuggingFace/GitHub | Use gdown to download from folder ID `1itOPXtorFEgweQCd71m2bIRwWAUHcXuf`. Prompt template in Appendix C.2. |
| ARN | GitHub link in benchmarks.json (404) | Correct data location is Google Drive via bit.ly/3t7qZ3S. Paper: arxiv.org/abs/2310.00996 |
| MiQA | Data nested deep in Google Research monorepo | TSV at `language/miqa/data/metaphor_inference_qa.tsv`. Each row generates 2 question types. |
| MUNCH | 4-way MC with apt/inapt labels | Correct answer depends on label combination: A-apt+B-inapt→A, etc. Prompts in `tasks/prompts.md`. |
| LCC Metaphor | Span indices are word-level | Use `text.split()[span[0]:span[1]]` to extract target word. Labels: "Metaphor"/"Non-metaphor". |
| LCC Metaphor | No paper-specified prompt | Original is probing study (ACL 2022), not LLM prompting. Using standard binary classification format. |
| AnaloBench | Prompt in code/t1.py | S1 uses 'Sentence' field, S10/S30 use 'Story' field. Dataset on HuggingFace: jhu-clsp/AnaloBench. |
| WebNovelBench | HuggingFace dataset has schema errors | Use raw JSON files from GitHub repo; 4 quality-gradient subsets (a/b/c/d); 100 novels × 10 chapters = 1,000 instances per subset |
| WebNovelBench | Synopses are model-generated | Input synopses extracted by Doubao-pro-32k, not ground truth; no human-written reference continuations provided |
| WebNovelBench | Chinese language benchmark | All data and evaluation in Chinese; requires Chinese-capable models for generation and DeepSeek-V3 for judging |
| V-FLUTE | Gated dataset requires authentication | HuggingFace dataset ColumbiaNLP/V-FLUTE requires login; set HF_TOKEN or run `huggingface-cli login` |
| V-FLUTE | 21 paraphrased instruction variants | Paper uses random sampling from 21 prompt templates (seed=42) to prevent instruction-specific overfitting |
| V-FLUTE | Specific output format required | Expects "[explanation]\nLABEL: [entailment or contradiction]" format; label extraction has fallback patterns |
| II-Bench | Test split has hidden answers | Test set (1,399 questions) answers marked with "?" for EvalAI leaderboard; use dev split (35 questions) for evaluation |
| II-Bench | 6-option multiple choice | Each question has 6 options (A-F), unlike typical 4-option MC; answer field contains letter (A-F), correct_option contains full text |
| StoryER | Dataset on Google Drive, not HuggingFace | Manual download required from Google Drive links; mixed format (pickle for ranking, JSON for rating/reasoning) |
| StoryER | Meta-evaluation benchmark | Primarily evaluates existing stories (ranking/rating quality) rather than generating creative content; Task 3 (reasoning/comment generation) is the only generative component |
| StoryER | Three distinct tasks with different formats | Task 1: binary preference (ranking pairs), Task 2: aspect ratings (0-1 scale), Task 3: comment generation (open-ended text) |
| Creation-MMBench | Multi-image multimodal benchmark | 1-9 images per question (most have 1: 677/765); images stored as list of PIL objects in 'image' field; instance-specific evaluation criteria stored as Python dict string in 'criteria' field (parse with ast.literal_eval) |
| Creation-MMBench | Mixed reference sources | 356/765 examples have human 'ground_truth', remaining 409 use 'reference_answer_by_gpt4o'; prefer ground_truth when available |
| Creation-MMBench | Instance-specific judge criteria | Each example has unique evaluation criteria dict with keys 'subjective requirement' and 'groundtruth alignment'; criteria must be passed to GPT-4o judge for proper evaluation |
| ESP dataset | COCO-format JSON structure | Dataset in COCO format (images dict + annotations dict); images from COCO 2014 val split; use coco_url field for image URLs |
| ESP dataset | Not all styles available per image | 5 styles total (sns, blog, news, story, instruction) but not every image has all styles; check field existence before creating instances; creates ~4,373 instances from 996 images |
| ESP dataset | Style-conditioned generation | Use style prefix prompts (sns:, blog:, news:, story:, instruction:) based on training code pattern; evaluated with standard caption metrics (BLEU-4, METEOR, CIDEr) |
| Unfun Corpus | TSV format with paired data | Format: unfun_id, unfunned_headline, funny_id, funny_headline, url; human unfunned headlines vary in style (some all caps, some mixed case); input is satirical, target is serious |
| Unfun Corpus | Two prompt styles available | Chat style: "You are a helpful assistant that edits humorous headlines to make them realistic"; Completion style: "The following humorous headlines can be edited to be realistic"; chat style default |
| Unfun Corpus | Anti-creativity task | Evaluates humor REMOVAL (unfunning) not generation; paper notes LLMs excel at removing humor but underperform at creating novel jokes; asymmetrical task for humor understanding |
| DeltaScore / Story Generation | Adapted from metric validation data | Original dataset contains pre-generated stories from 6 models (ROCStories) and 4 models (WritingPrompts) with human ratings; adapted by extracting unique prompts+references; 20 unique prompts per dataset (40 total) |
| DeltaScore / Story Generation | Dataset parameter for flexibility | Use dataset="roc" for ROCStories only, "wp" for WritingPrompts only, "both" for combined (default); ROCStories are short 5-sentence stories, WritingPrompts are longer creative fiction |
| DeltaScore / Story Generation | Multiple models per prompt in raw data | Raw data has 6 outputs per ROCStories prompt (BART, GPT-3, MEGATRON, PROGEN3, HINT, KGGPT2) and 4 per WritingPrompts (BART, DAVINCI, PROGEN3, HINT); scenario extracts one reference per prompt |
| BHP / Hypothesis Generation | Seen vs unseen test splits | test_seen (190) from earlier literature likely in training data; test_unseen (197) from recent publications for less contamination; both can be used for evaluation depending on goals |
| BHP / Hypothesis Generation | Data extracted by GPT models | Background-hypothesis pairs extracted from literature using GPT-3.5 or GPT-4; default to gpt-4 version (higher quality); formatted as numbered points (1), (2), (3)... |
| BHP / Hypothesis Generation | Few-shot prompting recommended | Repository includes 5 example background-hypothesis pairs used in multi-agent framework; scenario includes 3 few-shot examples in prompt for better performance; hypotheses should follow numbered format |
| CROWDCOUNTER | Data in GitHub, JSONL format | GitHub hate-alert/CrowdCounter; data in Datasets/CrowdCounter/ directory; JSONL format (one JSON per line); 1,288 test, 2,047 train, 100 val; counterspeech generation task |
| CROWDCOUNTER | Type name normalization needed | Dataset uses underscores/hyphens: empathy_affiliation → "empathy affiliation", warning-of-consequences → "warning of consequences"; 6 types total; both vanilla and type-specific prompts available; CoNLL 2024 |
| AVA | HuggingFace dataset incomplete | HuggingFace (Iceclear/AVA) has only images, no ratings/votes; must download AVA.txt from GitHub (imfing/ava_downloader) for vote distributions; only 'train' split available, no 'test' split; need to match images with ratings by ID |
| AVA | Vote distribution format | Columns 3-12 in AVA.txt contain counts of ratings 1-10; compute mean aesthetic score as weighted average: `mean = sum(ratings * counts) / total_votes`; ~200 votes per image; Column 2 is image ID for matching |
| AVA | Index-based matching solution | HuggingFace dataset has only 'image' field (no IDs); scenario uses index-based matching as fallback (assumes HuggingFace order matches AVA.txt Index column); 255,537 images in HuggingFace, 255,530 in AVA.txt; risky assumption but necessary; alternative: Kaggle dataset (nicolacarrassi/ava-aesthetic-visual-assessment) with CSV |
| AVA | Adapted as regression task | Original AVA is aesthetic assessment research dataset; adapted to score prediction task (1-10 scale) to evaluate visual aesthetic understanding; requires regression metrics (MAE, RMSE, Pearson correlation) instead of standard accuracy |
| AVA | Multimodal vision benchmark | Images stored as PIL objects in HuggingFace dataset; use MultimediaObject for prompt + image; CVPR 2012 paper (Murray et al.); 255,500 images total; scenario auto-downloads AVA.txt from GitHub |
| PerMPST | Data from tar.gz, not HuggingFace | Dataset at https://dl.fbaipublicfiles.com/perse/PerMPST.tar.gz (236MB); JSONL format; multiple files for k=0 through k=5 (number of historical reviews); use review.valid.c{k}.jsonl for validation; auto-download and extract in scenario |
| PerMPST | Prompt already formatted | Dataset includes pre-formatted 'prompt' field with full instruction + historical examples + new plot; don't reconstruct prompt from fields; use prompt directly as Input.text |
| PerMPST | JSON extraction from completion | Completion field contains JSON string with ```json markers; need to strip code blocks and parse JSON to extract score; use regex to remove ``` before json.loads(); default to 5.0 if parsing fails |
| PerMPST | Personalized evaluation task | Not story generation, but story evaluation from reviewer perspective; each example has same reviewer throughout historical + target; tests preference learning, not universal quality assessment; 92 unique reviewers in validation |
| PerMPST | Multiple k-value subsets | k=0-5 parameter controls historical context richness; all have same 915 validation examples but different prompts; k=1 (1 historical review) is paper's main version; higher k gives more context for personalization |
| WritingBench | Data in GitHub, not HuggingFace | Dataset at https://github.com/X-PLUG/WritingBench; JSONL format in benchmark_query/benchmark_all.jsonl; 1,000 queries (555 EN, 445 ZH) across 6 domains and 100 subdomains; requires ensure_file_downloaded from raw GitHub URL |
| WritingBench | Instance-specific evaluation criteria | Each query has 5 unique criteria in 'checklist' field with detailed rubrics for 5 score ranges (1-2, 3-4, 5-6, 7-8, 9-10); criteria vary by task type (academic, business, legal, literary, etc.); store in extra_data for judge access |
| WritingBench | Natural language prompts | No standardized prompt template; each query is a complete writing task in natural language with domain-specific context; use query field directly as Input.text; multilingual (criteria/rubrics in same language as query) |
| Cue-word Story | Dataset contains pre-generated stories with ratings | 479 stories (236 human, 243 AI) with expert/non-expert creativity ratings; ratings are for specific stories in dataset, not for new generations; use human stories as references, extract cue words from item_id field; 4 cue word sets total |
| Cue-word Story | Semantic distance parameter | 4 cue word sets classified by semantic distance: low (stamp-letter-send, petrol-diesel-pump) and high (gloom-payment-exist, organ-empire-comply); scenario supports semantic_distance parameter ("all", "low", "high") to filter sets; each set has 59 human reference stories |
| Cue-word Story | Adapted from comparative study | Original dataset for comparing human vs AI creativity; adapted to generative benchmark by extracting cue words and using human stories as references; similar pattern to DeltaScore/Story Generation; ICCC 2025; mismayil/creative_story_generation_dataset |
| GAUSS | 12 skill dimensions with category mapping | 41 graduate-level math problems across 12 dimensions; category field format: digit + letter (e.g., "12a", "12b"); first digit(s) map to dimension (1-12); dimension 12 is Creativity (3 problems); supports dimension parameter to filter ("12" for creativity only, "all" for all problems, "1"-"11" for specific dimensions) |
| GAUSS | Rubric-based scoring | Each problem includes: problem_statement (prompt), standard_solution (reference), rubric (scoring criteria), total_score (typically 1-3 points); rubrics stored in extra_data for LLM-as-judge evaluation; creativity problems have open-ended rubrics (e.g., "Award 1 point for each meaningful property/question") |
| GAUSS | Model outputs in dataset | Dataset includes model_name, model_response, model_score, evaluation fields (GPT-5-Thinking outputs); these are NOT ground truth, skip them; use standard_solution as reference; GaussMath/GAUSS on HuggingFace; hyperbolic.ai; arXiv:2509.18122 |
| FutureGen | Multiple CSV files with inconsistent schemas | Dataset has ACL (2012-2024) and NeurIPS (2021-2022) files with different column structures; only NeurIPS file (df_neurips_future_work_dataset.csv) has ground truth in future_work_combined field; ACL files only have paper sections without ground truth; use NeurIPS subset only; 278 papers total |
| FutureGen | Ground truth is combined from multiple sources | future_work_combined merges Future_Work_extraction (author-mentioned) + LLM_extracted_review_future_work (OpenReview peer suggestions); stored as JSON list string, needs parsing with json.loads() and joining; provides comprehensive ground truth from both authors and reviewers |
| FutureGen | Very long input contexts | df_Concatenated Text contains full paper text (often 20k+ tokens); includes OpenReview URL, title, keywords, and all sections; scientific writing generation task requiring long-context models; evaluated with ROUGE, BLEU, BERTScore or LLM-as-judge |
| GraphRAG-Bench | WRONG DATASET ONBOARDED | Mistakenly onboarded wrong dataset; jeremycp3/GraphRAG-Bench dataset used was NOT the correct dataset for the intended paper; need to identify correct dataset source and re-onboard; marked as wrong_dataset_onboarded in benchmarks.json; scenario file (if exists) should not be used |
| Outline to Story (O2S) | Uses WritingPrompts dataset | Paper proposes outline-to-story generation but uses existing WritingPrompts dataset (euclaise/writingprompts); prompts from Reddit r/WritingPrompts serve as outlines/guidance; 15,138 test, 15,620 validation, 272,600 train examples; stories typically 500-2000 words (multi-paragraph); supports split parameter and max_instances for limiting size; arXiv:2101.00822 |

## Common Patterns

- **Datasets requiring `trust_remote_code`**: Add `trust_remote_code=True` to `load_dataset()`
- **Suspected model output fields**: Check the paper to confirm field purpose before skipping

## Multimodal Support in HELM

**HELM supports multimodal benchmarks** through `MediaObject` and `MultimediaObject` classes. Use these for benchmarks with images, audio, or video inputs.

### Supported Content Types

| Modality | MIME Types | Location Type | Example |
|----------|-----------|---------------|---------|
| **Images** | `image/png`, `image/jpeg`, `image/gif`, `image/webp` | Local file path or HTTP(S) URL | Vision-language tasks, visual QA |
| **Audio** | `audio/mp3`, `audio/wav`, `audio/ogg` | Local file path or HTTP(S) URL | Audio understanding, speech tasks |
| **Video** | `video/mp4`, `video/webm` | Local file path or HTTP(S) URL | Video understanding tasks |
| **Text** | `text/plain` | Inline text (no location) | Part of multimedia sequence |

### When to Use Multimodal vs Text-Only

**Use MediaObject for:**
- Benchmarks where visual/audio content is essential to the task
- Vision-language models (VHELM framework)
- Tasks requiring actual image/audio understanding

**Use text descriptions for:**
- Spatial/visual information that can be serialized (see Sudoku-Bench)
- When models don't need to process actual media
- Text-only models being evaluated

### Key Imports

```python
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.multimodal_request_utils import (
    get_contents_as_bytes,
    get_contents_as_base64
)
```

### Quick Example

```python
# Create multimedia content with image + text
multimedia_content = MultimediaObject([
    MediaObject(content_type="text/plain", text="Question: What's in this image?"),
    MediaObject(content_type="image/jpeg", location="/path/to/image.jpg")
])

instance = Instance(
    input=Input(multimedia_content=multimedia_content),
    references=[...],
    split=TEST_SPLIT
)
```

See `examples/multimodal_visual_qa.py` for a complete example.

## Benchmarks That Don't Qualify

Some papers/repos don't meet the criteria for benchmark onboarding:

| Name | Issue | Reason |
|------|-------|--------|
| ANALOGYKB | Training data resource, not evaluation benchmark | No test set or evaluation task; authors evaluated on other benchmarks |
| CreataSet | Non-English (Chinese); meta-evaluation benchmark | Entire test set (3196 examples) is in Chinese; evaluates creativity evaluators not generators |
| DeepMath-Creative | Non-English (Chinese) | 78 proof problems + 101 counterexample problems; advanced mathematics; entirely in Chinese |
| Pron vs Prompt | Research study, not benchmark | One-time comparison of fixed model outputs (Pron vs GPT-4 vs Claude); no test set for new models |
| PACE | Evaluation metric, not benchmark | 110 seed words for generating association chains; no ground truth; measures creativity via semantic distance; pre-computed results |
| Creative Problem-Solving (CPS) Task | Test item generation framework | CPIG (Creative Psychometric Item Generator) generates CPS items to assess human creativity, not LLM creativity; iterative framework with shot selection algorithms (random, greedy, constraint satisfaction); no standardized test set for evaluating arbitrary LLMs; requires originality model weights (RoBERTa on RLPS) not publicly available; meta-benchmark evaluating whether LLMs can create valid psychometric assessments; Beaty Lab; arXiv:2409.00202 |
| Sonnet Generation (PoetryDiffusion) | Model training repository, not benchmark | Diffusion model for poetry generation; 335 test sonnets are for model validation, not benchmarking |
| TIGeR-Bench | Multimodal (text-to-image generation/retrieval); no decision labels | Unified text-to-image generation and retrieval benchmark; dataset (leigangqu/TIGeR-Bench) lacks ground-truth labels for "generate vs retrieve" decisions—only has creative/knowledge domain splits; decision-making version explored but infeasible without manual annotation; would require HEIM framework integration for full evaluation |
| LayoutSAM-Eval | Multimodal (layout-to-image generation) | Dataset HuiZhang0812/LayoutSAM-eval; evaluates image generation from bounding boxes + region captions; would require HEIM framework integration |
| Random Number Generation | Research study, no benchmark dataset | Paper arXiv:2505.00047 analyzes aligned vs base models on randomness tasks; no public test set or evaluation framework |
| Fann or Flop | Non-English (Arabic) | Dataset omkarthawakar/FannOrFlop; 6,984 Arabic poetry explanations across 12 eras; EMNLP 2025 |
| HumorousAI Benchmark | Multimodal (cartoon caption generation) | Dataset yguooo/newyorker_caption_ranking; 250M ratings on 2.2M captions for New Yorker cartoons; requires cartoon images; could be onboarded as vision-language task |
| HumorDB | Multimodal (visual humor understanding) | Dataset kreimanlab/HumorDB; 3,545 images (photos, cartoons, sketches); binary classification and funniness rating tasks; could be onboarded as vision-language task |
| PopBlends Evaluation Framework | No public dataset | arXiv:2111.04920; system for suggesting conceptual blends; user study only, no benchmark dataset available |
| Pun Understanding Evaluation | Complex multi-task requiring LLM-as-judge | GitHub Zhijun-Xu/PunEval; 3 tasks (recognition, explanation, generation); primary creativity tasks need LLM-as-judge and custom metrics not in HELM; recognition is just binary classification |
| RPGBENCH | Multi-turn interactive; requires LLM-as-judge | Dataset DongmingShenDS/RPGBench; 2 tasks (Game Creation, Game Simulation); multi-turn interactive gameplay simulation not compatible with HELM's single-turn architecture; arXiv:2502.00595 |
| QUDsim | Evaluation methodology, not benchmark | GitHub AlliteraryAlligator/QUDsim; similarity metric for measuring discourse structure similarities; analyzes LLM text reuse patterns; no ground truth evaluation task; arXiv:2504.09373 |
| MIXASSIST | Training dataset, not benchmark | Dataset mclemcrew/MixAssist; 640 conversational turns about music mixing; requires audio files (HELM supports audio via MediaObject); but no evaluation task or ground truth |
| LLM-MA Balderdash | Simulation framework; no public dataset | GitHub ParsaHejabi/Simulation-Framework-for-Multi-Agent-Balderdash; multi-agent Balderdash game; datasets require contacting authors |
| Speak-to-Structure (S2-Bench/TOMG-Bench) | Domain-specific chemistry benchmark | GitHub phenixace/TOMG-Bench; molecule generation from natural language; 3 tasks (MolEdit, MolOpt, MolCustom) with 15K samples; requires molecular structure validation; arXiv:2412.14642 |
| ArenaHard v2.0 | General LLM benchmark, not creativity-specific | Dataset lmarena-ai/arena-hard-auto; 500 diverse challenging prompts (software engineering, math, creative writing, etc.); evaluates overall model capabilities, not creativity; arXiv:2406.11939 |
| Design Problems Task (DPT) | Research study, not benchmark | GitHub Beaty-Lab/CogSci-2025-Scientific-Creativity; comparative analysis of human vs LLM creativity reasoning; human ratings/explanations of design problem responses; not a model evaluation task; arXiv:2502.03253 |
| ROBOTOOLBENCH | Domain-specific robotics; requires code execution | VLMgineer benchmark (arXiv:2507.12644); 12 robotic tool design and manipulation tasks; requires code execution in simulation environment (ManiSkill) for evaluation; engineering-focused creativity, not general text generation |
| CoMPAS3D | Multimodal (3D motion, video, audio); domain-specific | Dataset Rosie-Lab/compas3d; salsa dance motion capture; 3+ hours of improvised dance; motion generation and style transfer tasks; 3D motion/dance generation outside HELM's scope |
| Comparative Artistic Generation from LDM Prompts | No public information available | Paper ID 2a9db8d5445cf3f078ec608966cac6e09597ce74; no URL provided; no dataset or paper found in search results |
| HypoGen | Training dataset, not evaluation benchmark | Dataset UniverseTBD/hypogen-dr1; 5.5K paper-hypothesis pairs for fine-tuning models on scientific hypothesis generation (Bit→Flip task); no ground truth evaluation; uses LLM-as-judge for assessing quality; arXiv:2504.12976 |
| Co-Creative Meme Generation and Evaluation | Research study, not benchmark | arXiv:2501.11433; user study comparing human-only, human-AI collaboration, and AI-only meme generation; crowdsourced ratings on creativity, humor, shareability; no public dataset or evaluation framework |
| CLAP-based Novelty and Relevance Metrics | Model paper, not benchmark | arXiv:2308.01546; MusicLDM text-to-music generation system; CLAP-based metrics for assessing model outputs (relevance, novelty, quality); no test dataset with ground truth; audio generation requires HEIM framework |
| Computer Sciences Dataset | Multi-agent framework, not benchmark | arXiv:2309.17288; AutoAgents framework for automatic agent generation; Trivia Creative Writing benchmark (200 instances) created for framework validation; no public dataset release; requires multi-agent coordination |
| SciMON | Primary evaluation requires LLM-as-judge | arXiv:2305.14259; Scientific hypothesis generation from literature; public dataset (67K ACL papers, 5.7K PubMed); human evaluation by domain experts is primary (novelty, relevance, scientific validity); automatic metrics (ROUGE-L, BERTScore) acknowledged as inadequate for novelty assessment |
| CreativePair | Not a text creativity benchmark | arXiv:2508.12628; advertising image selection dataset; 8K image pairs for comparative evaluation; task is selecting which ad creative performs better (click prediction), not text generation; domain-specific advertising; dataset on Quark cloud storage |
| Crowd Vote | Evaluation platform, not benchmark | CreativityBenchmark.ai; Springboards industry initiative for evaluating LLM creativity in advertising; crowd voting platform (678 ad professionals, 11K comparisons); no public test set; proprietary marketing challenges; closed leaderboard service |
| Research Idea Execution Study | Research study, not benchmark | arXiv:2506.20803; Stanford study analyzing ideation-execution gap; 43 researchers implemented AI vs human research ideas (100+ hours each); one-time comparative experiment showing AI ideas degrade more when executed; human evaluation only; no reusable test set |
| Chinese Lyric Generation Evaluation | Non-English; research study | arXiv:2301.05402; In BLOOM study of Chinese lyric generation using BLOOM-176B; MojimLyrics dataset (39K songs from 230 artists) is training corpus; human evaluation for coherence and creativity; MAUVE metric noted as inadequate for creative writing |
| Listening Test for Agent-based Lyric Generation | Method paper, not benchmark | arXiv:2410.01450; multi-agent system for Mandarin lyric generation (rhyme, syllable count, lyric-melody alignment, consistency); one-time listening test with 22 participants ranking 4 versions of 3 songs for internal validation; Non-English (Mandarin); uses Mpop600 dataset (Chu et al. 2020, availability unclear); no public evaluation framework; O-COCOSDA 2024 |
| MOH-X | Method paper using existing datasets | arXiv:2504.11190; Logic-Augmented Generation (LAG) framework for metaphor detection; MOH-X (647 sentences) is existing dataset from prior work; metaphor detection is linguistic binary classification, not creative generation; repository contains framework implementation |
| SHOE SALES | Domain-specific scientific reasoning | arXiv:2504.11524; synthetic task in HypoBench for hypothesis generation from tabular data; predict shoe color from customer appearance; requires structured data input and inference testing; HDR metric compares against ground-truth hypotheses; not general text creativity |
| Research Ideation Evaluation Framework | Method/system paper, not benchmark | arXiv:2506.12317; Budget AI Researcher RAG-based research idea generator; comparative evaluation vs baselines (GPT-4o-mini, Claude 3.5); human evaluation for interestingness, novelty, feasibility; no standardized test set; repository contains system implementation |
| Human Evaluation of Fashion Generation | Method paper; image generation output | arXiv:2407.14944; AutoFashion fashion image generation using LLMs (Mistral-7B, Falcon-7B) + Stable Diffusion; human surveys evaluate visual appeal, relevance, creativity; comparative study of prompting techniques; domain-specific fashion; requires HEIM for image evaluation |
| Chinese Homophonic Puns and Slang Dataset | Non-English (Chinese); method paper | arXiv:2405.15818; DuanzAI system for slang-enhanced LLM with prompt engineering for humor understanding; GitHub YesianRohn/DuanzAI; enhances LLM comprehension of Chinese slang and cultural humor using phonetic matching and pinyin2hanzi conversion; includes PER-Task (punchline entity recognition, 0.97 F1) and Understand-Humor-Task; developed ChatDAI chatbot; entire dataset and evaluation in Chinese |
| Javanese and Sundanese Story Cloze | Non-English (Javanese and Sundanese) | arXiv:2502.12932; culturally grounded commonsense reasoning benchmark for Indonesian low-resource languages; classification and generation tasks using Story Cloze format; compares three data strategies: LLM-generated with cultural prompting, machine-translated from Indonesian, native-written reference stories; human evaluation for cultural fidelity, coherence, correctness; LLM-generated data outperformed other approaches; public dataset released; entire evaluation in Javanese and Sundanese |
| FlickrStyle10k | Training data only (no test set) | CVPR 2017 paper (StyleNet); stylized image captioning with humorous and romantic styles; only 7,000 training examples publicly available (download: https://zhegan27.github.io/Papers/FlickrStyle_v0.9.zip); no test/validation splits with ground truth; requires Flickr8k images as input; multimodal vision-language task; cannot evaluate without test labels |
| MusicSwarm Composition Benchmark | Method paper, not a benchmark | arXiv:2509.11973; swarm intelligence system for music composition using decentralized agents with stigmergic coordination; paper evaluates their own system using creativity metrics but provides no dataset or evaluation framework for benchmarking other models; GitHub repo (lamm-mit/MusicSwarm) does not exist; generative system demonstration, not an evaluation benchmark |
| Deceptive Humor Dataset (DHD) | Dataset unavailable | arXiv:2503.16031; multilingual humor-infused misinformation detection; dual classification task (satire level 1-3, humor type 5-way); 9,000 samples (7,200 train, 900 val, 900 test); GitHub repo and project website return 404 errors; no HuggingFace dataset; paper states restricted access requiring formal agreement, but no access request mechanism exists; published March 2025, may be in preparation |
| MoPS Premise Evaluation | Generation system, not evaluation benchmark | arXiv:2406.05690; Modular Story Premise Synthesis system; dataset (ManTle/mops) contains 7.6K model-generated premises/stories (outputs from MoPS system using GPT, RecurrentGPT, Dramatron); paper evaluates MoPS vs baseline generation methods using GPT-4 as internal judge; no evaluation task for testing external models; ACL 2024 |
| Beyond Memorization: Originality-Quality Frontier | Method paper, not benchmark | arXiv:2504.09389; proposes novelty evaluation methodology combining n-gram originality (fraction unseen during training) with task-specific quality scores using harmonic mean; uses three existing datasets as test beds (MacGyver, TinyStories, CoPoet); evaluation requires access to model training data for n-gram comparison; 2025; individual datasets should be evaluated separately for onboarding |
| AI Idea Bench 2025 | Meta-evaluation framework, not standalone benchmark | arXiv:2504.14191; dataset (yanshengqiu/AI_Idea_Bench_2025) contains 3,495 AI research papers with metadata (summaries, motivations, methods, citations); evaluates research idea generation systems via 6 separate methodologies (MCQ matching, idea-to-idea similarity, idea-to-topic alignment, competition, novelty assessment via literature search, feasibility assessment); missing core generation task prompts—dataset is ground-truth papers not evaluation instances; requires external dependencies (Semantic Scholar API, literature search) and multiple LLM-as-judge evaluations; better suited for custom evaluation pipeline outside HELM's single-task Scenario architecture; GitHub: yansheng-qiu/AI_Idea_Bench_2025 |
| Future Research Idea Generation Benchmark | Research study with pre-computed outputs | arXiv:2409.06185; "Can Large Language Models Unlock Novel Scientific Research Ideas?" by Kumar et al.; GitHub repo (sandeep82945/Future-Idea-Generation) contains Excel files with LLM-generated ideas from Claude-2, GPT-4, GPT-3.5, Gemini across 5 domains (Chemistry, CS, Economics, Medical, Physics); dataset has `Response_Chat` field with model outputs and `Future_work` field with authors' actual future work sections; comparative study evaluating alignment, distinctness, novelty, relevance, feasibility of those specific model outputs using IAScore and human evaluation; not a reusable benchmark—pre-generated outputs for one-time comparison, no standardized test instances for evaluating new models; EMNLP 2025 |
| Patent Novelty Evaluation Dataset | Not a creativity benchmark; dataset unavailable | arXiv:2502.06316; "Can AI Examine Novelty of Patents?" by Ikoma & Mitamura; binary classification task (Novel vs Non-Novel) comparing patent claims against prior art using USPTO Non-Final Rejection documents (2014-2015); 3,975 patents total (1,396 train, 166 val, 398 test after preprocessing); evaluates legal/technical judgment for patent examination, NOT creative generation or creative thinking; classification task following USPTO procedures ("whether each element is found in single prior art reference"); dataset not publicly released (no GitHub/HuggingFace); paper published Feb 2025; IPC G06F domain only; February 2025 |
| FunQA | Video-based benchmark; outside HELM scope | arXiv:2306.14899; "FunQA: Towards Surprising Video Comprehension" (ECCV 2024); 4.3K video clips with 312K free-text QA pairs across HumorQA (1,769 clips), CreativeQA (927 clips), and MagicQA (1,672 clips); tasks include timestamp localization, video description, reasoning, title generation, creativity scoring (1-20 scale); requires actual video files for temporal localization and visual understanding—paper states "86% thought FunQA cannot be solved only by images"; evaluation uses BLEU-4, ROUGE-L, CIDEr, BLEURT, GPT-4 judge; HELM doesn't support video understanding tasks; dataset: fesvhtr/FunQA on HuggingFace, GitHub: Jingkang50/FunQA; License: CC-BY-NC-SA-4.0; CreativeQA subset has creativity components but core benchmark requires video processing infrastructure not available in HELM |

| LeMat-GenBench (MOF/Crystal Generation) | Materials science evaluation, not creativity | arXiv:2512.04562; standardized evaluation framework for crystal generative models; evaluates thermodynamic viability and scientific validity (stability, novelty, validity), NOT creative quality; requires specialized materials science software (pymatgen, MLIP ensembles, MACE-MP, UMA, Orb-v3) for crystallographic analysis and energy predictions; models generate CIF (Crystallographic Information File) format; metrics: energy above convex hull, charge neutrality, structural fingerprinting; evaluates 12 generative models on unconditional generation of 2,500 structures; GitHub: LeMaterial/lemat-genbench; public leaderboard on HuggingFace Spaces; focuses on scientific correctness not creativity; December 2024 |
| L2M3OF (MOF Structure-Property) | Model paper, not creativity benchmark | arXiv:2510.20976; presents multimodal LLM for Metal-Organic Frameworks (MOFs); introduces MOF-SPK dataset (133,737 MOF structures from CCDC, 35,000+ publications); four tasks: property prediction, structure extraction (SMILES), description generation, Q&A; evaluates scientific understanding and property prediction, NOT creative generation; temporal split (pre-2020 train, post-2022 test); evaluation uses MAE for properties, BLEU/EXACT for structures, GPT-4-mini for head-to-head comparisons; borderline creativity—focuses on knowledge extraction and prediction rather than creative synthesis; requires molecular encoders for multimodal components; October 2024 |
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
| YesBut | open_ended | `get_open_ended_generation_metric_specs()` | Satirical image explanation; BLEU, ROUGE, F1 against ground truth descriptions; multimodal (vision-language) |
| Rebus Puzzle | open_ended | `get_open_ended_generation_metric_specs()` | Free-form text answers; exact string match (case/space-insensitive) or LLM-as-judge for semantic equivalence; 432 puzzles across 11 cognitive skill categories; multimodal (image + text prompt) |
| HUMMUS (classification) | exact_match | `get_exact_match_metric_specs()` | Binary Yes/No metaphor detection |
| HUMMUS (naming) | open_ended | `get_open_ended_generation_metric_specs()` | Conceptual metaphor ID; paper uses sentence similarity (LaBSE) |
| HUMMUS (caption_highlight) | exact_match | `get_exact_match_metric_specs()` | Tag metaphor text; paper uses Jaccard index |
| HUMMUS (explanation) | open_ended | `get_open_ended_generation_metric_specs()` | <=30 word explanation; paper uses ROUGE |
| MACGYVER (all) | open_ended | `get_open_ended_generation_metric_specs()` | Creative problem-solving; paper uses human annotation with fine-grained categories |
| Meta4XNLI (interpretation) | exact_match | `get_exact_match_metric_specs()` | 3-way NLI classification (entailment/contradiction/neutral) on metaphorical sentences |
| Meta4XNLI (detection) | open_ended | `get_open_ended_generation_metric_specs()` | Token-level metaphor identification; paper uses sequence labeling F1 |
| ARN | exact_match | `get_exact_match_metric_specs()` | Binary choice (1 or 2). Supports subsets: all, near_high, near_low, far_high, far_low |
| MiQA | exact_match | `get_exact_match_metric_specs()` | Binary choice (1 or 2). Subsets: all (300), implies (150), implied_by (150) |
| MUNCH | exact_match | `get_exact_match_metric_specs()` | 4-way MC (A/B/C/D). Subsets: word_implicit, word_mword, sent_implicit, sent_mword (1,492 each) |
| LCC Metaphor | exact_match | `get_exact_match_metric_specs()` | Binary (Yes/No). Multilingual subsets: en (8,028), es, ru, fa |
| AnaloBench | exact_match | `get_exact_match_metric_specs()` | 4-way MC (A/B/C/D). Subsets by length: s1/s10/s30, subset (340) or full (24.4k) |
| NYT Connections | exact_match | `get_exact_match_metric_specs()` | Word grouping (4 groups of 4 words). 652 puzzles. COLING 2025 Best Dataset Paper. |
| WebNovelBench | llm_judge | Custom Annotator with PCA aggregation | Chinese long-form narrative; 8 dimensions (1-5 scale); z-score normalization + PCA weights + percentile ranking; 1,000 instances per subset (a/b/c/d) |
| V-FLUTE | custom | F1@ExplanationScore metric needed | Multimodal visual entailment; binary classification (entailment/contradiction) + explanation; combines BERTScore F1 + BLEURT; F1 at multiple thresholds (0, 50, 53, 60, 70, 80, 90); 6,027 instances (4,578 train, 726 valid, 723 test) |
| II-Bench | exact_match | `get_exact_match_metric_specs()` | Multimodal image implication; 6-way MC (A-F); 35 dev instances (test answers hidden); domains: Life, Art, Society, Psychology, Environment, Others; paper evaluates with multiple prompt modes (zero-shot, CoT, few-shot, domain/emotion/rhetoric hints) |
| StoryER | mixed | Task-dependent metrics | Meta-evaluation benchmark with 3 tasks: (1) Ranking - binary preference (exact_match), (2) Rating - aspect scores (correlation coefficients: Spearman, Pearson, Kendall), (3) Reasoning - comment generation (open_ended: BLEU, ROUGE); 100k ranked pairs, 46k ratings/comments; 10 story aspects evaluated |
| Creation-MMBench | llm_judge | Custom Annotator with dual evaluation | Multimodal creative generation (1-9 images per question); GPT-4o judge (gpt-4o-0806) with instance-specific criteria; Visual Factuality Score (VFS, 1-10) + Reward (-100 to +100); dual evaluation (position swapping) to reduce bias; 765 test examples across 51 tasks in 4 categories; 356 human ground_truth, 409 GPT-4o references |
| Metaphoric Analogies | open_ended | `get_open_ended_generation_metric_specs()` | Extract 4-term metaphoric analogies (T1, T2, S1, S2) from literary texts; paper uses lemmatized head noun match, approximated with standard NLG metrics; 203 examples; some terms may be implicit (require inference); COLING 2025 |
| PoetMT | open_ended | `get_open_ended_generation_metric_specs()` | Classical Chinese poetry translation; standard NLG metrics (BLEU, ROUGE, F1); custom GPT-4 metrics for beauty dimensions (sound, form, meaning) documented in metric_notes.md; 790 poems (Tang, Song, Yuan dynasties); EMNLP 2025 |
| IDRBench | llm_judge | Custom Annotator needed | Single-turn research report generation; adapted from interactive multi-turn; GPT-4 judge on 4 dimensions (accuracy, completeness, coherence, citations); 15 business research questions; annotator configuration in annotator_notes.md; January 2025 |
| Cue-word Story | open_ended | `get_open_ended_generation_metric_specs()` | Creative 5-sentence story generation with 3 cue words; standard NLG metrics (BLEU, ROUGE, F1); optional LLM-as-judge for 4 creativity dimensions (creativity, originality, surprise, value) on 1-5 scale documented in metric_notes.md; 4 instances (cue word sets), 59 human reference stories each; ICCC 2025 |
| AVA | regression | Custom regression metrics needed | Aesthetic score prediction (1-10 scale); multimodal (vision); regression metrics: MAE, RMSE, Pearson correlation, Spearman correlation; ~25,000 test images with ~200 human votes each; metric implementation notes in metric_notes.md; CVPR 2012 |
| PerMPST | regression | Custom correlation metrics needed | Personalized story evaluation; predict reviewer-specific scores (1-10) from historical reviews; correlation metrics: Pearson r, Spearman ρ, Kendall τ; 915 validation examples across 92 reviewers; k=0-5 controls historical context; JSON parsing required; metric implementation in metric_notes.md; EMNLP 2023 |
| Outline to Story (O2S) | open_ended | `get_open_ended_generation_metric_specs()` | Controllable story generation from writing prompts; standard NLG metrics (BLEU, ROUGE, F1); prompts serve as outlines for multi-paragraph creative fiction; 15,138 test examples; long-form generation (500-2000 words typical); WritingPrompts dataset from Reddit community; arXiv:2101.00822 |

**HELM RunSpec patterns:**
- `exact_match` → `get_exact_match_metric_specs()`
- `open_ended` → `get_open_ended_generation_metric_specs()` (BLEU, ROUGE, F1)
- `summarization` → `get_summarization_metric_specs()`
- `llm_judge` → Custom Annotator required
- `regression` → Custom regression metrics (MAE, MSE/RMSE, Pearson r, Spearman ρ)
- `custom` → New metric implementation needed

## LLM-as-Judge Benchmarks

| Benchmark | Judge Model | Rubric Location | Dimensions | Annotator Notes |
|-----------|-------------|-----------------|------------|-----------------|
| Pun2Pun | GPT-4 or similar | eval/aacc_pun.py | Hit (binary: pun preserved?), Overlap (cosine similarity) | scenarios/pun2pun/annotator_notes.md |
| MACGYVER | GPT-4 (paper) | Paper Section 4.2, benchmark_results.json | correctness, feasibility, efficiency | `scenarios/macgyver/annotator_notes.md` |
| WebNovelBench | DeepSeek-V3 | Paper Section 3.2, novel_original_critic.py | 8 narrative dimensions (literary devices, sensory detail, character balance/distinctiveness/consistency, thematic/contextual alignment, scene coherence); PCA-weighted aggregation + percentile ranking | `scenarios/webnovelbench/annotator_notes.md` |

**Workflow:**
1. Create Scenario as normal (Scenario stays pure—no eval info)
2. Extract judge config per Step 3b → `scenarios/benchmark_name/annotator_notes.md`
3. Set `eval_type: llm_judge` in benchmarks.json
4. Common dimensions: novelty, usefulness, fluency, coherence, surprise

## Benchmarks Reviewed but Not Onboarded

| Benchmark | Reason | Notes |
|-----------|--------|-------|
| Open-ended Data-Driven Discovery (DiscoveryBench) | Scientific reasoning, not creativity | Task is discovering patterns in tabular data; evaluation is hypothesis correctness, not creative merit |
| Collaborative Neural Painting (CNP) | Purely visual | No text component |
| Humor Transfer Learning | No dataset | "Code will be uploaded soon" |
| GuessBench | Multimodal | Minecraft images + text |
| LitBench | Data access issues | Test set only has IDs, requires joining with Reddit data |
| CreativeMath | Complex evaluation | 3-stage LLM-as-judge evaluation |
| FigLang-2024 Multimodal | Multimodal | Images + text |
| Eval3DAIGC-198 | Visual/3D | 3D generation evaluation |
| MineAnyBuild | Visual | Minecraft building |
| APT Minecraft Construction | Gaming agent framework | LLM-driven Minecraft construction agent requiring Minecraft Java Edition + Mineflayer (npm package); repository has agent implementation (actions.py, agent.py, llmAlgorithms.py, memory.py) but no public test set; interactive 3D voxel construction evaluating spatial reasoning, Redstone systems, in-game rules; requires complex game environment integration; outside HELM scope; arXiv:2411.17255 |
| Video Metaphor Captioning (VMCD) | Video-based | Requires video understanding |
| PuzzleWorld | Multimodal | Images + text, 1-2% model accuracy |
| SciMuse | Private dataset | Contact authors to test |
| GODBench | Multimodal + unreleased | Video-based, "datasets will soon be released" |
| GPT-WritingPrompts | Not a benchmark | Analysis project comparing human vs GPT stories |
| RFBench (Reality-Fantasy) | Image generation | Scene generation, not text evaluation |
| CreativEval | Domain-specific | Hardware design creativity, not general |
| Oogiri-GO | Multimodal + broken | Images + loading errors |
| TRIG-Bench | Image generation | Text-to-image evaluation |
| Conceptual Design Generation | Open-ended + LLM-judge | 12 design problems, requires human/LLM evaluation |
| Creative Story Plan Generation (CritiCS) | Not a benchmark | Story generation system code |
| Curated Sentence Analogy | Gated dataset | Requires HuggingFace authentication |
| Standard Science Analogies (STD) | Open-ended generation | Analogy generation, not classification |
| Creative Process (Verbal Fluency) | Custom metrics | Analyzes creative exploration patterns, not output quality |
| DAT_GPT | Custom metrics | DAT scoring, compression-based DSI metrics |
| C2-Eval | Empty repo | No content available |
| FuxiBench (Fùxì) | Domain-specific | Chinese classical literature benchmark with 21 tasks; LLM-as-judge in Chinese |
| ConstructiveBench | Dataset inaccessible | HuggingFace URL returns "dataset not found" |
| PressRelease Creative Planning | Corpus, not benchmark | 656k articles corpus for research; no formal test set or evaluation metrics |
| Fable Generation (ds-tf1-en-3m) | Training data | 3M fables for training; not an evaluation benchmark |
| AraStories | No formal evaluation | Arabic story generation corpus; no defined test set or metrics |
| LLM-SRBench | Scientific reasoning | Equation discovery from data; not creativity |
| NeoCoder | Complex evaluation | Code execution + custom NeoGauge metric required |
| Story Cloze Test | Manual download required | Requires Google Form registration for data access |
| Passau-SFCH | Multimodal + restricted | Video+audio humor recognition; requires signed EULA |
| subtleBias/CoGS | Bias detection | Creative tasks used for bias analysis, not creativity evaluation |
| Persian Poem Generation | URL broken | resodate.com link redirects to landing page |
| The AI Scientist | Research system | AI research automation framework, not a benchmark |
| OpenAGI | Agent framework | Task composition framework, not creativity evaluation |
| Persona Generation Task | Repo 404 | GitHub repository doesn't exist |
| MuseScorer | Annotation pipeline | AUT scoring pipeline, no public evaluation data |
| IdeaBench | Dataset unavailable | KDD 2025 research idea generation benchmark; dataset hosted on anonymous.4open.science (403 Forbidden); 2,374 papers + 29,408 references; evaluation via GPT-4o ranking and Insight Score; no public GitHub/HuggingFace release; requires contacting authors; arXiv:2411.02429 |
| D-RAP (DRAP) | Dataset unavailable | DeepRapper rap generation dataset (ACL 2021); training dataset not publicly released; only sample data in Microsoft Muzic GitHub repo; no formal evaluation benchmark or test set; evaluation done on 5,000 randomly generated samples; arXiv:2107.01875 |
| CodeScientist | Research automation system | Automated scientific discovery system that generates experiments dynamically through genetic mutation of papers/code; no standardized test set or ground truth; evaluation via human expert review; ACL Findings 2025; similar to "The AI Scientist" - tool for conducting research, not for benchmarking models; arXiv:2503.22708 |
| Scientist-Bench (AI-Researcher) | Research automation system | Autonomous research system automating full pipeline (literature review, hypothesis generation, algorithm implementation, manuscript preparation); Scientist-Bench evaluates AI-Researcher system capabilities (93.8% completeness with Claude), not arbitrary LLMs; two levels: Level-1 executes given ideas, Level-2 formulates novel directions; open-ended research tasks not standardized creativity evaluation; similar to CodeScientist and The AI Scientist; NeurIPS 2025; arXiv:2505.18705 |
| Learn2Gen (LDC) | Method paper, not benchmark | Training framework (SFT + RL) for research idea generation; collected 6,765 ICLR/NeurIPS papers (2023-2024) for training their LDC model; no public test set with ground truth for benchmarking arbitrary LLMs; evaluation focuses on validating their model using GPT-4 reviewer with retrieval-augmented scoring; evaluates ideas on novelty, feasibility, effectiveness dimensions; GitHub repo du-nlp-lab/Learn2Gen is empty; similar to The AI Scientist, CodeScientist, Scientist-Bench; December 2024; arXiv:2412.14626 |
| Generative Materials Design (Materials Transformers) | Domain-specific materials science | Benchmark for generating inorganic material compositions (chemical formulas); trains 7 transformer models (GPT, GPT-2, GPT-Neo, GPT-J, BLMM, BART, RoBERTa) on materials databases (ICSD, OQMD, Materials Project); evaluates chemical validity (charge neutrality, electronegativity balance) using DFT calculations; computational materials science, not creative text generation; MT dataset on Figshare; June 2022; arXiv:2206.13578 |
| SS-GEN (SS-Bench) | Domain-specific educational content | Social Stories generation for ASD therapeutic intervention; constraint-driven generation with strict structural/pedagogical requirements (Carol Gray methodology); evaluates compliance with educational guidelines not creativity; dataset gated on HuggingFace (FMiMiY/SS-GEN); AAAI 2025; arXiv:2406.15695 |
| IGA | System/model paper, not benchmark | Intent-guided writing assistant (fine-tuned LM); releases training data (~1.2M examples across 7 intent types: cause, effect, contrast, description, biography, idiom, paraphrase) and model checkpoint; evaluation was system-specific (automatic metrics, crowdsourced judgments, user study) to validate IGA vs baselines, not for benchmarking arbitrary LLMs; EMNLP 2021; arXiv:2104.07000 |
| VideoCanvasBench | Video generation benchmark | Evaluates video diffusion models on arbitrary spatio-temporal video completion (inpainting, outpainting, extension, interpolation from user-specified patches); visual modality only, no text generation; outside HELM scope (requires video generation infrastructure); beyond HEIM scope (which handles text-to-image, not video); KwaiVGI; October 2025; arXiv:2510.08555 |
| HCL (Hallucination-Creativity Layerwise) | Model interpretation framework | Analyzes hallucination-creativity tradeoff across LLM decoding layers; uses existing datasets (TriviaQA, Natural Questions) for open-domain QA analysis; layer-wise early-exit strategy tool; measures hallucination (error rates) and creativity (semantic diversity); finds earlier layers more creative, deeper layers more accurate; no new test set for evaluating creative capacity; arXiv:2503.02851 |
| Focus-Level Evaluation Framework for LLM Reviews | Not a creativity benchmark | EMNLP 2025 meta-evaluation framework for analyzing LLM paper review focus; multi-label classification task (7 target facets × 5 aspect facets) categorizing review points from 676 ICLR papers; 3,657 expert strengths/weaknesses; evaluates comprehension of academic review content, not creative generation; dataset on Figshare (d5adf26c802527dd0f62); arXiv:2502.17086 |
| IRFL | Images stored separately in zip file; UUIDs as filenames | Download IRFL_images.zip (1.6GB) from HuggingFace; images named as {uuid}.jpeg; scenario auto-downloads to output_path/images/ on first run; 10,062 images for 810 test examples |
| IRFL | Answer/distractors are JSON strings containing UUID lists | Parse with json.loads(); answer contains single-element list, distractors contains 3-element list; UUIDs map to image filenames |
| IRFL | Definition field is JSON-encoded list for idioms | For idioms, definition field is '["text"]' format; parse with json.loads()[0] to extract string |
| MLLM-Bench | Meta-evaluation framework, not a standard benchmark | Uses pairwise comparison (win/tie/lose) against anchor model; no ground truth answers; per-sample criteria private; requires GPT-4V judge; only 9.5% creativity tasks. Not suitable for HELM. ALTERNATIVE: Creation-MMBench (opencompass/Creation-MMBench) has 765 creative test cases |
| PoetMT | Data in GitHub, not HuggingFace; multiple dynasty files | Clone from andongBlue/PoetMT; data in all_poems/ directory; three files (tang.jsonl, song.jsonl, yuan.jsonl) with 790 total poems; auto-download via git clone in scenario; paper mentions 758 but dataset has 790 |
| PoetMT | Hybrid creativity benchmark (translation + elegance) | Primary task is translation, but "elegance" (雅) dimension evaluates creative/poetic expression; standard metrics (BLEU/ROUGE) plus custom GPT-4 metrics (Beauty of Sound/Form/Meaning); see metric_notes.md for implementation |
| Style-conditioned Quatrain Generation (ByGPT5/uniformers) | Model implementation with training data, not evaluation benchmark | GitHub potamides/uniformers; ByGPT5 token-free transformer for poetry generation; QuaTrain training corpus (2.7M English + 5.9M German quatrains); pretrained model checkpoints released; examples/ directory has generation scripts and performance benchmarking (speed/memory), not quality evaluation; no standardized test set with ground truth quatrains for LLM benchmarking; ACL 2023 |
| IDRBench | Adapted from interactive to single-turn | Original is multi-turn interactive research system; adapted to single-turn report generation for HELM compatibility; uses only supporting facts (not distractors); 15 questions with 3-16 facts each; tests analytical reasoning not creativity |
| IDRBench | Requires LLM-as-judge evaluation | No ground truth reports; must use GPT-4 judge on 4 dimensions (accuracy, completeness, coherence, citations); see annotator_notes.md for prompt templates; small high-quality eval set |
| AlpacaEval Creative Writing | Method paper, not benchmark | Min-p sampling technique paper (arXiv:2407.01082) using AlpacaEval for validation; "AlpacaEval Creative Writing" appears to be unofficial subset used only in Min-p paper; AlpacaEval is instruction-following benchmark (805 examples from AlpacaFarm), not creativity-specific; uses GPT-4 judge for pairwise comparisons; no standardized creative writing subset officially released; evaluation logs in external repo (IlyaGusev/quest) |
| HumorBench | Element annotations unavailable | Paper's core contribution is expert-annotated element rubrics (1-3 essential facts per joke) for GPT-4o autograder evaluation; public dataset (jmhessel/newyorker_caption_contest, explanation_4 config) has 130 test examples with images, descriptions, captions, and ground truth explanations, but NOT the element annotations required for HumorBench evaluation; without elements, cannot replicate paper's evaluation method; arXiv:2507.21476 |
| Material Generation Benchmark | Text-to-structure generation task for materials science | Three datasets (MP-20, PEROV-5, Carbon-24) from CDVAE repo (github.com/txie-93/cdvae); LLMs generate CIF format crystal structures from composition+properties; custom evaluation metrics required (match rate, RMSD, validity) via pymatgen/ASE; CSV format with `cif` field as reference output; OpenReview 2025 paper |
| Material Generation Benchmark | MP-20 has 9 columns including nested lists | `elements` field is string representation of Python list: "['Ga', 'Te']"; parse if needed; `spacegroup.number` field has period in name |
| Material Generation Benchmark | CIF format is large structured text | CIF files are 500-2000 character structured text blocks with crystal parameters, atomic positions, symmetry operations; treat as text output but requires domain-specific structural comparison not string matching |
| Material Generation Benchmark | Not traditional creativity benchmark | Evaluates technical correctness (structural accuracy, chemical validity, physical plausibility) rather than creativity dimensions (novelty, usefulness, originality); de novo generation component could be considered creative; borderline case for creativity benchmarking |
| Hakka Cultural Knowledge Cognitive Domain Dataset | Not a creativity benchmark; dataset unavailable | Cultural knowledge benchmark evaluating Hakka culture comprehension across Bloom's Taxonomy's 6 cognitive domains (Remembering, Understanding, Applying, Analyzing, Evaluating, Creating); only 1/6 domains is creativity-related; focuses on cultural knowledge not creative thinking; 10,158 multiple-choice questions from Hakka Culture Encyclopedia; no public dataset found (no GitHub/HuggingFace release, IEEE paper paywall, no data availability statement); arXiv:2409.01556; October 2024 |
| Amuse Chord Generation | No ground truth references - generative task | 254 music keywords, each generates 30 chord progressions; evaluation measures diversity (Self-BLEU, lower=better) not accuracy; references are empty; requires Hooktheory dataset for coherence (JSD) metrics; CHI 2025 |
| Amuse Chord Generation | Batch vs. iterative prompting comparison | Paper tests whether generating 30 progressions in one prompt (batch) is more diverse than 30 separate queries (iterative); batch achieves Self-BLEU 0.30±0.12 vs. 0.61±0.18 for iterative; scenario supports both modes via diversity_eval parameter |
| Amuse Chord Generation | Complex prompt with musical theory | System prompt (2,315 chars) includes detailed instructions on chord functions, analysis, and formatting; specifies chord components (root, quality, extensions, suspensions, alterations, slash chords); uses music theory terminology (Tonic/Subdominant/Dominant) |
| Amuse Chord Generation | Hooktheory dataset required for evaluation | Download from https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz; 26,175 songs, filter for 'HARMONY' tag, transpose to C, extract progressions with 4+ chords; used for JSD coherence metric comparing generated vs. real music distributions |
| Amuse Chord Generation | Output format: one progression per line | Expected format: 30 lines, each with 4 chords separated by spaces (e.g., "C Em F G"); chords use specific notation (e.g., "Cmaj7", "Am", "Dsus4"); parser must split by newlines then spaces; validate chord symbols |
| Mathematical Framework for Objective Characterization of Ideas | Method paper, not benchmark | Proposes mathematical framework for evaluating idea quality/diversity in product design using embeddings (UMAP, DBSCAN, PCA); validates on 6 problem statements with 100 CAI-generated ideas each (600 total) but data not publicly released; software "not yet been publicly released"; tool for product designers to select promising ideas, not for benchmarking LLMs; no standardized test set with ground truth; arXiv:2409.07578; September 2024 (updated May 2025) |
| STYLEBREEDER | Image generation benchmark, not LLM benchmark | Evaluates text-to-image diffusion models (Stable Diffusion, SD-XL, ControlNet), not language models; 6.8M images + 1.8M prompts from 95K Artbreeder users; three tasks: style discovery (K-Means clustering), personalized image generation (LoRA, Textual Inversion, Custom Diffusion), style recommendation (matrix factorization); metrics: CLIP-I, CLIP-T, DINO for image quality; outside HELM scope (requires HEIM framework for image generation); gated dataset (requires HuggingFace authentication); CC0 license; NeurIPS 2024; arXiv:2406.14599 |
| InfoChartQA | Not a creativity benchmark | Multimodal question answering benchmark for understanding infographic charts (charts with pictograms/icons); evaluates visual comprehension and data interpretation, not creative generation or thinking; 5,948 pairs of infographic+plain charts with ~58,857 questions (~50,920 text-focused, ~7,937 visual-element focused); visual reasoning/chart understanding task similar to ChartQA; dataset on HuggingFace; arXiv:2505.19028; GitHub: CoolDawnAnt/InfoChartQA |
| VOILA | Not a creativity benchmark; audio/voice benchmark, not text LLM | Knowledge and reasoning benchmark for voice-language models evaluating factual accuracy, mathematical reasoning, code generation; 1,580 samples across 66 subjects (MMLU: 57 subjects/1,140 samples, MATH: 6/120, HumanEval: 1/100, NQ-Open: 1/100, GSM8K: 1/100); text questions converted to audio via Google Cloud TTS; designed for end-to-end voice systems processing audio inputs/outputs, not text LLMs; outside HELM scope; dataset: maitrix-org/Voila-Benchmark on HuggingFace; arXiv:2505.02707; May 2025 |
| POLLUX Creativity | Russian language benchmark | All instructions in Russian; 6-7 creativity task types out of 35 total; task_type field is in Russian (e.g., "Написать художественный текст"); requires Russian-capable models for generation and evaluation; December 2025 paper (arXiv:2505.24616) |
| POLLUX Creativity | Multiple model outputs per instruction | Dataset has 161K examples but only 2,115 unique instructions (~10K creative); 7 different models generated outputs per prompt; scenario deduplicates by instruction to get ~10K unique creative instances; extra_data stores task metadata |
| POLLUX Creativity | LLM-as-a-Judge with detailed criteria | 0-4 scale per criterion; 8 creativity-specific criteria (Creativity, Dramaturgy, Dialogue Expressiveness, Genre Appropriateness, Rhyme Quality, Verse Meter, Literary Accents, Character Adherence); criteria_name, criteria_description, rubrics, rubrics_example fields in dataset; expert annotations included as expected_score |
| POLLUX Creativity | No single correct answer for creative tasks | reference_answer field often empty for creative tasks; evaluation is criteria-based not correctness-based; References in Instance are empty or optional; focus on LLM-as-a-Judge evaluation (see annotator_notes.md) |
| POLLUX Creativity | Expensive expert annotation | Human annotation cost $262,316 for 24,447 hours across 161K samples; judge models released to replace human evaluation: ai-forever/pollux-judge-7b-r and pollux-judge-32b-r; key finding: SOTA models lag humans on creativity, criticized for "lack of soul" |
| POLLUX Creativity | Hierarchical task taxonomy | Three levels: task_type (35 types), task_subtype (88 subtypes), task_subsubtype (238 sub-subtypes); also has difficulty (5 levels), domain (6 domains); creative tasks span multiple levels (15 literary movements, 17 canonical authors, 93 substyles/genres) |
| Sonnet or Not, Bot? | Data in GitHub, not HuggingFace | Dataset auto-downloads from GitHub (maria-antoniak/poetry-eval/data/poetry-evaluation_public-domain-poems.csv); 1,453 public domain poems with expert form annotations from Poetry Foundation and Academy of American Poets; EMNLP Findings 2024 |
| Sonnet or Not, Bot? | Form group organization | Dataset has 'form_group' field organizing forms into 4 categories: verse forms (678 poems, 8 forms), stanza forms (315, 2 forms), types/modes (261, 9 forms), meters (199, 3 forms); create MC instances within each group |
| Sonnet or Not, Bot? | Poems truncated to 5000 chars | Paper's evaluation scripts truncate long poems (e.g., Canterbury Tales) to 5000 characters; preserves whitespace and line breaks as "central contribution" |
| Sonnet or Not, Bot? | Prompt format from evaluation scripts | Use exact prompt from paper's llm-poetry-tagger scripts: "Read the following poem and then choose the form... from one of these possible {form_group}: {possible_forms}" |
| FLUTE (Filtered) | Subset of original FLUTE dataset | Full FLUTE has 7,534 examples across 5 types (Metaphor, Simile, Sarcasm, Idiom, CreativeParaphrase); filtered version includes only Metaphor (1,250) + Simile (1,250) = 2,500 examples; filtering for "rhetorical clarity and visual interpretability"; HuggingFace: ColumbiaNLP/FLUTE; EMNLP 2022 |
| FLUTE (Filtered) | Used in Rhet2Pix paper for text-to-image | Filtered version created for rhetorical text-to-image generation (May 2025 arXiv:2505.22792); metaphors/similes chosen because they're visualizable; excludes sarcasm (no visual component), idioms (literal imagery misleading), creative paraphrase (focus on wording not imagery); scenario filters by type field |
| FLUTE (Filtered) | NLI task with explanations | Task: Given premise and hypothesis with figurative language, predict Entailment/Contradiction AND generate explanation; two output modes: classification only (faster) or classification + explanation (full evaluation); reference format: "Label\n\nExplanation: reasoning" |
| FLUTE (Filtered) | 7 semantic dimensions for downstream tasks | Rhet2Pix extracts: (1) rhetorical device, (2) literal subject, (3) metaphorical vehicle, (4) theme, (5) emotional tone, (6) subject keywords, (7) vehicle keywords; uses GPT-4o with generate-verify-retry loop; stored in extra_data for potential future extraction tasks |
| FLUTE (Filtered) | All data in single 'train' split | Original FLUTE has all 7,534 examples in 'train' split (no test split); scenario uses TEST_SPLIT tag for HELM consistency but data comes from 'train'; evaluation metrics: exact match for classification + BLEU/ROUGE/METEOR/BERTScore for explanations |
| ThemeVlogEval | Dataset not yet publicly available | Multimodal video generation benchmark from PersonaVlog paper (arXiv:2508.13602); evaluates vlog generation from theme + visual style + reference image; outputs include stories, storyboards, keyframe images, video clips, background music, character monologues; evaluation via MLLM-based storyboard metrics (1-5 scale), CLIP scores for images, and video quality metrics; paper states dataset "will be released" but no GitHub repo, HuggingFace dataset, or download links available as of Feb 2025; project page: https://personavlog-paper.github.io/; would require custom annotators and video evaluation infrastructure once released |
| GraphRAG-Bench | Inconsistent column schemas across question types | Dataset jeremycp3/GraphRAG-Bench has 5 JSONL files with different schemas; MC/MS have 'Choices' field, others don't; must load each file separately and combine; 1,018 total questions (FB:105, MC:217, MS:111, TF:316, OE:269); download individual files from HuggingFace |
| GraphRAG-Bench | Multi-Select answers are concatenated letters | MS answers like "ABD" represent multiple correct choices; parse as list of individual letters; format reference as comma-separated sorted choices (e.g., "A, B, D") for consistent ordering |
| GraphRAG-Bench | Domain-specific reasoning benchmark | Computer science domain knowledge from 20 textbooks spanning 16 disciplines; includes expert-crafted rationales with complete logical progression; Level-1/Level-2 topic hierarchies; more domain reasoning than creative generation |
