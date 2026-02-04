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
