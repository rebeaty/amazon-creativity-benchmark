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
| ARN | Data on Google Drive (xlsx), not HuggingFace/GitHub | Use gdown to download from folder ID `1itOPXtorFEgweQCd71m2bIRwWAUHcXuf`. Prompt template in Appendix C.2. |
| ARN | GitHub link in benchmarks.json (404) | Correct data location is Google Drive via bit.ly/3t7qZ3S. Paper: arxiv.org/abs/2310.00996 |
| MiQA | Data nested deep in Google Research monorepo | TSV at `language/miqa/data/metaphor_inference_qa.tsv`. Each row generates 2 question types. |
| MUNCH | 4-way MC with apt/inapt labels | Correct answer depends on label combination: A-apt+B-inapt→A, etc. Prompts in `tasks/prompts.md`. |
| LCC Metaphor | Span indices are word-level | Use `text.split()[span[0]:span[1]]` to extract target word. Labels: "Metaphor"/"Non-metaphor". |
| LCC Metaphor | No paper-specified prompt | Original is probing study (ACL 2022), not LLM prompting. Using standard binary classification format. |
| AnaloBench | Prompt in code/t1.py | S1 uses 'Sentence' field, S10/S30 use 'Story' field. Dataset on HuggingFace: jhu-clsp/AnaloBench. |

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
| ARN | exact_match | `get_exact_match_metric_specs()` | Binary choice (1 or 2). Supports subsets: all, near_high, near_low, far_high, far_low |
| MiQA | exact_match | `get_exact_match_metric_specs()` | Binary choice (1 or 2). Subsets: all (300), implies (150), implied_by (150) |
| MUNCH | exact_match | `get_exact_match_metric_specs()` | 4-way MC (A/B/C/D). Subsets: word_implicit, word_mword, sent_implicit, sent_mword (1,492 each) |
| LCC Metaphor | exact_match | `get_exact_match_metric_specs()` | Binary (Yes/No). Multilingual subsets: en (8,028), es, ru, fa |
| AnaloBench | exact_match | `get_exact_match_metric_specs()` | 4-way MC (A/B/C/D). Subsets by length: s1/s10/s30, subset (340) or full (24.4k) |
| NYT Connections | exact_match | `get_exact_match_metric_specs()` | Word grouping (4 groups of 4 words). 652 puzzles. COLING 2025 Best Dataset Paper. |

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
| Collaborative Neural Painting (CNP) | Purely visual | No text component |
| Humor Transfer Learning | No dataset | "Code will be uploaded soon" |
| GuessBench | Multimodal | Minecraft images + text |
| LitBench | Data access issues | Test set only has IDs, requires joining with Reddit data |
| CreativeMath | Complex evaluation | 3-stage LLM-as-judge evaluation |
| FigLang-2024 Multimodal | Multimodal | Images + text |
| Eval3DAIGC-198 | Visual/3D | 3D generation evaluation |
| MineAnyBuild | Visual | Minecraft building |
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
