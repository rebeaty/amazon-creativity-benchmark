# Creativity Benchmark Registry Summary

## Overview
- Total datasets: 158
- Datasets with reference targets: 108
- Datasets without reference targets: 50
- Datasets using defaults for generation config: 152
- Datasets with specific generation config: 6 (aidanbench, amuse_chord_generation, cpers, eqbench_creative_writing_v3, noveltybench, writingbench)

## Modality Breakdown

### Input Modality

Input modality fields are now lists (datasets may have multiple input modalities).

| Input Modality Combination | Count | Example Datasets |
|---------------------------|-------|-----------------|
| [text]                    | 122   | aaar, aidanbench, analobench, … |
| [image, text]             | 25    | artinsight, cii_bench, memecap, newyorker_caption, … |
| [image]                   | 8     | ava, csd100, fscg8, humordb, kiva, vgsg, yesbut, yesbut_v2 |
| [video]                   | 1     | funqa |
| [video, text]             | 1     | muse_perception |
| [image, text, graph]      | 1     | mars |

### Output Modality

Output modality fields are now lists.

| Output Modality | Count | Datasets |
|----------------|-------|---------|
| [text]         | 151   | most datasets |
| [image]        | 4     | banner_request_400, calligrapher, csd100, fscg8 |
| [other]        | 3     | recombination_extraction, rpgbench, textlogo3k |

## Metric Type Breakdown

| Metric Type    | Total Usages Across Datasets |
|---------------|------------------------------|
| formula_based | 198                          |
| model_based   | 26                           |
| llm_judge     | 87                           |

## Per-Dataset Summary

| Dataset | Input | Output | Has Ref? | Gen Config Source | # Metrics | Metric Types |
|---------|-------|--------|----------|-------------------|-----------|--------------|
| aaar | text | text | yes | defaults | 5 | model_based, llm_judge |
| aidanbench | text | text | no | metric_notes | 3 | formula_based, llm_judge, model_based |
| alpaca_eval_2 | text | text | yes | defaults | 1 | llm_judge |
| amuse_chord_generation | text | text | no | metric_notes | 3 | formula_based |
| analobench | text | text | yes | defaults | 2 | formula_based |
| arastories | text | text | yes | defaults | 5 | llm_judge |
| arena_hard_creative | text | text | no | defaults | 1 | llm_judge |
| arena_hard_v01 | text | text | no | defaults | 1 | llm_judge |
| arn | text | text | yes | defaults | 1 | formula_based |
| artinsight | image, text | text | no | defaults | 1 | llm_judge |
| assocam | image, text | text | yes | defaults | 2 | formula_based |
| ava | image | text | yes | defaults | 3 | formula_based |
| balancecc_prompt_generation | text | text | yes | defaults | 2 | formula_based |
| balderdash | text | text | no | defaults | 1 | formula_based |
| banner_request_400 | image, text | image | no | defaults | 1 | llm_judge |
| bhp_hypothesis_generation | text | text | yes | defaults | 3 | formula_based, model_based |
| brainteaser | text | text | yes | defaults | 1 | formula_based |
| c3_crosstalk | text | text | yes | defaults | 2 | formula_based |
| calligrapher | image, text | image | no | defaults | 1 | llm_judge |
| cdat | text | text | no | defaults | 3 | model_based, formula_based |
| chinese_homophonic_puns | text | text | yes | defaults | 2 | formula_based |
| cii_bench | image, text | text | yes | defaults | 1 | formula_based |
| clef_joker_2025_task2 | text | text | yes | defaults | 2 | formula_based |
| cm3d | image, text | text | yes | defaults | 1 | formula_based |
| conceptual_design | text | text | yes | defaults | 2 | formula_based, llm_judge |
| convbench | image, text | text | yes | defaults | 1 | formula_based |
| cpers | text | text | yes | temp=1.0 | 3 | formula_based, llm_judge |
| creai_cps | text | text | yes | defaults | 3 | formula_based, llm_judge |
| creation_mmbench | image, text | text | yes | defaults | 1 | llm_judge |
| creative_pair | image, text | text | yes | defaults | 1 | formula_based |
| creative_process | text | text | no | defaults | 2 | llm_judge |
| creativemath | text | text | no | defaults | 1 | llm_judge |
| creatset | text | text | yes | defaults | 3 | formula_based, llm_judge |
| critics_story | text | text | no | defaults | 1 | llm_judge |
| crowd_vote | text | text | no | defaults | 1 | llm_judge |
| crowdcounter | text | text | yes | defaults | 2 | formula_based |
| cs4 | text | text | no | defaults | 1 | llm_judge |
| csd100 | image | image | yes | defaults | 2 | formula_based |
| cue_word_story | text | text | yes | defaults | 3 | formula_based, llm_judge |
| d_humor | image, text | text | yes | defaults | 1 | formula_based |
| dat | text | text | no | defaults | 1 | model_based |
| dat_creative_writing | text | text | no | defaults | 2 | model_based, llm_judge |
| data_narrative | text | text | yes | defaults | 3 | formula_based, model_based |
| deep_math | text | text | no | defaults | 1 | formula_based |
| dialogue_diversity | text | text | yes | defaults | 4 | formula_based, model_based, llm_judge |
| discovery_bench | text | text | yes | defaults | 2 | formula_based, llm_judge |
| diverse_not_short | text | text | yes | defaults | 4 | formula_based |
| dpt | text | text | no | defaults | 1 | llm_judge |
| eqbench_creative_writing_v3 | text | text | no | scenario file | 2 | llm_judge |
| esp_dataset | image, text | text | yes | defaults | 2 | formula_based, model_based |
| fann_or_flop | text | text | yes | defaults | 2 | formula_based |
| fig_qa | text | text | yes | defaults | 1 | formula_based |
| flute_filtered | text | text | yes | defaults | 2 | formula_based |
| fscg8 | image | image | no | defaults | 1 | llm_judge |
| funqa | video | text | yes | defaults | 2 | formula_based |
| future_ideas | text | text | no | defaults | 2 | llm_judge |
| futuregen | text | text | no | defaults | 2 | formula_based |
| fuxibench | text | text | no | defaults | 1 | llm_judge |
| gauss | text | text | yes | defaults | 3 | formula_based, llm_judge |
| geo_story | text | text | no | defaults | 3 | formula_based, llm_judge |
| grapheval_ai_researcher | text | text | no | defaults | 1 | llm_judge |
| grapheval_iclr | text | text | no | defaults | 1 | llm_judge |
| grapheval_review_advisor | text | text | no | defaults | 1 | llm_judge |
| graphrag_bench | text | text | no | defaults | 2 | formula_based |
| graphragbench-wrongone | text | text | yes | defaults | 1 | formula_based |
| historical_analogy | text | text | no | defaults | 1 | llm_judge |
| hummus | image, text | text | yes | defaults | 2 | formula_based, llm_judge |
| humor_transfer | text | text | yes | defaults | 2 | formula_based |
| humordb | image | text | yes | defaults | 1 | formula_based |
| hypobench | text | text | no | defaults | 3 | llm_judge |
| hypogen | text | text | no | defaults | 1 | llm_judge |
| idrbench | text | text | no | defaults | 2 | formula_based |
| ii_bench | image, text | text | yes | defaults | 1 | formula_based |
| infochartqa | image, text | text | yes | defaults | 2 | formula_based |
| irfl | image, text | text | yes | defaults | 1 | formula_based |
| javanese_sundanese_story_cloze | text | text | yes | defaults | 2 | formula_based |
| kiva | image | text | yes | defaults | 1 | formula_based |
| layoutsam_eval | text | text | yes | defaults | 1 | llm_judge |
| lcc_metaphor | text | text | yes | defaults | 2 | formula_based |
| litbench | text | text | no | defaults | 1 | llm_judge |
| liveideabench | text | text | no | defaults | 1 | llm_judge |
| llm4biohypogen | text | text | yes | defaults | 3 | formula_based, model_based |
| llm_discussion | text | text | no | defaults | 3 | formula_based |
| llm_review_focus | text | text | no | defaults | 1 | llm_judge |
| llm_srbench | text | text | yes | defaults | 2 | formula_based, llm_judge |
| macgyver | text | text | yes | defaults | 2 | formula_based, llm_judge |
| mars | image, text, graph | text | yes | defaults | 1 | formula_based |
| matdesign | text | text | no | defaults | 1 | llm_judge |
| memecap | image, text | text | yes | defaults | 3 | formula_based |
| met_meme | image, text | text | yes | defaults | 1 | formula_based |
| meta4xnli | text | text | yes | defaults | 1 | formula_based |
| metaphor_generation | text | text | yes | defaults | 3 | formula_based |
| metaphoric_analogies | text | text | yes | defaults | 2 | formula_based |
| mineanybuild | image, text | text | yes | defaults | 2 | formula_based, llm_judge |
| miqa | text | text | yes | defaults | 2 | formula_based |
| mixassist | text | text | yes | defaults | 2 | formula_based |
| moh_x | text | text | yes | defaults | 2 | formula_based |
| mops | text | text | yes | defaults | 4 | formula_based, llm_judge |
| munch | text | text | yes | defaults | 1 | formula_based |
| muse_perception | video, text | text | yes | defaults | 2 | formula_based |
| music_theory_bench | text | text | yes | defaults | 2 | formula_based |
| neocoder | text | text | yes | defaults | 2 | formula_based |
| newyorker_caption | image, text | text | yes | defaults | 1 | formula_based |
| newyorker_humor | text | text | yes | defaults | 2 | formula_based |
| noveltybench | text | text | no | metric_notes | 2 | model_based |
| nyt_connections | text | text | yes | defaults | 1 | formula_based |
| ocw | text | text | yes | defaults | 1 | formula_based |
| ocw_connections | text | text | yes | defaults | 3 | formula_based, model_based |
| oogiri_go | image, text | text | yes | defaults | 1 | formula_based |
| outline_to_story | text | text | yes | defaults | 3 | formula_based |
| pace | text | text | no | defaults | 2 | formula_based, model_based |
| permpst | text | text | yes | defaults | 2 | formula_based, llm_judge |
| poetmt | text | text | yes | defaults | 5 | formula_based, llm_judge |
| pollux_creativity | text | text | no | defaults | 2 | llm_judge |
| pron_vs_prompt | text | text | yes | defaults | 3 | llm_judge |
| proparalogy | text | text | yes | defaults | 1 | formula_based |
| protein_bench | text | text | no | defaults | 4 | formula_based, model_based |
| pun_eval | text | text | yes | defaults | 3 | formula_based, llm_judge |
| puntuguese | text | text | yes | defaults | 2 | formula_based |
| puzzleworld | image, text | text | yes | defaults | 1 | formula_based |
| rebus_puzzle | image, text | text | yes | defaults | 2 | formula_based, llm_judge |
| recombination_extraction | text | other | yes | defaults | 2 | formula_based |
| research_idea_execution | text | text | no | defaults | 3 | llm_judge |
| riddlesense | text | text | yes | defaults | 2 | formula_based |
| robotoolbench | text | text | no | defaults | 2 | formula_based |
| rpgbench | text | other | no | defaults | 2 | formula_based, llm_judge |
| scar | text | text | yes | defaults | 2 | formula_based |
| schnovel | text | text | yes | defaults | 1 | formula_based |
| science_analogies | text | text | yes | defaults | 2 | formula_based |
| scimon | text | text | yes | defaults | 3 | formula_based, model_based |
| sdat | text | text | no | defaults | 1 | model_based |
| showerthoughts | text | text | no | defaults | 3 | llm_judge |
| simile_generation | text | text | yes | defaults | 2 | formula_based |
| slang_generation | text | text | yes | defaults | 2 | llm_judge |
| sonnet_or_not_bot | text | text | yes | defaults | 1 | formula_based |
| speak_to_structure | text | text | yes | defaults | 2 | formula_based, llm_judge |
| splat | text | text | yes | defaults | 3 | formula_based, model_based |
| ss_gen | text | text | yes | defaults | 3 | formula_based, llm_judge |
| story_generation_rocstories | text | text | yes | defaults | 2 | formula_based |
| story_quality | text | text | yes | defaults | 2 | formula_based |
| storyer | text | text | yes | defaults | 3 | formula_based |
| sudoku_bench | text | text | yes | defaults | 1 | formula_based |
| textlogo3k | image, text | other | yes | defaults | 2 | formula_based |
| thenextchapter | text | text | yes | defaults | 2 | formula_based |
| tiger_bench | text | text | yes | defaults | 1 | formula_based |
| tinyfabulist | text | text | no | defaults | 2 | llm_judge |
| tinystories | text | text | no | defaults | 3 | llm_judge |
| ttcw | text | text | yes | defaults | 2 | formula_based |
| twistlist | text | text | yes | defaults | 3 | formula_based, model_based |
| unfun_corpus | text | text | yes | defaults | 2 | formula_based |
| v_flute | image, text | text | yes | defaults | 2 | formula_based, model_based |
| vflute | image, text | text | yes | defaults | 2 | formula_based, model_based |
| vgsg | image | text | yes | defaults | 2 | formula_based |
| vietnamese_poem | text | text | yes | defaults | 4 | formula_based |
| webnovelbench | text | text | no | defaults | 3 | llm_judge, formula_based |
| writingbench | text | text | no | scenario file | 1 | llm_judge |
| yesbut | image | text | yes | defaults | 3 | formula_based |
| yesbut_v2 | image | text | yes | defaults | 3 | formula_based |

## Fields Set to Null (Gaps)

| Dataset | Field | Reason |
|---------|-------|--------|
| aaar | judge_prompt | not published in paper/annotator notes |
| aidanbench | source_paper | OpenReview URL (not arxiv) |
| alpaca_eval_2 | judge_prompt | proprietary annotator prompt not in metric notes |
| amuse_chord_generation | has_reference_target | chord gen is open-ended, no fixed reference |
| arn | source_repo | no public GitHub found |
| artinsight | has_reference_target | rubric-only evaluation, no text references |
| artinsight | judge_prompt | full rubric not extracted from annotator notes |
| balderdash | has_reference_target | multi-agent game, no single gold reference |
| banner_request_400 | has_reference_target | banner design is open-ended |
| bhp_hypothesis_generation | hypothesis_judge_prompt | not specified in metric notes |
| calligrapher | source_paper | arXiv paper from 2506 (recent) |
| cm3d | source_repo | not found in scenario file |
| cpers | source_repo | not found in scenario file |
| creative_pair | source_repo | not found in scenario file |
| creativemath | has_reference_target | solutions withheld; only invalid refs provided |
| crowd_vote | source_repo | not found in scenario file |
| csd100 | source_repo | not found in scenario file |
| d_humor | source_repo | not found in scenario file |
| deep_math | has_reference_target | evaluates process not final answer |
| eqbench_creative_writing_v3 | source_paper | benchmark-only, no dedicated paper |
| esp_dataset | source_paper | CVPR 2023 paper URL not extracted |
| fscg8 | source_repo | not found in scenario file |
| funqa | source_repo | not found in scenario file |
| fuxibench | judge_prompt | not specified in metric notes |
| gauss | source_repo | not found in scenario file |
| geo_story | has_reference_target | geography-influenced story; no fixed target |
| graphrag_bench | source_repo | not found in scenario file |
| hummus | source_paper | paper from arXiv 2504 |
| hypogen | source_repo | not found in scenario file |
| idrbench | source_repo | not found in scenario file |
| infochartqa | source_paper | no arxiv/doi paper found |
| layoutsam_eval | source_repo | not found in scenario file |
| lcc_metaphor | source_paper | ACL Anthology URL not arxiv |
| litbench | judge_prompt | not specified in metric notes |
| liveideabench | judge_prompt | not specified in metric notes |
| llm_review_focus | source_repo | not found in scenario file |
| matdesign | source_repo | not found in scenario file |
| met_meme | source_paper | SIGIR 2022 paper — no arxiv |
| muse_perception | has_reference_target | EULA-gated dataset |
| pace | source_paper | no associated paper found |
| protein_bench | source_repo | ProteinBench website only |
| rebus_puzzle | source_paper | CVPR paper not specified |
| robotoolbench | source_repo | no GitHub in scenario |
| rpgbench | has_reference_target | creative generation evaluated by judge |
| sdat | source_repo | online tool, no GitHub |
| slang_generation | source_paper | arXiv placeholder in scenario |
| webnovelbench | has_reference_target | percentile vs 4000 human novels |
