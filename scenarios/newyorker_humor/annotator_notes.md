# New Yorker Cartoon Caption Contest Humor 'Understanding' Benchmarks - Annotator Notes

## Overview

This benchmark evaluates AI models' ability to understand sophisticated multimodal humor through three carefully designed tasks based on The New Yorker's famous Cartoon Caption Contest. The benchmark tests progressively more complex aspects of humor comprehension, from simple caption matching to explaining what makes a caption funny.

**Key Achievement**: Winner of Best Paper Award at ACL 2023 (one of three awardees).

## Task Format

### Task 1: Matching (Caption Selection)

**Format**: 5-way multiple choice

**Task**: Given a cartoon description and 5 caption candidates, select the correct/winning caption.

**Example**:
```
Cartoon Description:
Location: outdoors
Scene: A man stands at a crossroads with multiple directional signs pointing different ways
Notable: All the signs point to "nowhere" or absurd destinations
Entities: man, signpost, roads

Which caption best matches this cartoon?
A. I think I'm lost.
B. All roads lead to nowhere.
C. This is my retirement plan.
D. I should have used GPS.
E. The scenic route isn't so scenic.

Correct: B (captures the existential humor of the signpost absurdity)
```

### Task 2: Ranking (Funniness Comparison)

**Format**: 2-way comparison

**Task**: Given a cartoon description and 2 captions, identify which caption is funnier.

**Example**:
```
Cartoon Description:
Scene: A fish is sitting at a psychiatrist's office on a couch
Notable: The fish is clearly out of water, literally and figuratively

Which caption is funnier?
A. I feel like I'm drowning in my problems.
B. Doctor, I've been feeling out of my element lately.

Correct: B (clever wordplay on "out of my element" which works both literally and emotionally)
```

### Task 3: Explanation (Humor Interpretation)

**Format**: Open-ended generation

**Task**: Given a cartoon description and its winning caption, explain why the caption is funny.

**Example**:
```
Cartoon Description:
Scene: A medieval knight in full armor stands at an airport security checkpoint
Notable: The metal detector is beeping continuously

Caption: "I'm going to be here a while."

Explain why this caption is funny:

Reference: The humor comes from the incongruity of a medieval knight in a modern airport. Knights wear complete metal armor, which would obviously set off every metal detector. The caption's understated acknowledgment of this absurd situation creates comedic effect through the contrast between the extreme impracticality and the knight's casual acceptance.
```

## Dataset Structure

### Data Splits

Each task has its own split sizes:

| Task | Train | Validation | Test |
|------|-------|------------|------|
| Matching | 9,792 | 531 | 528 |
| Ranking | 9,576 | 507 | 513 |
| Explanation | 2,340 | 130 | 131 |

### Cross-Validation

The dataset supports 5-fold cross-validation:
- Configuration names: `matching`, `matching_1`, `matching_2`, `matching_3`, `matching_4`
- Same for `ranking` and `explanation`
- Use numbered versions for cross-validation experiments

### Data Fields

Each instance contains:
- **image**: The actual cartoon image (not used in text-only evaluation)
- **contest_number**: New Yorker contest ID
- **image_location**: Where the cartoon is set (e.g., "outdoors", "office")
- **image_description**: Main textual description of visual scene
- **image_uncanny_description**: Description focusing on unusual/humorous elements
- **entities**: List of objects/characters in the scene
- **questions**: Guiding questions about the scene
- **caption_choices**: List of caption options
- **label**: Correct answer (letter or text)
- **instance_id**: Unique identifier

### Text-Only Evaluation

**Important**: This implementation uses textual descriptions instead of cartoon images, enabling evaluation with text-only language models.

**Description Components**:
1. **Location**: General setting (indoor/outdoor, specific place)
2. **Scene**: Main visual description of what's happening
3. **Notable**: Uncanny/humorous elements (recommended for better performance)
4. **Entities**: Objects and characters present
5. **Questions**: Help guide understanding (e.g., "Why is the character in this location?")

## Evaluation Methodology

### Metrics

**Matching & Ranking**: Accuracy (percentage of correct selections)

**Explanation**: Multiple approaches possible:
1. **Human evaluation**: Preference comparison between model and reference explanations
2. **Automated metrics**: BLEU, ROUGE, BERTScore (with caveats)
3. **LLM-as-judge**: GPT-4 evaluation of explanation quality

### Performance Benchmarks

From the ACL 2023 paper:

#### Matching Task (5-way multiple choice)

| Model | Accuracy |
|-------|----------|
| Human performance | 94.0% |
| CLIP ViT-L/14 (fine-tuned) | 62.0% |
| GPT-3 (175B, few-shot) | ~55% |
| Random baseline | 20.0% |

**Gap**: Models fall ~30 accuracy points behind humans.

#### Ranking Task

| Model | Accuracy |
|-------|----------|
| Human performance | High (exact numbers vary) |
| Fine-tuned models | Moderate performance |
| Random baseline | 50.0% |

#### Explanation Task

| Comparison | Human Preference |
|------------|------------------|
| Human explanations vs GPT-4 (5-shot) | Humans preferred 66%+ of time |
| Human explanations vs fine-tuned models | Humans preferred even more often |

## Key Challenges

### 1. Sophisticated Humor Understanding

The New Yorker cartoons require:
- **Cultural knowledge**: References to modern life, professions, social situations
- **Incongruity detection**: Spotting what's absurd or unexpected
- **Wordplay comprehension**: Puns, double meanings, idioms
- **Situational irony**: Understanding implicit contradictions

### 2. Multimodal Reasoning (Even with Text)

Even with textual descriptions:
- Must integrate multiple description components
- Visualize the scene from text
- Connect visual and verbal elements
- Understand why specific visual-text combinations are humorous

### 3. Subjective Nature of Humor

- No single "correct" explanation exists
- Cultural and personal factors affect humor perception
- Context-dependent interpretation
- Subtle differences between good and great captions

### 4. Explanation Generation

Requires:
- Identifying the source of humor
- Articulating implicit connections
- Explaining incongruity without over-explaining
- Balancing detail and conciseness

## Recommended HELM Evaluation Approach

### Primary Evaluation

1. **Matching Task** (start here)
   - Use default configuration or cross-validation folds
   - Report accuracy on validation and test splits
   - Compare to human performance (94%) and random baseline (20%)

2. **Ranking Task** (if resources permit)
   - Simpler than matching (2-way vs 5-way)
   - Tests relative humor judgment
   - Compare to 50% random baseline

3. **Explanation Task** (most resource-intensive)
   - Requires qualitative evaluation
   - Consider using LLM-as-judge for automated assessment
   - Sample-based human evaluation recommended

### Analysis Recommendations

1. **Error Analysis by Contest**
   - Group by contest_number
   - Identify systematic failures
   - Compare model vs human preferences

2. **Caption Type Analysis**
   - Puns vs situational humor
   - Wordplay vs visual gags
   - Cultural references vs universal humor

3. **Performance by Description Richness**
   - Compare with vs without uncanny_description
   - Evaluate impact of entity information
   - Test with minimal vs full descriptions

### Prompting Strategy

**Zero-shot prompting**:
```
Cartoon Description:
[description]

Which caption best matches this cartoon?
[choices]

Answer:
```

**Few-shot prompting** (recommended for better performance):
- Include 2-3 examples with explanations
- Show how to reason about incongruity
- Demonstrate connection between scene and caption

**Chain-of-thought prompting** (for explanations):
- First describe what's unusual about the scene
- Then explain how the caption relates
- Finally state why this creates humor

## Comparison to Related Benchmarks

### vs. Other Humor Benchmarks

- **HaHackathon/Humicroedit**: Editing text to make it funny (not multimodal)
- **FunLines**: Movie subtitle humor (temporal, not visual)
- **ColBERT**: Satire detection (classification, not generation)
- **Difference**: New Yorker requires sophisticated visual-linguistic integration

### vs. Caption Generation

- **COCO Captions**: Descriptive captions (not humor)
- **Conceptual Captions**: Web images (not creative)
- **Difference**: New Yorker requires humor, not just description

### vs. Visual Commonsense

- **VCR**: Multiple choice about scenes (literal understanding)
- **VisPrag**: Pragmatic reasoning with images
- **Difference**: New Yorker requires humor interpretation, not just commonsense

## Implementation Notes

### Text Description Format

The scenario implementation concatenates multiple description fields:

```
Location: [image_location]
Scene: [image_description]
Notable: [image_uncanny_description]
Entities: [entities list]
Questions: [questions list]
```

The `Notable` field (uncanny description) is particularly valuable for humor understanding.

### Choice Format

- **Matching**: A, B, C, D, E (5 choices)
- **Ranking**: A, B (2 choices)
- **Explanation**: Free-form text generation

### Cross-Validation

To use cross-validation folds:
```python
scenario = NewYorkerHumorScenario(task="matching", cross_val_fold=1)
```

This loads the `matching_1` configuration with a different train/val/test split.

## Citation

```bibtex
@inproceedings{hessel-etal-2023-androids,
    title = "Do Androids Laugh at Electric Sheep? Humor ``Understanding'' Benchmarks from {T}he {N}ew {Y}orker Caption Contest",
    author = "Hessel, Jack  and
      Marasovi{\'c}, Ana  and
      Hwang, Jena D.  and
      Lee, Lillian  and
      Da, Jeff  and
      Zellers, Rowan  and
      Mankoff, Robert  and
      Choi, Yejin",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2209.06293",
    pages = "688--714",
}
```

## External Resources

- **Paper**: https://arxiv.org/abs/2209.06293
- **Dataset**: https://huggingface.co/datasets/jmhessel/newyorker_caption_contest
- **GitHub**: https://github.com/jmhessel/caption_contest_corpus
- **Website**: www.capcon.dev
- **New Yorker Contest**: https://www.newyorker.com/cartoons/contest

## Notes on Related Work

### The 2024 Large-Scale Dataset

A follow-up paper "Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning" (arXiv:2406.10522, NeurIPS 2024) presents:
- 365 contests (vs 650+ in this benchmark)
- 2.2M captions with 250M ratings
- Focus on caption **generation** rather than understanding
- Different evaluation: ranking model-generated captions

**This is a separate benchmark** implemented elsewhere in the codebase (yguooo/newyorker_caption_ranking dataset).

### ACL 2023 Best Paper Recognition

This benchmark was recognized as one of three best papers at ACL 2023, highlighting its significance for:
- Rigorous task design
- Careful data collection
- Thorough evaluation methodology
- Important research questions about AI humor understanding

The benchmark revealed a substantial gap between AI and human humor comprehension, establishing an important frontier for future research.
