## Vietnamese Poem Quality Scoring System - Annotator Notes

## Overview

This benchmark evaluates LLM ability to generate Vietnamese poems conforming to traditional Vietnamese poetry genres with strict structural, tonal, and rhyme rules.

## Vietnamese Poetry Genres

### 1. Lục Bát (Six-Eight)
- **Structure**: Alternating 6-syllable and 8-syllable lines
- **Rhyme**: Complex rhyme scheme between lines
- **Most common**: Highest sample size in training data (87,609 poems)
- **Example**:
  ```
  oằn vai gánh nặng cuộc đời    (6 syllables)
  dẫu bao gian khổ đầy vơi nỗi niềm    (8 syllables)
  ```

### 2. Tứ Tuyệt (Four Characters - 4 chữ)
- **Structure**: 4 syllables per line
- **Lines per stanza**: Typically 4 lines
- **Rules**: Strict tonal and rhyme patterns

### 3. Ngũ Ngôn (Five Characters - 5 chữ)
- **Structure**: 5 syllables per line
- **Origin**: Influenced by Chinese poetry
- **Rules**: Tonal patterns at specific positions

### 4. Thất Ngôn (Seven Characters - 7 chữ)
- **Structure**: 7 syllables per line
- **Rules**: More complex tonal patterns

### 5. Bát Ngôn (Eight Characters - 8 chữ)
- **Structure**: 8 syllables per line
- **Challenge**: Longer lines increase difficulty

## Evaluation Methodology

### Automatic Rule-Based Scoring

**Formula**: `score = L/10 + 3T/10 + 6R/10`

Where:
- **L (Length)**: Correct syllable count per line (weight: 10%)
- **T (Tone)**: Correct tonal patterns (weight: 30%)
- **R (Rhyme)**: Correct rhyme patterns (weight: 60%)

### L: Length Score
- Checks if each line has correct number of syllables for the genre
- For lục bát: alternating 6 and 8 syllables
- For n-chữ genres: n syllables per line

### T: Tone Score
Vietnamese has 6 tones divided into two categories:
- **Even tones (bằng)**: không dấu (no tone mark), huyền (`), nặng (.)
- **Uneven tones (trắc)**: sắc (ˊ), hỏi (?), ngã (~)

Traditional poetry requires specific tonal patterns at key positions in each line.

### R: Rhyme Score
- Vietnamese rhyme based on ending vowel/consonant sounds
- Different genres have different rhyme schemes
- Lục bát: syllable 6 of first line rhymes with syllable 6 of second line;
           syllable 8 of second line rhymes with syllable 6 of next line
- Rhyme categories defined in `utils/rhymes.txt`

## Genre Classification

Before scoring, a BERT-based classifier (99.7% accuracy) determines the poem genre. This is used for:
- **Blind test**: When genre not specified in prompt
- **Validation**: Ensuring generated poem matches requested genre

## Implementation Details

### Vietnamese NLP Requirements

The evaluation requires Vietnamese-specific NLP tools:

1. **Syllable Counting**: Vietnamese is a syllabic language
2. **Tone Detection**: Extract tone marks from characters
3. **Rhyme Analysis**: Compare ending sounds using Vietnamese phonology
4. **Word Segmentation**: Vietnamese uses spaces between syllables, not words

### Scoring Code

The reference implementation is in the repository:
- `utils/check_rule.py`: Core scoring functions
- `utils/poem_classifier.py`: Genre classification
- `utils/rhymes.txt`: Vietnamese rhyme dictionary
- `utils/start_vowels.txt`: Vietnamese vowel/tone mappings

## Dataset Details

### Test Dataset
- **Size**: 480 instances
- **Format**: JSONL (one JSON object per line)
- **Fields**:
  - `prompt`: Vietnamese natural language instruction
  - `completion`: Reference Vietnamese poem

### Prompt Format
Example prompt (translated):
```
Viết bài thơ lục bát về đam mê làm cha mẹ, ước mơ cho gia đình
hạnh phúc và con cái thành nhân. Có chứa từ khóa "gia đình hạnh phúc",
"con cái thành nhân".

(Write a lục bát poem about passion for parenting, dreams for a happy
family and children becoming good people. Contains keywords "happy family",
"children becoming good people".)
```

### Training Dataset
Available at fsoft-ailab/Poem-Generator:
- **Total**: 171,188 Vietnamese poems
- **Genres**: lục bát (87,609), 5 chữ, 7 chữ, 8 chữ, 4 chữ
- **Source**: Historical Vietnamese poetry collections

## Model Performance Benchmarks

From the paper (text-to-poem task, lục bát genre):

| Model | Score |
|-------|-------|
| ChatGPT (zero-shot) | 0.440 |
| GPT-3 Davinci (1K samples) | 0.580 |
| BLOOM-7B (20K samples) | 0.678 |
| GPT-3 Babbage (20K samples) | 0.718 |
| GPT-3 Babbage (full) | 0.805 |

The Babbage model fine-tuned on full dataset achieves highest score, showing importance of Vietnamese-specific training.

## Key Challenges

1. **Language Barrier**: Requires Vietnamese language understanding
2. **Cultural Knowledge**: Understanding Vietnamese poetic themes and idioms
3. **Strict Rules**: Poetry rules are rigid and unforgiving
4. **Tone System**: Vietnamese tones critical to meaning and poetic flow
5. **Rhyme Complexity**: Vietnamese rhyme system differs from English
6. **Evaluation Complexity**: Requires specialized Vietnamese NLP tools

## Recommended HELM Evaluation Approach

### Primary Metric
- **Rule-Based Score**: Implement or wrap the reference scoring function
  - Weight: L=0.1, T=0.3, R=0.6
  - Report overall score and breakdown by component

### Secondary Analysis
- **By Genre**: Stratify results by poetry genre
  - Expect lục bát to score highest (most training data)
  - Other genres (4-8 chữ) likely more challenging

- **Genre Classification Accuracy**: Check if generated poems match requested genre

### Important Note
**Do not use BLEU or standard text similarity metrics** for this benchmark. Vietnamese poetry evaluation requires rule-based scoring that checks structural conformance, not lexical similarity to reference poems.

## Citation

```bibtex
@misc{huynh2024vietnamese,
  title={Vietnamese Poem Generation & The Prospect Of Cross-Language Poem-To-Poem Translation},
  author={Triet Minh Huynh and Quan Le Bao},
  year={2024},
  eprint={2401.01078},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Related Work

- **SP-GPT2** (arXiv:2110.15723): Semantics Improvement in Vietnamese Poetry Generation
  - First published Lục Bát dataset
  - Addressed semantic drift in GPT-2 generated poems
  - Evaluation method inspired this work

## External Resources

- Main repository: https://github.com/Anshler/poem_generator
- Original dataset: https://github.com/fsoft-ailab/Poem-Generator
- Demo: https://colab.research.google.com/drive/1Mw_MsCix-NeUGRu77E-BkkvW6tut-AI-
