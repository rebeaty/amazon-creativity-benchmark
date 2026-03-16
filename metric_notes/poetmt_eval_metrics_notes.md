# Metric Requirements: PoetMT

Source: Paper Section 4.2 (Evaluation Methodology)
Paper: https://arxiv.org/abs/2408.09945

## Overview

PoetMT evaluates classical Chinese poetry translation across three dimensions inspired by traditional translation theory (信达雅):

1. **Adequacy (信)**: Accuracy of meaning and cultural faithfulness
2. **Fluency (达)**: Smoothness of rhythm and structural alignment
3. **Elegance (雅)**: Poetic beauty and aesthetic depth

## Standard Metrics (Already in HELM)

Use HELM's `get_open_ended_generation_metric_specs()` which includes:
- **BLEU-1, BLEU-4**: N-gram overlap with reference
- **ROUGE-L**: Longest common subsequence
- **F1**: Token-level precision and recall

## Custom GPT-4 Metrics (Require Implementation)

The paper proposes three GPT-4-based metrics to evaluate the creative "elegance" dimension:

### 1. Beauty of Sound (BS)
Evaluates the acoustic and rhythmic qualities of the translation.

**Criteria:**
- Rhyme scheme preservation
- Rhythm and meter
- Phonetic harmony
- Sound symbolism

### 2. Beauty of Form (BF)
Evaluates the structural and visual aesthetics.

**Criteria:**
- Line length consistency
- Stanza structure
- Visual poetry effects
- Parallelism and symmetry

### 3. Beauty of Meaning (BM)
Evaluates semantic depth and artistic conception.

**Criteria:**
- Imagery preservation
- Emotional resonance
- Cultural allusions
- Metaphorical depth
- Aesthetic interpretation

## Implementation Notes

### GPT-4 Evaluation Approach

The paper uses GPT-4 to score translations on the three beauty dimensions. The evaluation is:

1. **Sentence-level** for adequacy
2. **Discourse-level** for fluency and elegance

### Proposed Prompt Template (from paper)

```
You are an expert in classical Chinese poetry translation. Evaluate the following English translation of a Chinese poem on [dimension].

Source poem (Chinese): {chinese_poem}
Translation: {model_translation}
Reference translation: {reference_translation}

Rate the translation's [dimension] on a scale of 1-5:
1 - Poor
2 - Below Average
3 - Average
4 - Good
5 - Excellent

Consider: [dimension-specific criteria]

Provide your rating as a single number.
```

### Baseline Comparison

The paper compares models against:
- **Human expert translations** (reference)
- **RAT (Retrieval-Augmented Translation)** method
- Standard MT models (ChatGPT, GPT-4, Claude, etc.)

## Human Evaluation Protocol

The paper also conducts human evaluation with:
- Expert judges familiar with classical Chinese poetry
- Rating on adequacy, fluency, and elegance
- Comparative ranking of translations

## Correlation with Automatic Metrics

The GPT-4 metrics show:
- **High correlation** with human judgments on elegance
- **Moderate correlation** on adequacy and fluency
- **Better than BLEU/ROUGE** for poetic quality assessment

## Implementation Priority

For HELM integration:

1. **Phase 1** (Immediate): Use standard open-ended metrics (BLEU, ROUGE, F1)
2. **Phase 2** (Future): Implement GPT-4-based beauty metrics as custom Annotator
3. **Phase 3** (Advanced): Add RAT baseline for comparison

## References

- Paper: "Large Language Models for Classical Chinese Poetry Translation: Benchmarking, Evaluating, and Improving" (EMNLP 2025)
- Authors: Andong Chen et al., Harbin Institute of Technology
- Dataset: 758 sentence-level translations (790 poems in full dataset)
