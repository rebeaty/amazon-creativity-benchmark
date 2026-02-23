# Metric Requirements: V-FLUTE

Source: Paper Section 4, GitHub repository eval/ directory

## Evaluation Overview

V-FLUTE requires **dual evaluation**: both label prediction accuracy and explanation quality assessment.

## Primary Metric: F1@ExplanationScore

The paper's main evaluation metric is **F1 with explanation quality thresholds**, which penalizes correct label predictions when explanation quality is insufficient.

### ExplanationScore Calculation

**ExplanationScore** combines two semantic similarity metrics:

```python
ExplanationScore = (BERTScore_F1 + BLEURT) / 2
```

**Components:**
1. **BERTScore**: Contextual embedding-based similarity (F1 metric)
   - Uses `bert-base-uncased` model
   - Measures token-level semantic overlap between generated and gold explanations

2. **BLEURT**: Learned evaluation metric trained on human judgments
   - Uses `bleurt-large-512` checkpoint
   - Provides fluency and coherence assessment

### F1@Threshold Metrics

The paper reports multiple threshold levels:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| F1@0 | 0 | Standard F1 (no quality filtering) |
| F1@50 | 50 | Flip prediction if ExplanationScore ≤ 0.50 |
| F1@53 | 53 | Flip prediction if ExplanationScore ≤ 0.53 |
| F1@60 | 60 | Flip prediction if ExplanationScore ≤ 0.60 |
| F1@70 | 70 | Flip prediction if ExplanationScore ≤ 0.70 |
| F1@80 | 80 | Flip prediction if ExplanationScore ≤ 0.80 |
| F1@90 | 90 | Flip prediction if ExplanationScore ≤ 0.90 |

**Algorithm:**
```python
def compute_f1_at_threshold(predictions, gold_labels, explanations, gold_explanations, threshold):
    for i, (pred, expl) in enumerate(zip(predictions, explanations)):
        # Calculate explanation quality
        bertscore_f1 = compute_bertscore(expl, gold_explanations[i])
        bleurt_score = compute_bleurt(expl, gold_explanations[i])
        explanation_score = (bertscore_f1 + bleurt_score) / 2

        # Flip prediction if explanation quality below threshold
        if explanation_score <= threshold:
            predictions[i] = flip_label(pred)  # entailment ↔ contradiction

    return f1_score(gold_labels, predictions)
```

## Output Format Requirements

Models must generate outputs in this specific format:

```
[explanation text]
LABEL: [entailment or contradiction]
```

**Label Extraction Rules** (from `extract_label_and_expl.py`):

1. **Primary pattern**: Split on "LABEL:", "Label:", or "label:" marker
2. **Fallback keywords** if no marker:
   - **Entailment**: "entail", "supports the claim", "is consistent", "in harmony with", "is in agreement", "confirms the claim"
   - **Contradiction**: "contradict", "appears to contest"
   - **Neither**: "neither", "not possible to definitively label", "does not support or contradict"
3. **Post-processing**: Remove "Therefore," statements and clean whitespace

## Implementation Requirements

### Required Libraries

```python
# For BERTScore
from bert_score import score as bertscore_fn

# For BLEURT
from bleurt import score as bleurt_fn
import tensorflow as tf

# Model checkpoints
BERTSCORE_MODEL = "bert-base-uncased"
BLEURT_CHECKPOINT = "bleurt-large-512"
```

### Computing ExplanationScore

```python
def compute_explanation_score(generated_expl: str, gold_expl: str) -> float:
    """Compute combined BERTScore + BLEURT metric."""

    # BERTScore (F1 component)
    _, _, bertscore_f1 = bertscore_fn(
        [generated_expl],
        [gold_expl],
        lang="en",
        model_type=BERTSCORE_MODEL,
        verbose=False
    )
    bertscore_value = bertscore_f1.item()

    # BLEURT
    bleurt_scorer = bleurt_fn.BleurtScorer(BLEURT_CHECKPOINT)
    bleurt_value = bleurt_scorer.score(
        references=[gold_expl],
        candidates=[generated_expl]
    )[0]

    # Combined score
    explanation_score = (bertscore_value + bleurt_value) / 2

    return explanation_score
```

## Per-Phenomenon Breakdown

The paper reports results broken down by figurative phenomenon:

| Phenomenon | Test Instances | Source Datasets |
|------------|----------------|-----------------|
| Metaphor | ~200 | HAIVMet, IRFL |
| Simile | ~150 | IRFL |
| Idiom | ~75 | IRFL |
| Sarcasm | ~130 | MuSE |
| Humor | ~168 | MemeCap, NYCartoons |

**Total:** 723 test instances

For the IRFL subset specifically, the paper reports separate metrics for:
- **Idioms** (smaller subset)
- **Metaphor + Simile** (combined)

## Human Baseline Performance

From the paper:
- **Human F1@0**: 89.09
- Human annotators achieved high label accuracy with adequate explanations
- Sets upper bound for model performance

## Additional Evaluation Considerations

### Human Evaluation Categories

The paper's human evaluation categorized explanation errors as:
1. **Hallucination** - Incorrect facts about image content
2. **Unsound reasoning** - Logical errors in inference
3. **Incomplete reasoning** - Missing key elements
4. **Verbosity** - Overly long or redundant explanations

These categories are not part of automatic evaluation but provide qualitative insights.

### Label-Only Accuracy

Standard accuracy/F1 without explanation quality consideration (F1@0) is useful for:
- Comparing models without explanation capabilities
- Isolating visual entailment performance
- Establishing baseline performance

## Implementation in HELM

To implement this metric in HELM:

1. **Extract label and explanation** from model output using the parsing rules above
2. **Compute label F1** for F1@0 baseline
3. **Compute ExplanationScore** for each instance
4. **Apply thresholding** to compute F1@50, F1@53, F1@60, etc.
5. **Report all threshold levels** as separate metrics

The Scenario provides gold explanations as references (tagged with "gold_explanation") for metric computation.

## Reference Implementation

See the V-FLUTE repository for reference implementations:
- `eval/extract_label_and_expl.py` - Label extraction and parsing
- `eval/compute_metrics_bscore_bleurt.py` - ExplanationScore and F1@threshold computation
- `eval/run_eval.py` - Full evaluation pipeline

## Citation

```bibtex
@inproceedings{vflute2025,
  title={V-FLUTE: Visual Figurative Language Understanding with Textual Explanations},
  author={Akyan, Arkadiy Saakyan and others},
  booktitle={Proceedings of NAACL 2025},
  year={2025}
}
```
