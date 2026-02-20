# Evaluation Notes: D-HUMOR

**Paper**: D-HUMOR: Dark Humor Understanding via Multimodal Open-ended Reasoning
**Source**: IEEE ICDM 2025, https://arxiv.org/abs/2509.06771
**Dataset**: UVSKKR/D-Humor (HuggingFace, gated access)

## Evaluation Type

D-HUMOR uses **standard classification metrics**, not LLM-as-judge evaluation.

All three tasks are evaluated using:
- **Primary metric**: Accuracy (exact match)
- **Additional metrics**: F1-score (macro/weighted), Precision, Recall

## Task-Specific Evaluation

### Task 1: Dark Humor Detection
- **Type**: Binary classification
- **Classes**: Yes (1), No (0)
- **Metrics**:
  - Accuracy
  - F1-score (binary)
  - Precision/Recall for positive class

### Task 2: Target Identification
- **Type**: 6-class classification
- **Classes**:
  - 0: Gender/Sex-Related Topics
  - 1: Mental Health
  - 2: Disability
  - 3: Race/Ethnicity
  - 4: Violence/Death
  - 5: Other
- **Metrics**:
  - Accuracy
  - Macro F1-score (average across all 6 classes)
  - Weighted F1-score (weighted by class frequency)
  - Per-class precision/recall

### Task 3: Intensity Classification
- **Type**: 3-class classification (ordinal)
- **Classes**:
  - 1: Mild
  - 2: Moderate
  - 3: Severe
- **Metrics**:
  - Accuracy
  - Macro F1-score
  - Weighted F1-score
  - Per-class precision/recall

## Dataset Statistics

- **Total instances**: 4,379 Reddit memes
- **Modalities**: Image + Text (OCR-extracted)
- **Splits**: The dataset provides train/test splits (exact distribution TBD upon access)

## Paper's Evaluation Setup

The paper evaluates their proposed method (TCRNet - Tri-stream Cross-Reasoning Network) against several baselines:

### Baseline Models Compared
1. **Text-only models**: BERT-based classifiers
2. **Vision-only models**: ViT-based classifiers
3. **Multimodal models**: VisualBERT, CLIP-based models
4. **Reasoning-augmented**: Their proposed TCRNet with Role-Reversal Self-Loop prompting

### Reported Performance (from paper)
The paper reports that TCRNet outperforms baselines across all three tasks, demonstrating the value of:
- Multimodal fusion (text + image)
- Reasoning augmentation (explanation generation via VLM)
- Cross-attention mechanisms between modalities

## HELM Integration

### RunSpec Configuration

For exact match evaluation in HELM:

```python
from helm.benchmark.run_specs import RunSpec, get_exact_match_metric_specs

# Task 1: Detection
RunSpec(
    name="d_humor:detection",
    scenario_spec=ScenarioSpec(
        class_name="scenarios.d_humor.scenario.DHumorDetectionScenario"
    ),
    metric_specs=get_exact_match_metric_specs()
)

# Task 2: Target
RunSpec(
    name="d_humor:target",
    scenario_spec=ScenarioSpec(
        class_name="scenarios.d_humor.scenario.DHumorTargetScenario"
    ),
    metric_specs=get_exact_match_metric_specs()
)

# Task 3: Intensity
RunSpec(
    name="d_humor:intensity",
    scenario_spec=ScenarioSpec(
        class_name="scenarios.d_humor.scenario.DHumorIntensityScenario"
    ),
    metric_specs=get_exact_match_metric_specs()
)
```

### Expected Model Output Format

For all three tasks, models should output:
- **Binary (Detection)**: "Yes" or "No"
- **6-class (Target)**: "A", "B", "C", "D", "E", or "F"
- **3-class (Intensity)**: "A", "B", or "C"

The scenario implementations use standard HELM Reference patterns with `CORRECT_TAG` for the correct answer.

## Additional Considerations

### Class Imbalance
The dataset may have class imbalance (especially for rare target categories). Consider:
- Using weighted F1-score alongside accuracy
- Reporting per-class performance for target identification
- Analyzing performance on underrepresented classes

### Multimodal Evaluation
Unlike text-only benchmarks:
- Models must process both image and text inputs
- Performance depends on vision-language model capabilities
- Text-only models will only see the OCR-extracted text

### Ethical Considerations
This benchmark contains sensitive content (dark humor about marginalized groups):
- Use only for research purposes
- Results should be interpreted in context of content moderation/safety applications
- Not appropriate for production deployment without careful consideration

## Notes

- **Dataset Access**: Gated - requires form submission and approval
- **Languages**: English (Reddit memes)
- **Original Purpose**: Proposed as part of a reasoning-augmented framework paper
- **HELM Adaptation**: Treats as standard multimodal classification tasks without the reasoning augmentation component (which was part of the paper's proposed method, not the benchmark itself)
