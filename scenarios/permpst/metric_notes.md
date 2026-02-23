# Metric Requirements: PerMPST (Personalized Movie Plot Evaluation)

Source: Paper Section 4 (Experiments) and Section 5 (Results)
Paper: https://arxiv.org/abs/2310.03304

## Overview

PerMPST evaluates personalized story assessment through a regression task.
Models predict reviewer-specific scores (1-10 scale) for movie plots based on
the reviewer's historical ratings and review text, testing the ability to learn
and adapt to individual preferences.

## Ground Truth

Each validation example includes:
- **Historical reviews**: 0-5 previous movie reviews from the same reviewer
- **Historical scores**: Corresponding 1-10 ratings
- **Target plot**: New movie plot to evaluate
- **Ground truth score**: Reviewer's actual rating (1-10)
- **Ground truth review**: Reviewer's actual review text

Example:
```json
{
  "Review": "Classic Suburban Black comedy!!! ||| Superior black comedy...",
  "Score": 8
}
```

## Correlation Metrics

The paper uses correlation coefficients to measure agreement between predicted
and ground truth scores, which is appropriate for evaluating personalized
preference learning on ordinal scales.

### 1. Pearson Correlation (r)

Measures linear relationship between predicted and actual scores.

**Formula**: `r = cov(predicted, actual) / (std(predicted) * std(actual))`

**Interpretation**:
- Range: -1 to +1
- r > 0.7: Strong positive correlation (good personalization)
- r = 0: No linear correlation
- Higher is better

**Use case**: Evaluates if model captures the general rating tendencies

### 2. Spearman Rank Correlation (ρ)

Measures monotonic relationship using rank ordering.

**Formula**: Pearson correlation of rank-transformed scores

**Interpretation**:
- Range: -1 to +1
- More robust to outliers than Pearson
- ρ > 0.7: Strong monotonic relationship
- Evaluates if model correctly orders plot preferences

**Use case**: Tests if model understands relative preferences

### 3. Kendall-Tau Correlation (τ)

Measures ordinal association between rankings.

**Formula**: `τ = (concordant_pairs - discordant_pairs) / total_pairs`

**Interpretation**:
- Range: -1 to +1
- More conservative than Spearman
- τ > 0.5: Good ordinal agreement
- Robust to ties and small sample sizes

**Use case**: Evaluates pairwise ranking agreement

## Implementation Notes

### For HELM Integration

1. **Parse Model Output**: Extract score from JSON response
   ```python
   import json
   import re

   def extract_score(response):
       # Remove code block markers
       text = re.sub(r'```(?:json)?', '', response).strip()
       try:
           data = json.loads(text)
           score = float(data.get('Score', 5.0))
           return max(1.0, min(10.0, score))  # Clamp to 1-10
       except:
           return 5.0  # Default to middle score
   ```

2. **Compute Metrics**: Calculate three correlation coefficients
   ```python
   from scipy.stats import pearsonr, spearmanr, kendalltau

   predicted = [extract_score(r.text) for r in responses]
   ground_truth = [float(ref.output.text) for ref in references]

   pearson_r, _ = pearsonr(predicted, ground_truth)
   spearman_rho, _ = spearmanr(predicted, ground_truth)
   kendall_tau, _ = kendalltau(predicted, ground_truth)
   ```

3. **Aggregate Results**: Report mean across all reviewers
   - Overall correlation (all 915 validation examples)
   - Per-reviewer correlation (92 unique reviewers)
   - Stratified by k (number of historical reviews)

### Example Usage

```python
from helm.benchmark.metrics.regression_metrics import (
    pearson_correlation,
    spearman_correlation,
    kendall_tau_correlation
)

# Extract scores from model outputs
predicted_scores = []
for response in model_responses:
    score = extract_score_from_json(response.text)
    predicted_scores.append(score)

# Get ground truth scores
ground_truth_scores = [float(ref.output.text) for ref in references]

# Compute correlation metrics
pearson_r = pearson_correlation(predicted_scores, ground_truth_scores)
spearman_rho = spearman_correlation(predicted_scores, ground_truth_scores)
kendall_tau = kendall_tau_correlation(predicted_scores, ground_truth_scores)

print(f"Pearson r: {pearson_r:.3f}")
print(f"Spearman ρ: {spearman_rho:.3f}")
print(f"Kendall τ: {kendall_tau:.3f}")
```

## Baseline Performance

Expected performance ranges (from paper Table 2):

| Model | k | Pearson r | Spearman ρ | Kendall τ |
|-------|---|-----------|------------|-----------|
| GPT-3.5 | 1 | 0.45 | 0.43 | 0.32 |
| GPT-4 | 1 | 0.52 | 0.49 | 0.37 |
| PerSE (LLaMA-2 fine-tuned) | 1 | **0.58** | **0.54** | **0.41** |

**Notes:**
- Performance improves with more context (k=2-5 gives higher correlations)
- Fine-tuned models (PerSE) outperform general LLMs
- All correlations are moderate, indicating personalization is challenging

## Additional Analysis

### Stratified Evaluation

Break down performance by:

1. **Number of historical reviews (k)**:
   - k=0: Cold start (no personalization)
   - k=1: Minimal context
   - k=5: Rich context

2. **Reviewer activity level**:
   - Low-activity reviewers (<10 historical reviews total)
   - Medium-activity reviewers (10-50 reviews)
   - High-activity reviewers (>50 reviews)

3. **Score distribution**:
   - Harsh reviewers (mean score <5)
   - Average reviewers (mean score 5-7)
   - Lenient reviewers (mean score >7)

### Error Analysis

Examine cases with large prediction errors (`|predicted - actual| > 3`):
- Are certain reviewers harder to personalize?
- Do specific genres or plot types cause issues?
- Is the model biased toward middle scores (regression to mean)?

## Alternative Evaluation Approaches

If correlation metrics are insufficient:

1. **Mean Absolute Error (MAE)**: Direct score difference
   - MAE = mean(|predicted - actual|)
   - Complements correlation (measures calibration)

2. **Classification Accuracy**: Binned scores
   - Convert to categories: Low (1-4), Medium (5-7), High (8-10)
   - Report 3-way classification accuracy

3. **Personalization Gain**: Improvement over baseline
   - Compare against non-personalized average score per reviewer
   - Measure lift from personalization

## Quality Control

**Recommended checks:**
1. Ensure model outputs valid JSON with "Score" field
2. Verify scores are in 1-10 range (flag out-of-range predictions)
3. Check for degenerate solutions (all predictions same score)
4. Compare score distribution: predicted vs ground truth

## Notes

- This is a **personalized evaluation task**, not story generation
- Focus is on learning individual preferences, not universal story quality
- Reviewer writing style consistency is NOT evaluated (only scores)
- 915 validation examples across 92 unique reviewers (~10 examples per reviewer)
- Paper emphasizes interpretability: PerSE explains *why* a reviewer would rate something

## References

- Original paper: Rao et al., "Learning Personalized Alignment for Evaluating Open-ended Text Generation", EMNLP 2023
- Dataset: PerMPST (Personalized Movie Plot Synopsis and Tags)
- Code: https://github.com/facebookresearch/perse
