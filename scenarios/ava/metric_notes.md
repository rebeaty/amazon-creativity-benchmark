# Metric Requirements: AVA Score Prediction

Source: Adapted from AVA (Aesthetic Visual Analysis) paper - CVPR 2012
Paper: https://refbase.cvc.uab.es/files/MMP2012a.pdf

## Overview

AVA Score Prediction evaluates aesthetic understanding through a regression task.
Models predict mean aesthetic scores (1-10 scale) for images, compared against
ground truth computed from ~200 human ratings per image.

## Ground Truth

Each image in AVA has:
- **Vote distribution**: Counts of ratings 1-10 (10 columns in AVA.txt)
- **Mean aesthetic score**: Weighted average computed from distribution
- **Standard deviation**: Measure of rater agreement

Example:
```
Image ID: 12345
Votes: [2, 5, 15, 30, 45, 50, 35, 15, 8, 5]  # Counts for ratings 1-10
Mean score: 5.67
Std dev: 1.82
```

## Regression Metrics

### 1. Mean Absolute Error (MAE)

Measures average absolute difference between predicted and ground truth scores.

**Formula**: `MAE = (1/n) * Σ|predicted - actual|`

**Interpretation**:
- Lower is better
- MAE = 0.5 means predictions are off by 0.5 points on average
- Good baseline: MAE < 1.0 (within 1 point of ground truth)

### 2. Mean Squared Error (MSE) / Root MSE (RMSE)

Measures squared differences, penalizing larger errors more heavily.

**Formula**: `MSE = (1/n) * Σ(predicted - actual)²`

**Interpretation**:
- Lower is better
- RMSE gives error in same units as scores (1-10 scale)
- Sensitive to outliers/large mistakes

### 3. Pearson Correlation

Measures linear correlation between predicted and actual scores.

**Formula**: `r = cov(predicted, actual) / (std(predicted) * std(actual))`

**Interpretation**:
- Range: -1 to +1
- r = 1: Perfect positive correlation
- r = 0: No correlation
- r > 0.7: Strong correlation (good aesthetic understanding)

### 4. Spearman Rank Correlation

Measures monotonic relationship using rank ordering.

**Formula**: Correlation of rank-ordered predictions and ground truth

**Interpretation**:
- Range: -1 to +1
- More robust to outliers than Pearson
- Evaluates if model correctly orders images by aesthetic quality

## Implementation Notes

### For HELM Integration

1. **Parse Model Output**: Extract numeric score from model response
   - Expected format: Single number "7.5" or "8"
   - Handle ranges: "7-8" → take midpoint
   - Handle invalid: "beautiful image" → flag as parse error

2. **Compute Metrics**: Calculate MAE, RMSE, Pearson r, Spearman ρ

3. **Aggregate Results**: Report mean and std across test set

### Example Usage

```python
from helm.benchmark.metrics.regression_metrics import (
    mean_absolute_error,
    mean_squared_error,
    pearson_correlation,
    spearman_correlation
)

# Extract scores
predicted_scores = [parse_score(response.text) for response in responses]
ground_truth_scores = [float(ref.output.text) for ref in references]

# Compute metrics
mae = mean_absolute_error(predicted_scores, ground_truth_scores)
rmse = sqrt(mean_squared_error(predicted_scores, ground_truth_scores))
pearson_r = pearson_correlation(predicted_scores, ground_truth_scores)
spearman_rho = spearman_correlation(predicted_scores, ground_truth_scores)
```

## Baseline Performance

Expected performance ranges (from aesthetic assessment literature):

| Model Type | MAE | RMSE | Pearson r |
|-----------|-----|------|-----------|
| Random | 2.5 | 3.0 | 0.0 |
| Simple CNN | 0.8-1.2 | 1.0-1.5 | 0.5-0.6 |
| NIMA (SOTA 2018) | 0.6-0.8 | 0.8-1.0 | 0.6-0.7 |
| LLMs (expected) | 0.8-1.5 | 1.0-2.0 | 0.4-0.6 |

**Note**: LLMs are not trained specifically for aesthetic prediction, so
performance may be lower than specialized vision models.

## Additional Analysis

### Stratified Evaluation

Break down performance by:
1. **Score ranges**: Low (1-4), Medium (4-7), High (7-10)
2. **Image categories**: Landscapes, portraits, architecture (from semantic tags)
3. **Agreement level**: High-agreement vs controversial images (by std dev)

### Error Analysis

Examine cases where `|predicted - actual| > 2.0`:
- Are certain image types consistently over/under-rated?
- Does model have biases (e.g., prefers bright images)?
- Correlation with human disagreement (high std dev)?

## Alternative Evaluation Approaches

If regression is not suitable:

1. **Binary Classification**: Predict high (≥6) vs low (<6) aesthetic quality
2. **Ordinal Classification**: Predict rating buckets (1-3, 4-6, 7-10)
3. **Ranking**: Pairwise comparison of aesthetic quality
4. **Distribution Prediction**: Predict full vote distribution, not just mean

## References

- Original paper: Murray et al., "AVA: A large-scale database for aesthetic visual analysis", CVPR 2012
- NIMA: Talebi & Milanfar, "NIMA: Neural Image Assessment", IEEE TIP 2018
- Dataset info: https://github.com/imfing/ava_downloader
