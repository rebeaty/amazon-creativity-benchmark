# Metric Requirements: CREAI Creative Problem-Solving

Source: CREAI-item-generation repository, RLPS_RoBERTa.py, config.py

## Original Evaluation Approach

The CREAI framework evaluates creative responses using **two proprietary RoBERTa-based scoring models**:

### 1. Originality Scoring Model
- **Location**: `itemResponseOriginalityModelDir` in config
- **Model**: Fine-tuned RoBERTa
- **Output**: Originality score (float, can be negative)
- **Training**: Based on the RLPS (Remote Learning Problem Solving) dataset
- **Availability**: ❌ Pre-trained weights NOT publicly available
  - README states: "the authors cannot provide the pre-trained weights"
  - Must contact original RLPS paper authors to obtain

### 2. Quality Scoring Model
- **Location**: `itemResponseQualityModelDir` in config
- **Model**: Fine-tuned RoBERTa
- **Output**: Quality score (float, 0-1 range)
- **Training**: Based on the RLPS dataset
- **Availability**: ❌ Pre-trained weights NOT publicly available

## Example Scores from Dataset

From the processed data:
- Originality scores range: approximately -2.0 to +2.0
- Quality scores range: approximately 0.0 to 1.0
- Example: Scenario 0 response scored -0.22 (originality), 0.81 (quality)

## Alternative Evaluation Approaches

Since the proprietary models are unavailable, alternative metrics for creativity evaluation:

### Option 1: Standard Open-Ended Metrics (Current Default)
- **BLEU-1, BLEU-4**: N-gram overlap with reference
- **ROUGE-L**: Longest common subsequence
- **F1**: Token-level overlap
- **Limitation**: These measure similarity, not creativity

### Option 2: LLM-as-Judge for Creativity
While the paper doesn't specify exact judge prompts for **responses**, the framework does define evaluation criteria for **scenarios** (Prompts.py lines 8-56):
- Complexity (number of unique demands)
- Difficulty (competing demands)
- Accessibility (no specialized knowledge needed)
- Controversial (avoid harmful topics)

A similar rubric could be adapted for evaluating **response creativity**:
- **Novelty**: How original/unexpected is the solution?
- **Appropriateness**: Does it address the scenario demands?
- **Effectiveness**: How well does it resolve competing demands?
- **Practicality**: Is the solution feasible?

### Option 3: Custom Metric Implementation
To replicate the original evaluation:
1. Obtain RLPS dataset and labels
2. Fine-tune RoBERTa models for originality and quality
3. Implement as HELM metric classes
4. Requires significant effort and access to training data

## Recommendation

For initial HELM integration, use **Option 1** (standard open-ended metrics) as baseline.

For proper creativity evaluation, pursue **Option 2** (LLM-as-judge) with a carefully designed rubric based on creativity research principles:
- Divergent thinking
- Fluency, flexibility, originality
- Problem-solving effectiveness

Future work should explore **Option 3** if the RLPS scoring model weights become available or can be retrained.

## References

- CREAI paper: Laverghetta, A., Luchini, S., Linell, A., Reiter-Palmon, R., & Beaty, R. (2024). The creative psychometric item generator: a framework for item generation and validation using large language models. CREAI 2024: International Workshop on Artificial Intelligence and Creativity.
- RLPS originality scoring: Contact original paper authors for model weights
- Code: `RLPS_RoBERTa.py` in the CREAI repository (modified with PEFT)
