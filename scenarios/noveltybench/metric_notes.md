# Evaluation Metrics: NoveltyBench

Source: https://github.com/novelty-bench/novelty-bench
Paper: "NoveltyBench: Evaluating Language Models for Humanlike Diversity" (Zhang et al., 2025)
       https://arxiv.org/abs/2504.05228

## Task Overview

Models generate **multiple outputs** (typically 5-10) for each prompt. NoveltyBench
evaluates both:
1. **Diversity**: How different the outputs are from each other
2. **Quality**: How good each output is

This dual evaluation captures the tension between diversity and quality - models
that generate very different outputs may sacrifice quality, while high-quality
models may lack diversity (mode collapse).

## Required Custom Metrics

### 1. Diversity Metric (Primary)

**Classifier**: deberta-v3-large fine-tuned for binary functional equivalence

**Model**: `deberta-v3-large-generation-similarity`
- Trained on 1,000 human-annotated pairs from NoveltyBench
- Binary prediction: Are two generations functionally equivalent?

**Calculation**:
```
For each prompt:
  1. Generate N outputs (e.g., N=10)
  2. For each pair of outputs (i, j):
     - Classify as equivalent (0) or distinct (1)
  3. Diversity score = % of pairs classified as distinct
```

**Example**:
- 10 outputs → 45 pairs (combinations)
- 30 pairs classified as distinct
- Diversity = 30/45 = 66.7%

**Implementation**:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "deberta-v3-large-generation-similarity"  # hypothetical
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def compute_diversity(outputs):
    distinct_count = 0
    total_pairs = 0

    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            # Classify pair
            inputs = tokenizer(outputs[i], outputs[j], return_tensors="pt")
            logits = model(**inputs).logits
            prediction = logits.argmax(dim=-1).item()

            distinct_count += prediction  # 1 if distinct, 0 if equivalent
            total_pairs += 1

    return distinct_count / total_pairs if total_pairs > 0 else 0
```

### 2. Quality Metric (Secondary)

**Model**: Skywork/Skywork-Reward-Gemma-2-27B-v0.2

**Purpose**: Assess individual output quality (helpfulness, correctness, coherence)

**Calculation**:
```
For each output:
  1. Pass through reward model
  2. Get quality score (typically 0-1 or logit)
  3. Average across all N outputs
```

**Note from DARLING paper**: The reward model is vulnerable to reward hacking,
so the paper primarily uses the distinct classifier (trained on human annotations)
for evaluation.

## Evaluation Pipeline

1. **Generate**: Model produces N outputs per prompt (typically N=5-10)
   - Use temperature > 0 (e.g., 0.7-1.0) for diversity
   - Optionally vary other params (top_p, top_k, seed)

2. **Diversity**: Compute pairwise distinctness using classifier
   - All pairs compared
   - Fraction of distinct pairs = diversity score

3. **Quality**: Average reward model scores
   - Each output scored individually
   - Mean quality = average across N outputs

4. **Combined**: Report diversity, quality, and optionally Pareto frontier
   - Higher diversity at same quality = better
   - Trade-off curve shows model's diversity-quality balance

## Metrics Summary

| Metric | Model/Method | Output | Interpretation |
|--------|--------------|--------|----------------|
| Diversity | deberta-v3-large classifier | 0-100% | % of output pairs that are distinct |
| Quality | Skywork-Reward-Gemma-2-27B | 0-1 score | Average quality across outputs |
| Human-Level | Baseline comparison | - | Humans: ~85% diversity on average |

## Key Findings (from Paper)

- **State-of-the-art models**: Significantly less diverse than humans
- **Larger models**: Often less diverse than smaller ones (counterintuitive!)
- **Quality vs. Diversity**: No inherent trade-off - some models achieve both
- **Standard benchmarks**: Don't predict diversity performance

## Datasets

### NB-Curated (100 prompts)
Four categories:
1. **Randomness**: "Generate a random number between 1-100"
2. **Factual knowledge**: "Name 5 European capitals"
3. **Creative writing**: "Tell me a story about a girl and her dog"
4. **Subjectivity**: "What's your favorite color and why?"

### NB-WildChat (1,000 prompts)
- Real user interactions from ChatGPT (Zhao et al., 2024)
- Natural distribution of user requests
- More realistic evaluation setting

## Implementation Requirements

### Dependencies
- `transformers`: For deberta-v3-large classifier
- `torch`: PyTorch backend
- Model weights: HuggingFace Hub (deberta-v3-large fine-tuned version)

### Generation Settings
- **N outputs**: 5-10 per prompt (paper uses 10)
- **Temperature**: 0.7-1.0 (higher = more diverse, lower = mode collapse)
- **Sampling**: Use sampling methods (top-p, top-k) not greedy decoding

### Classifier Training Data
- 1,000 annotated pairs from NoveltyBench
- 1,000 pairs used for training
- Binary labels: functionally equivalent (0) or distinct (1)
- Inter-annotator agreement: High (paper reports specifics)

## Notes for HELM Integration

- **RunSpec**: Requires custom metric implementation (diversity classifier)
- **Multi-generation**: Models generate N times per instance (not standard HELM pattern)
- **Sampling**: Must use non-deterministic generation (temperature > 0)
- **Baseline**: Include human diversity scores for comparison
- **Classifier**: Need to host deberta-v3-large fine-tuned model or use API
- **Reward model**: Optional (DARLING paper notes it's vulnerable to hacking)

## References

- Paper: https://arxiv.org/abs/2504.05228
- Code: https://github.com/novelty-bench/novelty-bench
- Website: https://novelty-bench.github.io/
- DARLING usage: arXiv:2509.02534 (diversity evaluation)
- WildChat dataset: Zhao et al., 2024
