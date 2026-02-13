# Metric Implementation Notes: S-DAT

## Overview

S-DAT (Synthetic-Divergent Association Task) requires a custom semantic distance metric not currently implemented in HELM. This metric measures divergent thinking by computing the average semantic dissimilarity between generated words.

## Scoring Method

### From S-DAT Paper (Haase et al., 2025)

1. **Extract words**: Parse model output to extract exactly 10 words
2. **Embed words**: Use `granite-embedding-278m-multilingual` model to convert each word to a vector
3. **Compute pairwise dissimilarity**: For each pair of words (i, j):
   ```
   dissimilarity(w_i, w_j) = 1 - cosine_similarity(embed(w_i), embed(w_j))
   ```
4. **Average across all pairs**: With 10 words, there are C(10,2) = 45 unique pairs:
   ```
   score = mean(dissimilarity) for all i < j
   ```
5. **Optional calibration**: Transform to match human DAT distribution (μ=78.5, σ=15.2)

### Alternative: Original DAT (Olson et al., 2021)

- Uses **GloVe 840B embeddings** (300-dimensional, English only)
- Same pairwise distance calculation
- Reference implementation: https://github.com/jayolson/divergent-association-task
- Simpler but less multilingual than S-DAT approach

## Implementation Requirements

### Dependencies

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import re

# For S-DAT (multilingual)
model = SentenceTransformer('ibm-granite/granite-embedding-278m-multilingual')

# Alternative: For original DAT (English only)
# import gensim.downloader as api
# model = api.load('glove-wiki-gigaword-300')
```

### Word Extraction Logic

The model output may vary in format. Need to handle:

```python
def extract_words(text: str) -> list[str]:
    """
    Extract 10 words from model output.

    Common formats:
    - Numbered list: "1. cat\n2. democracy\n..."
    - Comma-separated: "cat, democracy, mountain, ..."
    - Bullet points: "• cat\n• democracy\n..."
    """
    # Remove numbers, bullets, punctuation
    # Split by newlines or commas
    # Take first 10 valid words
    # Return list of exactly 10 words (pad or truncate if needed)
    pass
```

### Semantic Distance Calculation

```python
def compute_dat_score(words: list[str], model) -> float:
    """
    Compute DAT score as average pairwise semantic distance.

    Args:
        words: List of exactly 10 words
        model: Embedding model (Sentence Transformer or GloVe)

    Returns:
        Average dissimilarity score (0-1 range, higher = more divergent)
    """
    # Embed all words
    embeddings = [model.encode(word) for word in words]

    # Compute pairwise dissimilarities
    dissimilarities = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            # Cosine dissimilarity = 1 - cosine_similarity
            dissim = 1 - np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            dissimilarities.append(dissim)

    # Average across all 45 pairs
    return np.mean(dissimilarities)
```

### Optional: Calibration to Human Scale

```python
def calibrate_to_dat_scale(raw_score: float) -> float:
    """
    Transform raw score to match human DAT distribution.

    Human DAT statistics (Olson et al., 2021):
    - Mean: 78.5
    - Standard deviation: 15.2
    - Range: typically 40-120

    Args:
        raw_score: Average dissimilarity (0-1 range)

    Returns:
        Calibrated score matching human scale
    """
    # Linear transformation: score_calibrated = a * raw_score + b
    # Fit a, b using human validation data
    # For now, simple scaling: raw_score * 100 approximates DAT scale
    return raw_score * 100
```

## Edge Cases to Handle

1. **Incorrect word count**:
   - Model generates <10 words → Pad with random words or assign penalty score
   - Model generates >10 words → Take first 10 or sample randomly

2. **Invalid words**:
   - Non-English words → S-DAT model supports 11+ languages, so this is acceptable
   - Multi-word phrases → Split or reject
   - Proper nouns → Technically against rules, but hard to detect automatically
   - Technical jargon → Technically against rules, but hard to detect automatically

3. **Duplicate words**:
   - Model repeats words → Distance to self = 0, lowers overall score naturally

4. **Out-of-vocabulary words**:
   - Embedding model doesn't recognize word → Use subword embeddings or assign mean embedding

5. **Non-compliant outputs**:
   - Model doesn't follow format → Attempt flexible parsing, assign low score if extraction fails

## Validation Data

### Human Baseline

- **Dataset**: 8,900+ human participants (98 countries)
- **Location**: https://osf.io/kbeq6/
- **Statistics**: Mean=78.5, SD=15.2 (calibrated DAT scale)
- **Use**: Compare LLM performance to human distribution

### LLM Baseline

- **Study**: Bellemare et al. (2025) - "Divergent Creativity in Humans and Large Language Models"
- **Models tested**: GPT-3.5, GPT-4, Claude, Gemini, Falcon, StableLM, Vicuna, others
- **Dataset**: 500+ responses per model
- **Location**: https://github.com/AntoineBellemare/DAT_GPT
- **Finding**: Some LLMs exceed human performance on DAT but struggle with narrative creativity

## Integration into HELM

### Metric Class Structure

```python
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat

class SemanticDistanceMetric(Metric):
    """
    Computes semantic distance for divergent thinking tasks (DAT/S-DAT).
    """

    def evaluate_generation(
        self,
        adapter_spec,
        request_state,
        metric_service,
        eval_cache_path,
    ) -> list[Stat]:
        """
        Extract words from generation and compute average semantic distance.
        """
        # Extract model output
        # Parse words
        # Compute DAT score
        # Return statistics
        pass
```

### RunSpec Configuration

```python
# In run_specs.py or scenario-specific config
def get_sdat_metric_specs() -> list[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.semantic_distance_metric.SemanticDistanceMetric",
            args={"embedding_model": "ibm-granite/granite-embedding-278m-multilingual"}
        )
    ]
```

## References

### Papers

1. **S-DAT Framework**
   - Haase, J., Hanel, P. H. P., & Pokutta, S. (2025)
   - "S-DAT: A Multilingual, GenAI-Driven Framework for Automated Divergent Thinking Assessment"
   - AAAI/ACM Conference on AI, Ethics, and Society
   - https://arxiv.org/abs/2505.09068

2. **Original DAT**
   - Olson, J. A., Nahas, J., Chmoulevitch, D., Cropper, S. J., & Webb, M. E. (2021)
   - "Naming unrelated words predicts creativity"
   - PNAS, 118(25)
   - https://www.pnas.org/doi/10.1073/pnas.2022340118

3. **LLM Evaluation Study**
   - Bellemare, A., et al. (2025)
   - "Divergent Creativity in Humans and Large Language Models"
   - Scientific Reports
   - https://arxiv.org/abs/2405.13012

### Code Repositories

- S-DAT online tool: https://sdat.iol.zib.de/
- S-DAT data/code: https://osf.io/pv84c/
- Original DAT: https://github.com/jayolson/divergent-association-task
- LLM evaluation: https://github.com/AntoineBellemare/DAT_GPT
- Granite embeddings: https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual

## Implementation Priority

**High Priority**: This is a validated, widely-used creativity assessment with:
- Published methodology in PNAS and AAAI
- Existing LLM benchmarking data for comparison
- Simple implementation (compared to other creativity metrics)
- Multilingual support via S-DAT
- Fast evaluation (no human annotation required)

**Recommended next steps**:
1. Implement `SemanticDistanceMetric` class
2. Add granite-embedding model to HELM dependencies
3. Test on existing LLM outputs from DAT_GPT study
4. Validate correlation with human ratings
