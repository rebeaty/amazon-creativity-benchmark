# Custom Metric Requirements: Divergent Association Task (DAT)

**Source:** Chen & Ding (2023) EMNLP - [Paper](https://aclanthology.org/2023.findings-emnlp.858/)
**Code:** [probing_creativity](https://github.com/DingNLab/probing_creativity)
**Original:** Olson et al. (2021) PNAS - [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8237676/)

## Overview

The DAT (Divergent Association Task) is a creativity measure that evaluates semantic diversity in word generation. Models generate 10 unrelated nouns, and creativity is scored by computing average cosine distance between all word pairs using word embeddings.

## Metric Implementation Requirements

### 1. Parse Model Output

Extract 10 nouns from the generated text. See `dataset.py` (lines 73-102) for parsing logic:

- Split by newlines (preferred), commas, or asterisks
- Clean: strip non-alphabetic characters, convert to lowercase
- Validate: single words only, minimum 2 characters
- Handle compound words (e.g., "cul-de-sac" → "cul-de-sac")
- Take first 10 unique valid words

**Example outputs:**
```
Low creativity (DAT ≈ 50): arm, eyes, feet, hand, head, leg, body
Average (DAT ≈ 78): bag, bee, burger, feast, office, shoes, tree
High creativity (DAT ≈ 95): hippo, jumper, machinery, prickle, tickets, tomato, violin
```

### 2. Load Word Embeddings

**Primary:** GloVe 840B 300d embeddings
- Download: https://nlp.stanford.edu/projects/glove/
- File: `glove.840B.300d.txt` (5.65 GB)
- Dimensions: 300d vectors for 2.2M tokens

**Optional validation:** Word2Vec, FastText (paper tested multiple embeddings)

### 3. Compute DAT Score

Algorithm from `dat_score.py` (lines 83-105):

1. **Validate words:** Map each word to embedding vector (skip if not in vocabulary)
2. **Keep first 7 valid unique words** (minimum threshold)
3. **Compute pairwise distances:** For all C(7,2) = 21 word pairs, calculate cosine distance
4. **Average and scale:** `DAT_score = mean(distances) × 100`

**Formula:**
```
cosine_distance(w1, w2) = 1 - (v1 · v2) / (||v1|| × ||v2||)
DAT = (Σ cosine_distance(wi, wj) for all pairs) / num_pairs × 100
```

**Score range:** 0-100 (higher = more creative)

### 4. Human Baseline Statistics

From `dataset/human.json` (8,572 participants, 98 countries):

- **Mean:** ~78 (SD: ~10)
- **Low performers:** 50-65
- **Average:** 70-85
- **High performers:** 85-95
- **Top 4%:** >90

### 5. LLM Benchmarks (Chen & Ding 2023)

**Greedy decoding:**
- GPT-4: ~90 (96th percentile vs humans)
- GPT-3.5-turbo: ~78 (50th percentile)
- Other models: 65-80

**With temperature scaling (t=0.7, top_p=0.9):**
- Most models improve by 5-15 points
- GPT-4 remains stable
- Trade-off: higher variance

### 6. Implementation Notes

**Word validation edge cases:**
- Compound words: Try hyphenated and concatenated forms
- Multi-word phrases: Extract first non-article word
- Out-of-vocabulary: Skip and continue (need ≥7 valid words for score)
- Duplicates: Only count first occurrence

**Failure modes:**
- Model generates <7 valid words → Return None/NaN
- Model outputs numbered lists → Strip numbers during parsing
- Model outputs sentences → Extract nouns (may need POS tagging)

**Efficiency:**
- Load GloVe embeddings once, cache in memory
- Only load vocabulary subset if memory constrained
- Consider approximate nearest neighbor for large-scale evaluation

## HELM Integration

### Metric Class Structure

```python
class DATMetric(Metric):
    def evaluate(self, adapter_spec, request_state, metric_service, eval_cache_path):
        # 1. Parse model output → extract 10 words
        # 2. Validate words against GloVe vocabulary
        # 3. Compute average pairwise cosine distance
        # 4. Scale to 0-100 and return as MetricResult
```

### Required Dependencies

- `numpy` - for vector operations
- `scipy` - for `scipy.spatial.distance.cosine`
- GloVe embeddings file (download separately)
- Optional: `gensim` for Word2Vec/FastText validation

### Configuration

Add to RunSpec:
```python
metric_specs = [
    MetricSpec(class_name="DATMetric", args={"embedding_path": "glove.840B.300d.txt"})
]
```

## Validation

Test with known examples from `examples.py`:

```python
# Low creativity (expected: 50)
["arm", "eyes", "feet", "hand", "head", "leg", "body"]

# Average (expected: 78)
["bag", "bee", "burger", "feast", "office", "shoes", "tree"]

# High creativity (expected: 95)
["hippo", "jumper", "machinery", "prickle", "tickets", "tomato", "violin"]
```

## References

1. **Original DAT:** Olson, J. A., et al. (2021). Naming unrelated words predicts creativity. *PNAS*, 118(25).
2. **LLM Evaluation:** Chen, H., & Ding, N. (2023). Probing the "Creativity" of Large Language Models. *EMNLP Findings*.
3. **GloVe Embeddings:** Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
4. **Code Reference:** https://github.com/DingNLab/probing_creativity
