# Custom Metrics for PACE

**Source:** PACE repository (src/association_calculate_model.py)

## Overview

PACE requires custom metric implementation that cannot use standard HELM metrics like exact_match or BLEU. The benchmark evaluates creativity through two complementary metrics:

1. **Type-Token Ratio (TTR)** - Vocabulary diversity
2. **Association Distance** - Semantic creativity via GloVe embeddings

## Metric 1: Type-Token Ratio (TTR)

### Calculation
```python
def calculate_ttr(word_list):
    num_tokens = len(word_list)
    num_types = len(set(word_list))
    ttr = num_types / num_tokens
    return ttr
```

### Interpretation
- **Range**: 0 to 1
- **Higher is better**: More unique words = more vocabulary diversity
- **Computed over**: All words across the 3 parallel chains (excluding the seed word)

## Metric 2: Association Distance

### Requirements
- **GloVe embeddings**: GloVe 6B 300d word vectors
  - Download from: https://nlp.stanford.edu/projects/glove/
  - File: glove.6B.300d.txt (~1GB)
- **Computation**: Cosine distance between consecutive word embeddings

### Calculation Process

1. **Load GloVe embeddings**:
   ```python
   word_vector = {}
   with open('glove.6B.300d.txt', 'r') as f:
       for line in f:
           splitline = line.rstrip().split(' ')
           word = splitline[0]
           embedding = np.asarray(splitline[1:], dtype='float32')
           word_vector[word] = embedding
   ```

2. **For each association chain** (3 chains per seed):
   ```python
   from scipy.spatial.distance import cosine

   chain_distance = []
   for i in range(1, len(chain)):
       # Accumulate distances from current word to all previous words
       cumulative_dist = 0
       for j in range(i):
           vec1 = word_vector[chain[i].lower()]
           vec2 = word_vector[chain[j].lower()]
           cumulative_dist += cosine(vec1, vec2)

       # Average distance from word i to all previous words
       chain_distance.append(cumulative_dist / i)

   # Average over all positions in chain
   association_distance = sum(chain_distance) / len(chain_distance)
   ```

3. **Final score**: Average association distance across 3 chains

### Interpretation
- **Range**: 0 to 1 (cosine distance)
- **Higher is better**: Greater semantic distance = more creative associations
- **Measures**: How far each new word ventures from previous words in semantic space

## Output Parsing

Models must generate structured output for 3 parallel chains. Expected format:

```
Chain 1: word1 (reason) → word2 (reason) → ... → word20 (reason)
Chain 2: word1 (reason) → word2 (reason) → ... → word20 (reason)
Chain 3: word1 (reason) → word2 (reason) → ... → word20 (reason)
```

The metric implementation needs to:
1. Parse the model output to extract word sequences
2. Verify each chain has exactly 20 words
3. Remove seed word if repeated
4. Calculate TTR and association distance

## Implementation Notes

- **Missing embeddings**: If a word isn't in GloVe vocabulary, use mean embedding vector
- **Casing**: Convert all words to lowercase for GloVe lookup
- **Chain length**: Original PACE uses chains of 20 words each (60 words total + seed)
- **Proper nouns**: Should be excluded per task instructions (not in GloVe anyway)

## HELM Integration Steps

1. Implement custom metric class extending `Metric`
2. Add GloVe embedding loading (with caching)
3. Parse model outputs to extract word chains
4. Calculate TTR and association distance
5. Return scores as metric values

## Example Output Format

```json
{
  "ttr": 0.9298,
  "association_distance": 0.6947,
  "num_chains": 3,
  "chain_lengths": [20, 20, 20],
  "total_words": 60,
  "unique_words": 56
}
```

## References

- PACE GitHub: https://github.com/ziliang6/PACE
- GloVe embeddings: https://nlp.stanford.edu/projects/glove/
- Original calculation code: `src/association_calculate_model.py`
