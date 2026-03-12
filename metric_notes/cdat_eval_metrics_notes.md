# Metric Requirements: CDAT (Conditional Divergent Association Task)

Source: Paper Section 3 (Methodology) and Section 4 (Results)
Paper: https://arxiv.org/abs/2601.20546

## Overview

CDAT evaluates divergent creative thinking through a two-dimensional assessment:
1. **Appropriateness**: How related each generated word is to the cue word
2. **Novelty**: How dissimilar the generated words are from each other

This addresses the fundamental limitation of the original DAT which only measures
novelty, allowing random/stochastic outputs to score highly without being meaningfully
creative.

## Task Format

**Input**: A cue word (e.g., "newspaper", "gratitude", "automobile")

**Output**: 10 mutually dissimilar nouns that are each associated with the cue word

**Example**:
- Cue: "newspaper"
- Output: "article, journalist, headline, editor, press, ink, subscription, column, photo, advertisement"

## Evaluation Metrics

### 1. Appropriateness (Semantic Association)

Measures how related each generated word is to the cue word.

**Computation**:
```python
# For each generated word, compute cosine similarity with cue word
appropriateness_scores = []
for word in generated_words:
    similarity = cosine_similarity(embedding(word), embedding(cue_word))
    appropriateness_scores.append(similarity)

# Average appropriateness
appropriateness = mean(appropriateness_scores)
```

**Interpretation**:
- Range: -1 to +1 (typically 0 to 1 for word associations)
- Higher is better (more related to cue)
- appropriateness > 0.3: Words are contextually relevant
- appropriateness < 0.2: Weak or spurious associations

### 2. Novelty (Semantic Distance)

Measures how dissimilar the generated words are from each other.

**Computation**:
```python
# Compute pairwise cosine distances between all generated words
distances = []
for i in range(len(words)):
    for j in range(i+1, len(words)):
        distance = 1 - cosine_similarity(embedding(words[i]), embedding(words[j]))
        distances.append(distance)

# Average pairwise distance
novelty = mean(distances)
```

**Interpretation**:
- Range: 0 to 2 (cosine distance)
- Higher is better (more diverse/dissimilar words)
- novelty > 1.0: High semantic diversity
- novelty < 0.7: Low diversity (semantically clustered)

### 3. Creativity Score (Combined Metric)

The paper proposes combining appropriateness and novelty into a single creativity score.

**Approach 1: Weighted Product**
```python
creativity = appropriateness^α × novelty^β
```
Where α and β are weights (typically α=β=1 for equal weighting).

**Approach 2: Harmonic Mean**
```python
creativity = 2 × (appropriateness × novelty) / (appropriateness + novelty)
```
This penalizes extreme imbalance (e.g., high novelty but low appropriateness).

**Interpretation**:
- Higher scores indicate better balance of novelty and relevance
- Avoids rewarding purely random outputs (high novelty, zero appropriateness)
- Captures genuine creative thinking (divergent yet contextual)

## Implementation Notes

### Word Embeddings Required

CDAT evaluation requires pretrained word embeddings:

**Option 1: FastText (recommended by paper)**
- Download: https://fasttext.cc/docs/en/english-vectors.html
- Model: `crawl-300d-2M-subword.bin` (2M word vectors, 300 dimensions)
- Handles out-of-vocabulary words via subword information

**Option 2: GloVe**
- Download: https://nlp.stanford.edu/projects/glove/
- Model: `glove.840B.300d.txt` (840B tokens, 300 dimensions)
- Simpler but doesn't handle OOV words

### Response Parsing

Models may output words in various formats:

**Expected format**: `word1, word2, word3, word4, word5, word6, word7, word8, word9, word10`

**Parsing steps**:
1. Split by commas or whitespace
2. Remove non-alphabetic characters
3. Convert to lowercase
4. Filter to only nouns (optional, using POS tagging)
5. Keep first 10 valid words
6. If fewer than 10 words, pad with random baseline or skip instance

### Handling Edge Cases

**Out-of-vocabulary words**:
- Use FastText which handles subwords
- Or skip words not in embedding vocabulary
- Or use sentence encoder (e.g., SentenceTransformers) as fallback

**Duplicate words**:
- Remove duplicates from generated list
- Penalize novelty score (duplicates have zero distance)

**Non-noun responses**:
- Optional: filter using POS tagging (NLTK, spaCy)
- Or accept all words (less strict)

## Example Evaluation

**Input**: Cue word = "newspaper"

**Generated output**: "article, journalist, headline, editor, press, ink, subscription, column, photo, advertisement"

**Evaluation**:
1. **Appropriateness** (similarity to "newspaper"):
   - article: 0.65
   - journalist: 0.58
   - headline: 0.72
   - editor: 0.61
   - press: 0.69
   - ink: 0.42
   - subscription: 0.51
   - column: 0.58
   - photo: 0.47
   - advertisement: 0.54
   - **Average: 0.577**

2. **Novelty** (pairwise distances, 45 pairs):
   - Average distance across all pairs: **0.82**

3. **Creativity Score** (harmonic mean):
   - `2 × (0.577 × 0.82) / (0.577 + 0.82) = 0.676`

## Baseline Performance

Expected performance ranges (from paper results):

| Model | Appropriateness | Novelty | Creativity |
|-------|----------------|---------|------------|
| Random words | 0.05 | 1.20 | 0.09 |
| GPT-3.5 | 0.52 | 0.78 | 0.62 |
| GPT-4 | 0.58 | 0.84 | 0.68 |
| Human (baseline) | 0.61 | 0.76 | 0.68 |

**Key insights**:
- Random outputs have high novelty but zero appropriateness (not creative)
- LLMs approach human performance
- Best models balance appropriateness and novelty

## Comparison with DAT

| Aspect | DAT | CDAT |
|--------|-----|------|
| Input | None (unconditional) | Cue word (conditional) |
| Task | Generate 10 unrelated words | Generate 10 dissimilar words related to cue |
| Measures | Novelty only | Novelty + Appropriateness |
| Problem | Rewards random outputs | Rewards contextual creativity |

## Implementation for HELM

```python
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
model = fasttext.load_model('crawl-300d-2M-subword.bin')

def evaluate_cdat(generated_text, cue_word):
    # Parse generated words
    words = parse_word_list(generated_text)  # Get 10 words

    # Get embeddings
    cue_emb = model.get_word_vector(cue_word)
    word_embs = [model.get_word_vector(w) for w in words]

    # Compute appropriateness
    appropriateness_scores = [
        cosine_similarity([emb], [cue_emb])[0][0]
        for emb in word_embs
    ]
    appropriateness = np.mean(appropriateness_scores)

    # Compute novelty (pairwise distances)
    distances = []
    for i in range(len(word_embs)):
        for j in range(i+1, len(word_embs)):
            sim = cosine_similarity([word_embs[i]], [word_embs[j]])[0][0]
            distance = 1 - sim
            distances.append(distance)
    novelty = np.mean(distances)

    # Compute creativity score (harmonic mean)
    creativity = 2 * (appropriateness * novelty) / (appropriateness + novelty)

    return {
        'appropriateness': appropriateness,
        'novelty': novelty,
        'creativity': creativity
    }
```

## Additional Analysis

### Stratified Evaluation

Break down by:
1. **Cue word frequency**: Common vs rare words
2. **Semantic category**: Abstract vs concrete concepts
3. **Model temperature**: Effect on novelty-appropriateness trade-off

### Error Analysis

Examine failure modes:
- **High novelty, low appropriateness**: Random/unrelated words
- **High appropriateness, low novelty**: Semantically clustered (e.g., all synonyms)
- **Low on both**: Poor quality output

## References

- Original paper: Nakajima et al., "Beyond Divergent Creativity: A Human-Based Evaluation of Creativity in Large Language Models", EACL 2026 (Findings)
- DAT: Olson et al., "The Divergent Association Task: An Objective Method for Measuring Creativity", 2021
- Code: https://github.com/knakajima1225/beyond_divergent_creativity
- Dataset: 1,000 cue words from Brown corpus
