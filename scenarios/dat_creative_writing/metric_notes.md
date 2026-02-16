# Custom Metric Requirements: DAT + Creative Writing

Source: scripts/dat.py, scripts/analyze_stories_dsi-lziv.py in
https://github.com/AntoineBellemare/DAT_GPT

## 1. DAT Score (Divergent Association Task)

Reference: Olson et al. (2021) "Naming unrelated words predicts creativity"
https://doi.org/10.1073/pnas.2022340118

### Algorithm

1. Parse model output to extract 10 words (list format)
2. Validate each word against a dictionary + GLoVe vocabulary
3. Keep first 7 valid unique words
4. Compute pairwise cosine distances using GLoVe 840B.300d embeddings
5. DAT score = mean(pairwise distances) × 100

### Requirements

- GLoVe 840B.300d embeddings (~2.2GB): https://nlp.stanford.edu/projects/glove/
- Dictionary file (words.txt) for word validation
- Word validation: lowercase, nouns only, no proper nouns, compound word
  handling (spaces → hyphens)

### Reference implementation

```python
import itertools
import numpy as np
from scipy.spatial.distance import cosine

def dat_score(words, glove_vectors, minimum=7):
    """Compute DAT score from list of words."""
    # Validate and deduplicate
    valid = []
    for w in words:
        w = w.strip().lower()
        if w in glove_vectors and w not in valid:
            valid.append(w)
    if len(valid) < minimum:
        return None  # Not enough valid words
    subset = valid[:minimum]
    # Pairwise cosine distances
    distances = [
        cosine(glove_vectors[w1], glove_vectors[w2])
        for w1, w2 in itertools.combinations(subset, 2)
    ]
    return (sum(distances) / len(distances)) * 100
```

### Score interpretation

- Low (~50): semantically similar words (e.g., body parts)
- Average (~78): moderately diverse words
- High (~95): maximally distant concepts

Human baseline: mean ~78, range ~50-98 (N=100,000).

## 2. DSI (Divergent Semantic Integration)

Reference: Johnson et al. (2022) "Extracting Creativity from Narratives
using Distributional Semantic Modeling"

### Algorithm

1. Tokenize text into sentences (NLTK PunktSentenceTokenizer)
2. Extract BERT-large-uncased embeddings (layers 6-7) for each word
3. Compute pairwise cosine distances between all word embeddings
4. DSI = mean of all pairwise distances

### Requirements

- BERT-large-uncased model (HuggingFace transformers)
- NLTK sentence tokenizer
- Measures semantic diversity/spread of concepts in narrative text

## 3. Lempel-Ziv Complexity (Lziv)

### Algorithm

1. Compute normalized Lempel-Ziv complexity of raw text
2. Uses `antropy.lziv_complexity(text, normalize=True)`

### Requirements

- `antropy` Python package
- Information-theoretic measure of text compressibility
- Higher = more complex/less repetitive text

## 4. LLM-as-Judge Quality Rating

For flash fiction stories, GPT-4 ratings (1-5 scale) are available in the
paper's data (479 stories rated). This could be replicated as an annotator.

## Aggregation

Metrics are computed per-instance and aggregated across instances:
- DAT: mean score across N trials
- DSI/Lziv: mean across N generated texts
- Distribution statistics (std, min, max) are also informative for
  measuring creative consistency vs. variability
