# Metric Notes: Chinese Homophonic Puns (PER-Task)

Source: `evaluatePunchline.py`
(https://github.com/YesianRohn/DuanzAI/blob/main/evaluatePunchline.py)

## Task

Punchline Entity Recognition: given a Chinese homophonic pun joke, identify
the punchline word/phrase (usually 2–4 Chinese characters).

## Evaluation Methodology

The paper uses two metrics computed over all 1,000 examples:

### 1. Exact Match Accuracy
```
exact_match_accuracy = count(predicted == gold) / 1000
```

### 2. Fuzzy Similarity Accuracy
For non-exact matches, compute similarity using both SequenceMatcher and fuzzywuzzy,
taking the max:

```python
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz

similarity = SequenceMatcher(None, predicted, gold).ratio()
fuzzy = fuzz.ratio(predicted, gold) / 100
score = min(1.0, max(similarity, fuzzy))
```

Final score:
```
fuzzy_accuracy = (count_exact_match + sum_of_partial_scores) / 1000
```

## Baseline Results (from paper)

| Model              | Exact Match | Fuzzy Match |
|--------------------|-------------|-------------|
| GLM-6B (0-shot)    | ~57%        | ~65%        |
| GPT-3.5 (0-shot)   | ~92%        | ~95%        |
| GPT-3.5 (5-shot)   | ~97%        | ~97%        |
| DuanzAI-PER system | ~97%        | ~97%        |

## Implementation Notes

- Standard HELM `exact_match` metric does NOT apply: it uses English normalization
  (lowercase, punctuation removal) not appropriate for Chinese character sequences.
- Standard BLEU/ROUGE from `open_ended` metrics are NOT appropriate for this
  short-extraction task (gold is 2–4 characters; surface form must match closely).
- A custom metric using `fuzzywuzzy` is needed for accurate evaluation.

## Dependencies

```
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.12.0   # speeds up fuzzywuzzy
```
