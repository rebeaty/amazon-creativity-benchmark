# Metric Notes: AAAR-1.0 — PaperWeakness

Source: https://github.com/RenzeLou/AAAR-1.0/blob/main/scripts/subtask3_metric.py
        https://github.com/RenzeLou/AAAR-1.0/blob/main/scripts/subtask3_metric_cross_diversity.py

## Task

Given a model's predicted list of paper weaknesses, compare against the
human-reviewer ground truth using soft semantic matching.

## Metrics

All metrics use SentenceBERT (`all-mpnet-base-v2`) for semantic similarity.

| Metric | Description |
|--------|-------------|
| `S-F1` | Soft F1: harmonic mean of soft precision and soft recall |
| `S-Precision` | Fraction of predicted weaknesses semantically matched to a reference weakness |
| `S-Recall` | Fraction of reference weaknesses covered by at least one prediction |
| `En-F1` | Entailment-based F1 (binary match threshold applied to similarity scores) |

## Ground Truth Structure

Each paper has weaknesses from multiple reviewers (list-of-lists). The evaluation
script compares the model's predicted weakness list against the full set of
reviewer weaknesses. A prediction is "matched" if its cosine similarity with any
reference weakness exceeds a threshold (implicit in soft accumulation).

## Diversity Metric

`subtask3_metric_cross_diversity.py` additionally measures lexical and semantic
diversity of the predicted weaknesses across the paper set — rewarding models
that generate varied critiques rather than repetitive boilerplate.

## Implementation Requirements

1. Install `sentence-transformers` (`all-mpnet-base-v2` model).
2. Parse model output as a numbered list of weakness strings.
3. Compare each predicted weakness against all reference weaknesses per paper.
4. Aggregate soft precision, recall, and F1 across all 993 papers.

## Notes

- Ground truth consists of weaknesses from actual ICLR 2023 peer reviews
  (OpenReview data).
- Multiple reviewers per paper means there is no single "correct" weakness set;
  soft matching accommodates this ambiguity.
- The scenario stores flattened reviewer weaknesses as a single reference string
  for HELM compatibility; metric implementation should re-parse this as a list
  (split on newlines).
