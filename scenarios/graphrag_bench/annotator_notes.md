# Annotator Requirements: GraphRAG-Bench (Creative Generation)

Source: Evaluation/generation_eval.py and Evaluation/metrics/ in the dataset repo
Paper: arXiv:2506.05690

## Evaluation Overview

The paper uses RAGAS-based LLM-as-judge evaluation with two metrics for
Creative Generation (type4 in the eval script):

1. **answer_correctness** — semantic correctness vs. gold answer
   (LLM + embedding similarity via RAGAS)
2. **coverage_score** — whether the response covers the key information
   in the gold answer (LLM-as-judge)
3. **faithfulness** — whether the response is consistent with the retrieved
   context (not applicable in zero-shot HELM setting; skip)

## Judge Model

Default: `gpt-4-turbo` (configurable via `--model` flag in eval script)
Embeddings: `BAAI/bge-large-en-v1.5`

## Metric Notes

- `answer_correctness` in RAGAS combines factual similarity (F1) and
  semantic similarity (embedding cosine); weight typically 0.75/0.25
- `coverage_score` asks the judge: "Does the generated answer cover all
  key information from the ground truth?"
- No explicit prompt template is published for the judge; uses RAGAS defaults

## Dataset Stats (Creative Generation only)

| Domain  | Count |
|---------|-------|
| novel   | 67    |
| medical | 166   |
| **total** | **233** |

## Recommended HELM Configuration

eval_type: llm_judge
metrics: answer_correctness, coverage_score
judge_model: gpt-4-turbo
