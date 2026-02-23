# Annotator Notes: AAAR-1.0 — ExperimentDesign

Source: https://github.com/RenzeLou/AAAR-1.0/blob/main/scripts/calculate_metrics_subtask2_exp_entailment.py
        https://github.com/RenzeLou/AAAR-1.0/blob/main/scripts/prompt_templates.py (Exp_entailment class)

## Task

Given a model's predicted list of experiment ideas and the human-annotated ground
truth list, judge whether each predicted experiment is entailed by (i.e., matches)
any ground truth experiment, and vice versa.

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4 (gpt-4-1106-preview)
Evaluation type: pairwise entailment (binary, per experiment pair)

## Scoring Dimensions

Two aggregate scores derived from pairwise entailment decisions:

| Metric | Description |
|--------|-------------|
| `recall_gt_entail_score` | Fraction of ground-truth experiments covered by predictions |
| `precision_pred_entail_score` | Fraction of predicted experiments that match a ground-truth experiment |

A prediction "matches" a ground truth if the judge returns 1 (entailed).
Predictions that match no ground truth are logged as "novel_exps" (novel but
unverifiable).

## Judge Prompt

Uses `Exp_entailment` template from `scripts/prompt_templates.py`.
The template checks: does experiment idea A appear (semantically) in list B?
Exact prompt text: see `prompt_templates.Exp_entailment` class in the repo.

## Secondary Metric

SentenceBERT (`all-mpnet-base-v2`) soft F1 is also computed between predicted
and reference experiment lists (see `scripts/subtask2_metric.py`).

## Notes

- Ground truth annotations come from domain experts (ML researchers).
- Each paper has one annotated experiment list; the model must propose a matching
  set of experiments.
- "Novel" predictions (not matching any gold experiment) are tracked separately
  and not penalized as wrong — they represent creative but unverifiable ideas.
