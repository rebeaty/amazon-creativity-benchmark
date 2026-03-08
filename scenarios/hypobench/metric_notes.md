# Metric Notes: HypoBench

Source: Paper arXiv:2504.11524, Sections 4–5; https://chicagohai.github.io/HypoBench/

## Current Implementation

The scenario uses `eval_type: open_ended` (ROUGE-L, BERTScore) comparing generated
hypotheses against `known_hypotheses` from the literature (metadata.json).

This is a **proxy metric** — useful for automated comparison, but not the paper's
primary evaluation.

---

## Paper's Primary Evaluation: Two-Step Pipeline

The paper evaluates hypothesis generation methods via a **multi-step pipeline**:

1. **Hypothesis Generation**: Given N training observations, the model generates K hypotheses.
2. **Classification with Hypotheses**: For each test example, the model is prompted
   with the generated hypotheses and asked to predict the label.
3. **Accuracy**: Fraction of test examples correctly classified using the hypotheses.

This requires running two separate LLM calls per instance and cannot be expressed
as a single HELM Scenario without custom infrastructure.

---

## HDR: Hypothesis Discovery Rate (Synthetic Tasks Only)

For synthetic tasks (`admission`, `election`, `preference`, `shoe`, `marine`), the
dataset provides exact `ground_truth_hypotheses` (e.g., "Students with an A in Math
will be admitted, otherwise rejected"). The paper uses **HDR** to measure how many
ground truth rules the model rediscovers.

**HDR computation** (Paper Section 4.2):
1. For each ground truth hypothesis H_gt and each generated hypothesis H_gen:
   - Use an LLM judge (GPT-4) to decide if H_gen semantically matches H_gt
2. HDR = |{H_gt : ∃ H_gen that matches H_gt}| / |{H_gt}|

This requires:
- An LLM-as-judge step comparing generated vs. ground truth hypotheses
- Threshold tuning (binary match decision)
- Implementation as a custom HELM metric (not currently available)

**Human correlation**: HDR correlates well with human judgments of hypothesis quality
(Spearman ρ = 0.78 reported in paper Table 3).

---

## Recommendations

| Scenario | Recommended Metric | Status |
|---|---|---|
| Real tasks (7 tasks) | Two-step accuracy pipeline | Needs custom runner |
| Synthetic tasks (5 tasks) | HDR with LLM judge | Needs custom metric |
| Interim proxy | ROUGE-L / BERTScore vs. known_hypotheses | Implemented (open_ended) |

For synthetic tasks to be onboarded properly, the scenario would need to be extended
to load synthetic data from nested subdirectories (`synthetic/{task}/level_{1-4}/base/`).
