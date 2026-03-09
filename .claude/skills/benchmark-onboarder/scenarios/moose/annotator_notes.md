# Annotator Requirements: MOOSE

Source: `evaluate_utils.py` → `prompts_for_evaluator_modules()` in the MOOSE GitHub repo
(https://github.com/ZonglinY/MOOSE/blob/main/evaluate_utils.py)

## Configuration for LLMAsJuryAnnotator

**Judge model**: GPT-4 (default in original paper; GPT-3.5 also used in later variant)

**Dimensions**: Validness, Novelty, Helpfulness
**Scale**: 1–5 per dimension

## Judge Prompt Template

The original evaluator scores hypotheses **without background context** (just the hypothesis text).
For HELM, we extend it to include the research background for more grounded evaluation.

```
Given a not yet peer reviewed research hypothesis in business domain, try to evaluate
the research hypothesis from three research aspects and give score according to
evaluation guidelines provided below. All three aspects should be evaluated in a 5 point scale.

Validness:
5 - The hypothesis completely reflects the reality.
4 - The hypothesis mostly reflects the reality with minor issues.
3 - The hypothesis partially reflects the reality.
2 - The hypothesis mostly violates the reality.
1 - The hypothesis completely violates the reality.

Novelty:
5 - The hypothesis is completely novel compared to existing literature.
4 - The hypothesis is mostly novel with minor overlaps.
3 - The hypothesis is somewhat novel.
2 - The hypothesis is not very novel or inspiring.
1 - The hypothesis is not novel at all and is uninspiring.

Helpfulness:
5 - The hypothesis is a mature research hypothesis that can be submitted for publication.
4 - The hypothesis is mostly helpful for research.
3 - The hypothesis is somewhat helpful.
2 - The hypothesis is not very helpful or inspiring.
1 - The hypothesis is unhelpful and uninspiring.

Research Background:
{BACKGROUND}

Hypothesis to evaluate:
{MODEL_OUTPUT}

Please give a response to the initial question on scoring the hypothesis from three aspects.
(response format: 'Validness score: \nConcise reason: \nNovelty score: \nConcise reason: \nHelpfulness score: \nConcise reason: \n')
```

**Variables:**
- `{BACKGROUND}`: The research background from the scenario Instance input (extracted before the final instruction line)
- `{MODEL_OUTPUT}`: The model's generated hypothesis

## Score Parsing

From `evaluate_utils.py → pick_score()`:

Parse the three scores from the response using these patterns:
- `Validness score: <int>`
- `Novelty score: <int>`
- `Helpfulness score: <int>`

All scores are integers in range [1, 5]. The function validates that exactly 3 scores
and 3 reasons are present; raise an exception if parsing fails.

Final reported metrics:
- `moose_validness`: mean Validness score across instances
- `moose_novelty`: mean Novelty score across instances
- `moose_helpfulness`: mean Helpfulness score across instances
- `moose_mean`: mean of all three scores (aggregate quality)

## Deviation from Original Paper

The original `evaluator.py` passes only the hypothesis to GPT-4 (`pre_prompt + cur_hyp + post_prompt`),
without the research background. This means the judge scores scientific plausibility and novelty
of the hypothesis in isolation, not relative to the specific background provided.

For HELM, we include `{BACKGROUND}` in the judge prompt so that:
1. Validness can be assessed relative to what the background claims
2. Novelty can be judged given what the background already covers
3. Helpfulness is grounded in the actual research problem

If strict replication of the original is desired, remove the `Research Background:` block.

## Optional: Gold Hypothesis Comparison

The dataset includes `Main hypotheis` (gold-standard human hypothesis) as a Reference.
An alternative judge prompt variant can compare the model output against the gold hypothesis:

```
Gold hypothesis for reference:
{GOLD_HYPOTHESIS}

Model hypothesis to evaluate:
{MODEL_OUTPUT}
```

This enables relative scoring (is the model hypothesis better/worse than human baseline?).

## Human Evaluation Details (from paper)

Human expert evaluation uses the same 3-point scale with the same dimensions.
Paper reports GPT-3.5 judge correlates reasonably with human experts (MOOSE ACL 2024).
