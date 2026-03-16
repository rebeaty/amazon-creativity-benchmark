# Annotator Notes: ConvBench

Source: https://github.com/shirlyliu64/ConvBench
        Paper: ConvBench (NeurIPS 2024), Section 3 — Evaluation Protocol

## Task

Evaluate the quality of a model's creative response to Turn 3 of a multi-turn
visual conversation. The response must be grounded in the image and contextually
consistent with the preceding turns.

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4V (or any vision-capable judge; paper uses GPT-4V)
Evaluation type: rubric-based scoring per instance

## Evaluation Rubric

Each instance has a `third_turn_demands` field containing 2-3 yes/no criteria
specific to that instance's creative task. Example for a slogan generation task:

```
1. Whether the slogan refers to 'Hummer H2'?
2. Whether the slogan is catchy and creative?
3. Whether the slogan mentions off-road capability?
```

The judge scores each criterion as satisfied (1) or not (0), then aggregates
across all criteria for an overall score.

## Judge Prompt Template

Rate whether the following response satisfies each of the listed criteria.

Image: [image]

Conversation context:
Turn 1 - Q: {t1_question}
Turn 1 - A: {t1_answer}
Turn 2 - Q: {t2_question}
Turn 2 - A: {t2_answer}
Turn 3 - Q: {t3_question}

Model Response: {RESPONSE}

Criteria:
{third_turn_demands}

For each criterion, answer Yes (1) or No (0). Then provide an overall score
as the fraction of criteria satisfied (e.g., 2/3).

## Scoring

- Per-instance score: fraction of demands satisfied (0.0–1.0)
- Aggregate score: mean across all 578 instances
- Paper also reports per-category scores across the 60 instruction categories

## Notes

- The judge must be vision-capable (GPT-4V) since criteria often reference
  specific visual content in the image (e.g., "refers to Hummer H2").
- Turn 3 reference answers (third_turn_answer) can serve as additional grounding
  context for the judge but are not required.
- Criteria vary significantly by task type (poem: rhyme/theme; recipe: ingredients/steps;
  story: characters/plot; slogan: brand/creativity).
