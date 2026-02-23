# Annotator Notes: ArtInsight Artwork Description Evaluation

Source: https://github.com/makeabilitylab/ArtInsight/blob/main/Artwork-Description-Scoring/description_scorer.py
        https://github.com/makeabilitylab/ArtInsight/blob/main/Artwork-Description-Scoring/README.md

## Task

Given a model's description of a child's artwork (generated for a blind/low-vision
parent), score the description on a 0-16 rubric using a vision-capable LLM judge.
The judge receives both the original artwork image and the generated description.

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4o (with vision)
Evaluation type: rubric-based scoring per instance
Approach: few-shot prompted (6 example image+description+score turns)

## Scoring Rubric (0–16 points)

| # | Criterion | Points | Description |
|---|-----------|--------|-------------|
| 1 | Not presumptive | 0–4 | Does not make unwarranted inferences about what is depicted (e.g., assumes a blob is a cat without evidence) |
| 2 | Not reductive | 0–4 | Does not use dismissive or diminishing language about the child's artistic effort |
| 3 | Sufficient detail | 0–4 | Description is not too brief; covers the artwork substantively |
| 4 | All major elements captured | 0–4 | All significant visual elements (colors, shapes, textures, text, composition) are mentioned |
| 5 | Miscellaneous | subtractive | Deductions for asking questions, speculating about meaning, or other violations |

## Judge Prompt Approach

The scorer uses a **few-shot conversation** with 6 example turns, each containing:
- An artwork image (base64 JPEG)
- A model-generated description
- The correct score with rationale

After the few-shot examples, the judge scores the new description given the
artwork image.

Full implementation: `description_scorer.py` in the repository.

## Notes

- The judge **must be vision-capable** (GPT-4o or equivalent) since it needs to
  verify that descriptions accurately reflect visible image content.
- Temperature=0 for the judge to ensure deterministic scoring.
- The few-shot examples from `description_scorer.py` should be included in the
  judge prompt to calibrate scoring behavior.
- Paper finding: GPT-4o and GPT-4o Mini both achieve ~15.97/16 average with the
  optimal system prompt; Gemini 1.5 Flash scored only 11.5/16.
- Human study (5 BLV families + 1 therapist) validated the rubric; all preferred
  GPT-4o descriptions to human-written ones for this task.
