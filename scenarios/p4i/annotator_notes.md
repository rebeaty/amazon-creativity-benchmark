# Annotator Notes: P4I Summary-to-Poem Generation

Source: https://arxiv.org/abs/2507.13708
        https://github.com/SofeeyaJ/PoemTale-Diffusion

## Task

Given a model-generated poem (written from a prose summary + emotional tone),
evaluate the poem's creative and poetic quality using an LLM judge.

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4o
Evaluation type: rubric-based scoring per instance
Approach: zero-shot rubric prompt

## Scoring Rubric (1–5 scale per criterion)

| # | Criterion | Description |
|---|-----------|-------------|
| 1 | Thematic Fidelity | Does the poem address the themes and content described in the summary? |
| 2 | Emotional Resonance | Does the poem convey the specified emotional tone effectively? |
| 3 | Imagery & Language | Does the poem use vivid, evocative language with effective imagery? |
| 4 | Poetic Craft | Does the poem employ poetic devices (metaphor, rhythm, alliteration, enjambment, etc.) and demonstrate structural awareness (stanzas, line breaks)? |
| 5 | Originality | Does the poem feel fresh and creative rather than generic or clichéd? |

## Judge Prompt Template

```
You are evaluating a poem generated from a prose summary. Score the poem
on each criterion below (1-5, where 1=poor, 5=excellent):

1. Thematic Fidelity: Addresses the themes described in the summary
2. Emotional Resonance: Conveys the specified emotional tone effectively
3. Imagery & Language: Uses vivid, evocative language with effective imagery
4. Poetic Craft: Employs poetic devices and demonstrates structural awareness
5. Originality: Feels fresh and creative, not generic or clichéd

Summary given to the model:
{summary}

Emotional tone: {emotions}

Generated poem:
{response}

Reference poem (for context, not as strict ground truth):
{reference}

Provide scores as: Theme: X, Emotion: X, Imagery: X, Craft: X, Originality: X
Then provide a brief justification.
```

## Notes

- The reference poem is provided to the judge for context but should not be
  treated as the only valid response. Creative divergence from the reference
  is acceptable and even desirable if the result is poetically strong.
- The `our_summary` field (input) is a prose analysis, not a literal prompt
  from the paper. The paper's actual task is poem-to-image generation.
- The `text` field (detailed analysis) is deliberately excluded from the input
  to avoid making the task trivially reconstructive.
- Open-ended metrics (BLEU, ROUGE) will show low scores since valid poems can
  diverge significantly from the reference. The LLM judge scores are the
  primary quality signal.
- Dataset contains 3,008 poems from diverse sources and themes.
