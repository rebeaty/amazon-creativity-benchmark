# Annotator Requirements: MoPS Premise Evaluation

Source: Paper Section 4 (Evaluation Metrics), arXiv:2406.05690

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4-turbo
Dimensions: fascination, completeness, originality
Scale: 0–100 per dimension

## Judge Prompt Templates

The paper describes three quality dimensions evaluated by GPT-4-turbo. Exact prompts
are not fully reproduced in the paper; the following templates reflect the definitions
given in Section 4.

### Fascination
Rate the following story premise on fascination (how interesting and engaging it is)
from 0 to 100, where 0 is completely uninteresting and 100 is highly captivating.

Story Premise: {RESPONSE}

Provide your rating as a single integer.

### Completeness
Rate the following story premise on completeness (whether it includes a character,
setting, event, ending, and twist) from 0 to 100, where 0 is missing most components
and 100 means all narrative components are clearly present.

Story Premise: {RESPONSE}

Provide your rating as a single integer.

### Originality
Rate the following story premise on originality (how novel and unfamiliar it feels,
avoiding clichés and memorized patterns) from 0 to 100, where 0 is entirely
derivative and 100 is highly original.

Story Premise: {RESPONSE}

Provide your rating as a single integer.

## Notes

- Paper uses GPT-4-turbo for automated scoring; human evaluation used for validation
- Final quality score can be computed as the average of the three dimension scores
- The paper also measures diversity across a set of premises (breadth + density via
  2D semantic embeddings); these are set-level metrics, not per-instance, and are
  not implemented here
- Human correlation with automated scores is reported as high in the paper (see Table 2)
