# Annotator Requirements: LiveIdeaBench

Source: utils/prompts.json in https://github.com/x66ccff/liveideabench

## Configuration for LLMAsJuryAnnotator

Judge models (paper's CRITIC_MODELS): claude-3.5-sonnet, gpt-4o, qwen-2.5-72b, deepseek-chat, gemini-2.0-flash-thinking
Dimensions: originality, feasibility, clarity (scored individually)
Scale: 1-10 per dimension

## Critic Prompt Template (exact from utils/prompts.json)

```
You are an extremely demanding scientific reviewer with the highest critical standards, like those at Nature or Science. When evaluating scientific ideas, you will assess them on three key dimensions:

1. originality: Novel contribution to unexplored areas or innovative approaches to existing problems
2. feasibility: Technical implementation and practicality
3. clarity: How well-articulated and easy to understand the idea is

Your response should consist of two parts: a text analysis followed by a JSON score block.

First, provide your brief analysis (less than 100 words) of the idea. Then, for each dimension, provide a score from 1 to 10 where 1-3 = poor, 4-6 = average, 7-10 = excellent.

For example:
```json
{
    "originality": <score_1_to_10>,
    "feasibility": <score_1_to_10>,
    "clarity": <score_1_to_10>
}
```
```

## Fluency Evaluation (Pairwise Comparison)

Fluency is evaluated via pairwise comparison of ideas from the same keyword.
A separate prompt compares two ideas and rates similarity as A/B/C/D:

```
Here are two ideas submitted to "Good Scientific Ideas" Competition, which both relate to "{keyword}"

# The first idea

{A}

# The second idea

{B}

# Question

Evaluate the similarity between these two ideas that both relate to "{keyword}". Please choose the best answer:

A. Completely different ideas addressing different problems, despite relating to the same keyword.
B. Different ideas but addressing similar problems.
C. Similar ideas addressing similar or identical problems.
D. Academically identical ideas with the same core approach and problem statement.

ONLY ANSWER A/B/C/D, DO NOT EXPLAIN
```

Fluency scoring: A=10, B=7, C=4, D=1 (inferred from paper's 4-point scale mapping)

## Flexibility

Flexibility = 30th percentile of other scores (originality, feasibility, clarity).
Computed post-hoc, not from judge evaluation.

## Notes

- Paper uses dynamic panel of top-10 LiveBench models as judges to reduce individual bias
- Human expert validation conducted on subset; correlation with LLM judges reported
- 1,180 keywords across 22 scientific domains
- Keywords updated monthly to prevent contamination (the "live" aspect)
