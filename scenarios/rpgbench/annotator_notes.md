# Annotator Notes: RPGBENCH — Game Creation

Source: arXiv:2502.00595; https://github.com/boson-ai/rpgbench-public

## Original Evaluation Method

The paper evaluates Game Creation outputs on two axes:

1. **Structural Validity** — automated JSON schema compliance check
   - Source: `rpgbench/evaluation/game_creation_validity.py`
   - BFS-based analysis verifies game graph reachability (win/lose paths)
   - Binary pass/fail; no LLM needed

2. **Interestingness** — LLM-as-judge quality assessment
   - Source: `rpgbench/static/game_interestingness_prompt.txt`
   - Scale: 1–5 (1 = least interesting, 5 = most interesting)
   - Single dimension; judge returns score + brief explanation

## LLM-as-Judge Configuration

**Judge model:** GPT-4o (recommended; authors use API models for evaluation)
**Evaluation type:** Single-response quality assessment
**Dimensions:** interestingness (single dimension)
**Scale:** 1–5

## Judge Prompt Template

Verbatim from `rpgbench/static/game_interestingness_prompt.txt`:

```
Your task is to evaluate the **interestingness** of the following game content.
Please give a score from 1 (least interesting) to 5 (most interesting), with a
brief explanation of your rationale.


[[start of game content]]
{RESPONSE}
[[end of game content]]

Please return your evaluation score in a json dictionary with the following format:
{"interestingness": <int 1-5>, "explanation": "<string>"}

Example output:
{
  "interestingness": 3,
  "explanation": "The game content is moderately interesting, offering a few unique elements that keep the player engaged."
}
```

**Note:** The judge evaluates the full formatted game (from `format_game()` in
`rpgbench/game/game_utils.py`), not just the raw JSON. Convert the model's JSON
output to readable markdown before passing to the judge.

## Structural Validity (Non-LLM Metric)

The paper's validity check (`game_creation_validity.py`) performs:
- JSON parsing against `game_json_schema.json`
- ID format validation (S###, E###, V###, H###, P###)
- Sequential ordering check
- BFS reachability analysis: confirms at least one win path and one lose path exist
- Unreachable event/scene detection

This can be computed programmatically. A game is "valid" if:
1. JSON parses against the schema
2. `has_succeeded` and `has_failed` hidden variables are present
3. BFS finds at least one winning terminal state

## Game Content Formatting (for judge input)

Use `format_game()` from `rpgbench/game/game_utils.py` to convert raw JSON to
readable markdown before passing to the interestingness judge. The function
produces structured documentation covering world, characters, objectives, scenes,
variables, events, and checks.

## Notes

- Game Simulation task (multi-turn) is NOT evaluated in this scenario.
  Simulation metrics (action quality, factuality, personality consistency,
  mechanics, length) are defined in `rpgbench/evaluation/game_simulation_*.py`
  but require multi-turn gameplay execution.
- The benchmark has no public leaderboard results in the paper (ICLR 2025
  submission); model comparisons are based on validity rate and interestingness
  mean score.
- 100 test instances (Wikipedia character biographies); examples include
  Mickey Mouse, Superman, and other well-known fictional/historical figures.
