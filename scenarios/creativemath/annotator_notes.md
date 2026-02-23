# Annotator Requirements: CreativeMath

Source: src/evaluation.py, src/prompts/prompts.py in https://github.com/JunyiYe/CreativeMath

## 3-Stage Evaluation Pipeline

Original evaluation uses 3 judge models (Claude-3-Opus, Gemini-1.5-Pro,
GPT-4) at each stage. Results are aggregated per stage before proceeding.

### Stage 1: Correctness Evaluation

Judge prompt:
```
Given the following mathematical problem:
{problem}

Reference solutions:
Solution 1:
{solution_1}

Solution 2:
{solution_2}

New solution:
{new_solution}

Please output YES if the new solution leads to the same result as the
reference solutions; otherwise, output NO.
```

- Shows up to 2 reference solutions for context
- All 3 judges must say YES (unanimous) for the solution to pass
- Solutions failing correctness are automatically marked NO for novelty

### Stage 2: Coarse-Grained Novelty Assessment

Only applied to solutions that pass Stage 1.

Judge prompt:
```
Criteria for evaluating the novelty of a new mathematical solution include:
1. If the new solution used to arrive at the solutions is fundamentally
   different from reference solutions, such as algebraic manipulation versus
   geometric reasoning, it can be considered novel;
2. Even if the final results are the same, if the intermediate steps or
   processes involved in reaching those solutions vary significantly, the
   new solution can be considered novel;
3. If the new solution relies on different assumptions or conditions, it
   should be considered novel;
4. A solution might generalize to a broader class of problems, while
   another solution might be specific to certain conditions. In such cases,
   they are considered distinct;
5. If the new solution is significantly simpler or more complex than the
   others, it can be regarded as essentially novel, even if they lead to
   the same result.

Given the following mathematical problem:
{problem}

Reference solutions:
{first_k_solutions}

New solution:
{new_solution}

Please output YES if the new solution is a novel solution; otherwise,
output NO.
```

- Shows the first k reference solutions (same ones the model saw)
- Majority voting across 3 judges (2/3 YES → novel)

### Stage 3: Fine-Grained Novelty Assessment

Only applied to solutions passing Stage 2 where k < n (withheld solutions
exist).

Same prompt template as Stage 2, but reference solutions are the withheld
ones (solutions k+1 through n that the model did NOT see).

- Majority voting across 3 judges
- Tests whether the model genuinely discovered a novel approach vs.
  reproducing a known solution it wasn't shown

## Metrics

- **Correctness Ratio**: fraction passing Stage 1
- **Novelty Ratio**: fraction passing Stage 2 (coarse-grained novel / total)
- **Novel-Unknown Ratio**: fraction passing Stage 3 (fine-grained novel / total)
- **Novelty-to-Correctness Ratio**: Stage 2 passes / Stage 1 passes
- **Novel-Unknown-to-Novelty Ratio**: Stage 3 passes / Stage 2 passes

## Notes for HELM Adaptation

- Original uses 3 separate judge models with voting. HELM's
  `LLMAsJuryAnnotator` could replicate this or simplify to a single judge.
- The sequential dependency (correctness → novelty) requires a multi-stage
  annotator or a combined prompt that evaluates both.
- The paper warns evaluators to "remove transition sentences and
  justifications" from model output before evaluation to avoid bias.
- Output is YES/NO at each stage, making extraction straightforward.
