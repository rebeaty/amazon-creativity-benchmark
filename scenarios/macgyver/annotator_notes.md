# Annotator Requirements: MACGYVER

Source: Paper Section 4.2, Benchmark Results (data/Benchmark_results/benchmark_results.json)

## Evaluation Type

`llm_judge` or `open_ended` with human annotation

The paper uses human annotators to evaluate model-generated solutions on a spectrum
of correctness. Automatic evaluation (BLEU/ROUGE against gold solutions) can be
used as a baseline, but the paper emphasizes that human judgment is needed for
accurate assessment.

## Annotation Categories

Solutions are annotated into the following categories:

| Category | Description |
|----------|-------------|
| `correct_efficient` | Correct solution using efficient approach |
| `correct_inefficient` | Correct but uses more steps/resources than necessary |
| `correct_unsolvable` | Correctly identifies problem as unsolvable |
| `correct_right_reason` | Correct with proper reasoning |
| `correct_wrong_reason` | Correct answer but flawed reasoning |
| `wrong_partial_correct` | Partially correct solution |
| `wrong_solution` | Completely incorrect solution |
| `infeasible` | Proposed solution is not physically feasible |

## Judge Prompt Template (Suggested)

```
You are evaluating a solution to a creative problem-solving task.

**Problem:**
{PROBLEM}

**Proposed Solution:**
{RESPONSE}

**Gold Solution (for reference):**
{GOLD_SOLUTION}

Evaluate the proposed solution on the following dimensions:

1. **Correctness**: Does the solution actually solve the problem?
2. **Feasibility**: Are all steps physically possible with the given tools?
3. **Efficiency**: Does the solution use a reasonable number of steps?
4. **Tool Usage**: Are the tools used appropriately?

For unsolvable problems, check if the response correctly identifies it as unsolvable.

Rate the solution:
- 5: Perfect (correct and efficient)
- 4: Correct but inefficient
- 3: Partially correct
- 2: Correct identification of unsolvability (for unsolvable problems)
- 1: Wrong but shows some understanding
- 0: Completely wrong or infeasible

Provide your rating as a single number.
```

## Notes

- The benchmark includes both solvable (1,306) and unsolvable (377) problems
- For unsolvable problems, the expected answer is to identify infeasibility
- Human-model agreement in original study: GPT-4 achieved highest scores
- 4,700 human-annotated solution-annotation pairs available in benchmark_results.json
- Consider using the human annotations as training data for automatic evaluation

## Recommended HELM Configuration

For basic evaluation without LLM judge:
```yaml
metric_specs:
  - get_open_ended_generation_metric_specs()  # BLEU-1, BLEU-4, ROUGE-L
```

For LLM-as-judge evaluation:
```yaml
adapter_spec:
  method: generation
annotator_specs:
  - class_name: helm.benchmark.annotation.llm_jury_annotator.LLMAsJuryAnnotator
    args:
      dimensions: [correctness, feasibility, efficiency]
      judge_model: gpt-4-turbo
```
