# Evaluation Metrics: GAUSS

## Standard NLG Metrics (Implemented in HELM)

Use `get_open_ended_generation_metric_specs()` for:
- **BLEU-1, BLEU-4**: N-gram overlap with standard solutions
- **ROUGE-L**: Longest common subsequence
- **F1**: Token-level F1 score

## LLM-as-Judge Evaluation (Recommended)

Each problem in GAUSS includes expert-written rubrics with scoring criteria. This enables rubric-based evaluation using HELM's `LLMAsJuryAnnotator`.

### Rubric-Based Scoring

**General Format**: Problems are scored on a point scale (typically 1-3 points) based on specific criteria.

**Example Creativity Problem (12a - Massive SLE)**:
- Total Score: 3 points
- Rubric: "Award 1 point for providing one meaningful property and question, up to a maximum of 3 points."

**Example Creativity Problem (12b - 1977 IMO Multiple Solutions)**:
- Total Score: 2 points
- Rubric: "Award 1 point for one correct solution, up to maximum of 2 points."

**Example Creativity Problem (12c - Move One Digit Puzzle)**:
- Total Score: 1 point
- Rubric: "Award 1 point if the solution explicitly contains the equation 2^6 - 63 = 1."

### Judge Prompt Template

```
Evaluate the following mathematical solution using the provided rubric.

Problem: {PROBLEM_STATEMENT}

Student Solution: {MODEL_RESPONSE}

Standard Solution (for reference): {STANDARD_SOLUTION}

Rubric: {RUBRIC}

Total Possible Score: {TOTAL_SCORE} points

Based on the rubric, assign a score from 0 to {TOTAL_SCORE} points.
Provide your score as a single number.
```

### Implementation Notes

- **Judge model**: GPT-4 or Claude (dataset includes GPT-5-Thinking responses for reference)
- **Scoring**: Variable point scales per problem (1-3 points typical)
- **Rubric specificity**: Each problem has detailed, criterion-referenced rubrics
- **Dataset note**: Includes 41 problems across 12 dimensions; creativity dimension has 3 problems

### Alternative: Human Expert Evaluation

The original GAUSS benchmark uses human mathematicians and researchers to:
1. Create problems and standard solutions
2. Write detailed rubrics
3. Score model responses

For research purposes, human expert evaluation following the rubrics would provide the most accurate assessment.

## Domain-Specific Considerations

### Mathematical Notation

Solutions often include LaTeX mathematical notation. Evaluation should account for:
- Correct mathematical formulation
- Proper notation and terminology
- Logical structure of proofs/derivations

### Creativity Dimension Specifics

Creativity problems (dimension 12) evaluate:
- **Novel approaches**: Finding multiple solutions or alternative methods
- **Open-ended exploration**: Defining new concepts and exploring properties
- **Creative problem-solving**: Lateral thinking in mathematical puzzles

These require more nuanced evaluation than pure correctness checking.

## Recommendation

For HELM integration:
1. **Primary**: Use standard open-ended generation metrics (BLEU, ROUGE, F1) as baseline
2. **Enhanced**: Implement LLM-as-judge with problem-specific rubrics for more accurate assessment
3. **Gold standard**: Human expert evaluation for high-stakes assessment

The rubric-based approach is particularly important for creativity problems where multiple valid solutions exist.
