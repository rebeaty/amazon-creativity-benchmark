# Annotator Requirements: ModelingBench

Source: https://github.com/qiancheng0/ModelingAgent/tree/main/src/judger/

## Overview

ModelingBench uses **ModelingJudge** — a multi-dimensional LLM-as-judge framework.
Each submission is scored across 6 dimensions, with 4 of them evaluated from multiple
expert role perspectives. All dimensions use the same discrete scoring scale.

## Scoring Scale (all dimensions)

| Score | Meaning |
|-------|---------|
| 0.00  | Requirement ignored or fundamentally failed |
| 0.25  | Minimal treatment with major flaws |
| 0.50  | Partial/basic treatment addressing main points |
| 0.75  | Strong treatment with minor gaps |
| 1.00  | Comprehensive, thorough implementation |

The framework is intentionally strict — most submissions expected to score 0.25–0.50.

## Dimensions

### 1. Scoring Decomposition (`scoring_decomposition.py`)
Scores each graded requirement individually against the paper's rubric.

- **Input**: generated report + `requirements` list (from `instance.extra_data["requirements"]`)
- **Per-requirement score**: 0.0–1.0 (discrete)
- **Aggregate**: average across requirements
- **Role-based**: No — single judge perspective

**Judge prompt structure:**
- System: "You are an expert judge assessing whether a mathematical modeling report meets grading criteria."
- User: Report content + grading points → return JSON with per-point scores and explanations

### 2. Structural Coherency (`structural_coherency.py`)
Evaluates logical organization and flow of the report across 5 components:

1. Problem Restatement — depth of problem understanding
2. Assumptions and Justification — presence and quality of assumptions
3. Modeling Implementation — mathematical rigor
4. Solution Process — methodology clarity and validation
5. Analysis — interpretive depth and implications

- **Role-based**: No — single judge perspective
- **Aggregate**: average of 5 component scores

### 3. Modeling Groundedness (`modeling_groundedness.py`)
Evaluates mathematical foundations across 5 sub-dimensions:
Mathematical Foundation, Real-World Integration, Technical Sophistication,
Validation Approach, Implementation Quality.

- **Role-based**: Yes — evaluated from each `eval_roles` perspective
- **Aggregate**: average across roles, then average across sub-dimensions

### 4. Data Groundedness (`data_groundedness.py`)
Evaluates how well data collection, processing, and analysis are grounded.

- **Role-based**: Yes — evaluated from each `eval_roles` perspective

### 5. Analysis Groundedness (`analysis_groundedness.py`)
Evaluates evidence-based reasoning and analytical rigor.

- **Role-based**: Yes — evaluated from each `eval_roles` perspective

### 6. Innovativeness (`innovativeness.py`)
Evaluates novelty across 5 sub-dimensions:
Methodological Innovation, Problem Framing, Solution Creativity,
Technical Advancement, Impact Potential.

- **Role-based**: Yes — evaluated from each `eval_roles` perspective
- **Note**: "True innovation is rare" — scores expected to be very low

## Expert Roles (problem-specific)

Each problem in ModelingBench has 3–4 expert evaluator roles stored in
`instance.extra_data["eval_roles"]`. Each role has:
- `name`: role title (e.g., "Mathematician", "Epidemiologist")
- `thoughts`: high-level evaluation perspective
- `details`: specific evaluation guidelines for this problem

**Example roles for 2001_Adolescent_Pregnancy:**
- Mathematician, Data Scientist, Epidemiologist, Social Scientist

The role-based dimensions (3–6) must be run once per role and results averaged.

## Aggregation

Final score per instance:
```
final_score = mean([
    scoring_decomposition,
    structural_coherency,
    mean_over_roles(modeling_groundedness),
    mean_over_roles(data_groundedness),
    mean_over_roles(analysis_groundedness),
    mean_over_roles(innovativeness),
])
```

## Implementation Notes

- Judge model used in paper: GPT-4o (inferred from codebase)
- Parallelism: original code uses `ThreadPoolExecutor` for role-based calls
- Each full evaluation = 1 (decomposition) + 1 (structural) + 4×N_roles (role-based) LLM calls
  - With 4 roles: ~18 judge calls per instance, ~198 total for the 11-problem dataset
- Output format: JSON with per-dimension scores and explanations
- Requirements and eval_roles are available in `instance.extra_data` for the annotator
