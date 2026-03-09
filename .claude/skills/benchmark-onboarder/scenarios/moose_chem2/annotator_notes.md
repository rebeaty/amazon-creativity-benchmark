# Annotator Requirements: MOOSE-Chem2

Source: `Evaluation/evaluate.py`, `Evaluation/pairwise_compare.py`, `Method/utils.py`
in the MOOSE-Chem2 GitHub repo (https://github.com/ZonglinY/MOOSE-Chem2)

## Overview

Two complementary evaluation approaches are used in the paper:

1. **Component Recall** (primary): Break hypothesis into technical components, compare
   coverage against ground-truth components. Reports Soft Recall and Hard Recall.
2. **Pairwise Comparison** (secondary): Rank two hypotheses head-to-head on 5 dimensions.

---

## Approach 1: Component-Based Recall

### Step 1 — Component Extraction Prompt

The judge first decomposes a hypothesis into discrete technical components.

**Judge model**: GPT-4o-mini (original paper default)

**Component Extraction Prompt** (from `Method/utils.py → evaluation_instruction_prompts('break_finegrained_hyp_or_exp')`):

```
Given a scientific hypothesis, extract all specific technical components.

Include ONLY:
- Explicitly named chemicals or materials (e.g., "guanidine sulfate (Gdm)2SO4")
- Clearly stated functional groups or molecular structures
- Explicitly described chemical reactions or mechanisms
- Explicitly stated reaction conditions (temperature, pressure, concentrations)

Do NOT include:
- Experimental outcomes or performance evaluations
- General descriptions of benefits or applications
- Broad conceptual mechanisms without specific details
- Background information or motivation

Format each component as:
Id of the component: <number>
Component: <specific technical detail>
```

Apply this prompt to both the **generated hypothesis** and the **ground-truth hypothesis**
(from the `Finegrained Hypothesis` column in the reference) to get two component lists.

### Step 2 — Coverage Scoring Prompt

For each ground-truth component, score how well it is covered by the generated components.

**Coverage Scale** (4-level, from `Evaluation/evaluate.py`):
- **3** — Complete coverage: the generated hypothesis fully captures this component
- **2** — Partial coverage: the generated hypothesis partially addresses this component
- **1** — Minimal coverage: the generated hypothesis barely mentions this concept
- **0** — No coverage: this component is entirely absent from the generated hypothesis

**Coverage Prompt**:
```
Given the following ground-truth component and a list of generated components,
score how well the ground-truth component is covered by the generated components.

Ground-truth component:
{GROUNDTRUTH_COMPONENT}

Generated components:
{GENERATED_COMPONENTS_LIST}

Score (0-3):
- 3: Fully covered — the generated components explicitly contain this information
- 2: Partially covered — relevant but incomplete
- 1: Minimally covered — barely mentioned
- 0: Not covered — absent entirely

Respond with only the integer score (0, 1, 2, or 3).
```

### Step 3 — Metric Computation

From coverage scores across all ground-truth components:

- **Soft Recall**: fraction of ground-truth components with coverage > 0
  `soft_recall = count(coverage > 0) / total_groundtruth_components`

- **Hard Recall**: fraction of ground-truth components with full coverage (= 3)
  `hard_recall = count(coverage == 3) / total_groundtruth_components`

- **Weighted Recall**: weighted mean coverage normalized to [0, 1]
  `weighted_recall = mean(coverage) / 3.0`

Report all three. Primary metric for ranking: **Soft Recall**.

---

## Approach 2: Pairwise Comparison

**Judge model**: GPT-4o-mini

**What is passed to the judge**:
- Research question (`Background Question` field from instance input)
- Hypothesis A (model output)
- Hypothesis B (comparison target — e.g., ground truth or another model's output)

**Dimensions** (5 total, from `Evaluation/pairwise_compare.py`):
1. **Overall** — general quality
2. **Effectiveness** — likelihood of advancing the research goal
3. **Novelty** — originality relative to known literature
4. **Detailedness** — specificity and actionability of the hypothesis
5. **Feasibility** — practical lab testability

**Pairwise Prompt Template**:
```
You are an expert chemistry researcher. Compare the following two scientific hypotheses
for the given research question.

Research Question:
{RESEARCH_QUESTION}

Hypothesis A:
{HYPOTHESIS_A}

Hypothesis B:
{HYPOTHESIS_B}

Which hypothesis is better in terms of {DIMENSION}?
- Output "A" if Hypothesis A is better
- Output "B" if Hypothesis B is better
- Output "Tie" if they are roughly equal

Answer with only "A", "B", or "Tie".
```

**Result aggregation**: Win rate across all pairs for each dimension.

---

## Implementation Notes

- The component extraction prompt is the most critical step — errors here propagate to
  recall scores. Consider running extraction twice and taking the intersection for stability.
- Paper uses GPT-4o-mini as the judge throughout. GPT-4 would give more reliable results
  but is more expensive for 51 instances × N components each.
- Ground-truth `Finegrained Hypothesis` entries are already highly detailed (~500-1500 words);
  expect typical models to generate shorter outputs with lower recall.
- Human expert evaluation (paper Tables 3-5) shows GPT-4o-mini judge correlates reasonably
  with expert rankings on Detailedness and Feasibility dimensions.

## Column Reference (chem_research_2024_finegrained.xlsx)

| Column | Role |
|--------|------|
| Background Question | Input: research problem to solve |
| Background Little Survey | Input: prior-work context |
| Main hypothesis | Input: coarse-grained direction hint |
| Finegrained Hypothesis | Reference: ground-truth fine-grained hypothesis |
| Finegrained Experiment | Optional: ground-truth experiment protocols |
