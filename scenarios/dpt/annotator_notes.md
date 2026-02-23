# Annotator Notes: Design Problems Task (DPT)

Source: arXiv:2502.03253 (CogSci 2025); https://github.com/Beaty-Lab/CogSci-2025-Scientific-Creativity

## Original Evaluation Method

The paper used **expert human raters** (80 STEM-degreed participants from Prolific):
- Rated solutions on 5 Likert dimensions (1–5 scale)
- Each rater evaluated ~15 solutions
- Final dataset: 830 rated responses in `data/cleaned_data_explanations_gold.csv`
- Raters also provided 1–2 sentence free-text explanations justifying their originality scores

## LLM-as-Judge Configuration

**Judge model:** GPT-4o (recommended; outperformed humans in rating consistency per paper)
**Evaluation type:** Single-response quality assessment
**Dimensions:** originality, cleverness, uncommonness, effectiveness, conciseness
**Scale:** 1–5 per dimension

## Judge Prompt Template

You are an expert evaluator of scientific and engineering creativity.

**Design Problem:** {PROBLEM}
**Proposed Solution:** {RESPONSE}

Rate this solution on five dimensions (1 = very poor, 5 = excellent):

1. **Originality (1–5):** Is this solution novel and non-obvious?
   Would most people NOT think of this approach?
   5 = genuinely surprising, distinctive angle.
   1 = generic or predictable (e.g., "use solar panels" for renewable energy).

2. **Cleverness (1–5):** Does the solution show creative thinking or ingenuity?
   Does it reveal an insightful connection or inventive mechanism?
   5 = clever and witty; shows creative insight.
   1 = straightforward or mechanical.

3. **Uncommonness (1–5):** How rarely would this solution appear across many responses?
   5 = rare approach; few others would generate this.
   1 = common trope or frequently proposed idea.

4. **Effectiveness (1–5):** Is the solution practically feasible?
   Could it plausibly address the problem if implemented?
   5 = clearly actionable and likely effective.
   1 = impractical, vague, or clearly infeasible.

5. **Conciseness (1–5):** Does the response clearly describe a solution in 2–4 sentences?
   5 = well-scoped, clear, and within length.
   1 = too vague, too long, or incomplete.

Provide ratings as:
Originality: [1–5]
Cleverness: [1–5]
Uncommonness: [1–5]
Effectiveness: [1–5]
Conciseness: [1–5]
Overall: [mean, rounded to 1 decimal]

## Human Calibration Data

The repo provides 830 human-rated responses for judge calibration:

**File:** `data/cleaned_data_explanations_gold.csv`
**URL:** https://github.com/Beaty-Lab/CogSci-2025-Scientific-Creativity/blob/main/data/cleaned_data_explanations_gold.csv

**Fields:**
| Field | Description |
|-------|-------------|
| `problem` | Design problem text |
| `response` | Proposed solution |
| `originality` | Human originality rating (1–5) |
| `cleverness` | Human cleverness rating (1–5) |
| `uncommonness` | Human uncommonness rating (1–5) |
| `remoteness` | How far from conventional approaches (1–5) |
| `explanation` | 1–2 sentence justification from human rater |
| `condition` | "oracle" (saw examples) or "no_oracle" |

Use this data to:
1. Calibrate judge ratings against human consensus
2. Compute inter-rater reliability (human vs. LLM judge Pearson r)
3. Identify systematic biases (paper found LLMs collapse 4 dimensions to r≈0.99)

## Baseline Results (from paper)

| Rater | Pearson r (vs. human consensus) | Notes |
|-------|--------------------------------|-------|
| Human (no-oracle) | r = 0.44 | No example responses shown |
| Human (oracle) | r = 0.47 | Example responses shown first |
| GPT-4o-mini | r = 0.74–0.76 | Higher consistency, dimension collapse |
| Claude-3.5-haiku | r ≈ 0.74 | Similar pattern to GPT |

**Key finding:** LLMs rate more consistently than humans but collapse distinct
creativity dimensions (originality, cleverness, uncommonness, remoteness all
become highly correlated at r ≈ 0.99), suggesting they don't distinguish between
these conceptually separate aspects of creativity.

## Dimension Notes

- **Remoteness** (paper) ≈ **Uncommonness** (this rubric): both measure how
  far the solution departs from conventional approaches
- Paper's primary metric was `originality_rescaled_factor` (1–5 scale)
- `effectiveness_rescaled_factor` is the secondary metric
- `cleverness`, `uncommonness`, `remoteness` are secondary dimensions in the
  explanations dataset

## Task Domains Reference

| Domain | Problems | Example |
|--------|----------|---------|
| Accessibility | 5 | "Help people with hearing impairments participate in group conversations" |
| Transportation | 3 | "Reduce traffic congestion in mega cities" |
| Environment | 8 | "Reduce the environmental impact of air travel" |
