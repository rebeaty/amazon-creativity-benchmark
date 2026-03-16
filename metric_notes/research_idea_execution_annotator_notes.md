# Annotator Notes: Research Idea Peer Review

Source: Paper Appendix A (arXiv:2506.20803); expert review data in
`reviews_ideation/data_points_all_anonymized.json` (398 reviews, GitHub)

## Task

The model generates a peer review of a structured NLP research idea.
The LLM judge evaluates whether the generated review is:
- Accurately calibrated (scores align with the idea's actual quality)
- Well-reasoned (rationales are specific and substantive, not generic)
- Appropriately critical (identifies genuine weaknesses, not just praise)
- Consistent across dimensions (scores are internally coherent)

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4o (or GPT-4-turbo)
Evaluation type: Open-ended review quality assessment
Dimensions: calibration, reasoning_quality, specificity, coherence
Scale: 1–5 per dimension

## Judge Prompt Template

You are evaluating the quality of a peer review written by an AI system for
a research proposal.

**Research Idea:**
{IDEA_TEXT}

**AI-Generated Review:**
{RESPONSE}

**Expert Review Examples (for calibration):**
Expert reviews from the original study assigned novelty scores of 3–8 out of
10. High-quality reviews (score 7–8) cited specific related papers and
identified concrete methodological contributions. Low-quality reviews (score
3–4) noted the idea was incremental without explanation.

Rate the AI-generated review on:

1. **Calibration (1–5):** Do the numeric scores (novelty, excitement,
   feasibility, etc.) seem appropriate for this type of research idea?
   5 = scores are well-calibrated and differentiated across dimensions.
   1 = all scores identical, or clearly wrong (e.g., 10/10 for a trivial idea).

2. **Reasoning Quality (1–5):** Are the rationales specific and substantive?
   5 = cites concrete related work or methodological details.
   1 = generic praise/criticism without specifics ("this is novel" with no explanation).

3. **Specificity (1–5):** Does the review address the specific idea content
   (problem, method, experiment plan) rather than giving generic feedback?
   5 = references specific aspects of the proposed method or experiment design.
   1 = could apply to any paper.

4. **Coherence (1–5):** Are the five dimension scores internally consistent?
   5 = high novelty + high excitement but lower feasibility is internally logical.
   1 = contradictory scores (e.g., feasibility=9 but experiment_plan is "unclear").

Provide your ratings as:
Calibration: [1–5]
Reasoning Quality: [1–5]
Specificity: [1–5]
Coherence: [1–5]
Overall Judge Score: [average]

## Expert Review Calibration Data

From `reviews_ideation/data_points_all_anonymized.json` (398 records):
- Mean novelty score: ~5.2 / 10
- Score distribution: most ideas fall 4–7; extremes (1–3 or 8–10) are rare
- Typical high-quality rationale: "This approach is fairly novel in the
  intersection of safety and prompting. For instance, though there exist
  papers on prompt injection, there does not exist papers that focus on
  injecting 'misdirection' in prompts..."
- Typical low-quality rationale: "Multiple papers for jailbreak protection
  investigate the usage of prompt optimization. This paper seems more like
  a baseline approach..."

NOTE: `data_points_all_anonymized.json` records only `novelty_score` and
`novelty_rationale`. The full 5-dimension rubric scores are in the execution
study data zip (Execution_Study_Data.zip, Google Drive). The judge prompt
above covers all 5 dimensions using the rubric from Appendix A.

## Five Evaluation Dimensions (from Paper Appendix A)

| Dimension | Description | Scale |
|-----------|-------------|-------|
| Novelty | Originality vs. existing work | 1–10 |
| Excitement | Expected impact and interest to the community | 1–10 |
| Feasibility | Realism of implementation given current methods | 1–10 |
| Expected Effectiveness | Likelihood of achieving stated goals | 1–10 |
| Overall | Holistic quality assessment | 1–10 |

## Notes

- The paper found AI-generated ideas score ~1 point higher on novelty at
  ideation time but produce significantly worse execution outcomes — a useful
  calibration reference for the judge.
- Human expert reviewers had an average h-index of 7 and 15 publications,
  representing senior PhD / early postdoc level expertise.
- The judge should flag reviews that only assign scores without rationale
  (a known failure mode for LLMs asked to review research).
