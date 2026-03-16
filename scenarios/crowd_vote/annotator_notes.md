# Annotator Notes: Crowd Vote Marketing Creativity Benchmark

Source: arXiv:2509.09702 (CC-BY-4.0); https://creativitybenchmark.ai/

## Original Evaluation Method

The Springboards study used **pairwise crowd voting**:
- 678 practising advertising professionals
- 11,012 head-to-head comparisons across 16 AI systems
- Bradley-Terry model applied to pairwise votes for final ranking
- No rubric; preference only (A vs. B, with draw and skip options)

## LLM-as-Judge Configuration

Since pairwise crowd voting by ad professionals is not replicable at scale,
this scenario uses LLM-as-judge to evaluate single responses on four dimensions.

**Judge model:** GPT-4o (recommended for advertising domain reasoning)
**Evaluation type:** Single-response quality assessment (not pairwise)
**Dimensions:** originality, brand_relevance, creative_potential, conciseness
**Scale:** 1–5 per dimension

## Judge Prompt Template

You are an expert advertising creative director evaluating a marketing concept.

**Brand:** {BRAND}
**Category:** {CATEGORY}
**Task Type:** {TASK_TYPE}
**Model Response:** {RESPONSE}

Rate this marketing response on four dimensions (1 = very poor, 5 = excellent):

1. **Originality (1–5):** Is this insight/idea surprising and non-obvious?
   Does it avoid clichés? Would an experienced ad creative find this fresh?
   5 = genuinely unexpected angle, would turn heads in a pitch.
   1 = generic, predictable, or a common trope for this brand/category.

2. **Brand Relevance (1–5):** Does this connect meaningfully to the specific
   brand rather than being interchangeable with any brand in the category?
   5 = clearly rooted in what makes this brand distinctive.
   1 = could apply to any brand; no brand-specific insight.

3. **Creative Potential (1–5):** Could this plausibly become a real campaign?
   Does it suggest a compelling creative platform or activation?
   5 = immediately evokes a campaign; rich with executional possibilities.
   1 = abstract, vague, or not actionable as a campaign idea.

4. **Conciseness (1–5):** Does the response respect the word limit?
   - Insights: ≤10 words required
   - Ideas / Wild Ideas: ≤50 words required
   5 = within the limit, tightly written.
   1 = far exceeds the limit, or is incomplete/truncated.

Provide your ratings as:
Originality: [1–5]
Brand Relevance: [1–5]
Creative Potential: [1–5]
Conciseness: [1–5]
Overall: [average, rounded to 1 decimal]

## Task Type Reference (from paper)

| Type | Prompt | Limit |
|------|--------|-------|
| Insights | "What is a surprising insight about people, culture, category, or product that [BRAND] could build a campaign around?" | ≤10 words |
| Ideas | "Propose a big, campaignable platform idea for [BRAND]. Based on a strategic or cultural truth, works across any channel." | ≤50 words |
| Wild Ideas | "What is your wildest unconventional campaign idea for [BRAND], something no traditional agency would dare present?" | ≤50 words |

## Baseline Results (from paper, original 16-model study)

- Models were tightly clustered in performance
- GPT-4o and Claude Sonnet were top performers
- Wild Ideas showed highest variance between models
- Insights task showed smallest differentiation between models
- Human professionals preferred responses with concrete cultural observations
  over abstract brand claims

## Notes

- The specific 100 brands used in the original Springboards study are not
  publicly disclosed. This scenario uses curated well-known brands matching
  the paper's 12-category taxonomy.
- Original evaluation was pairwise (not absolute scores), so per-instance
  LLM judge scores are not directly comparable to the paper's rankings.
- For meaningful comparison across models, run all models on identical
  brand/task combinations and apply Bradley-Terry to pairwise judge votes.
