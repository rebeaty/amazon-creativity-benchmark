# Annotator Requirements: CritiCS Story Plan Evaluation

Source: CritiCS paper (EMNLP 2024, arXiv:2410.02428), evaluation/evaluation_prompt.py

## For Generation Subset (LLM-as-Judge)

### Configuration

Judge model: GPT-4 (paper uses temperature=0)
Format: Pairwise comparison or single-output scoring
Dimensions: Interesting, Coherent, Creative

### Pairwise Comparison Prompt (from repo, persona_comparision template)

```
Here are two storyline excerpts.
You shouldn't be concerned about the completeness of the plot.

Storyline A:
${storyline_A}

Storyline B:
${storyline_B}

Answer the following question:
1) Overall, which story do you prefer/find more interesting? A / B / C
2) Overall, which story has a more coherent overarching plot? A / B / C
3) Overall, which story has a more creative plot? A / B / C
4) Overall, Are both storylines closer to the premise? BY / OA / OB / BN / UN

Metrics explanation:
(1) Interesting: How does the story plan engage and captivate readers, making them find the narrative interesting?
(2) Coherent: How well are the paragraphs organized and connected with each other?
(3) Creative: How does the originality and inventiveness of the storyline offer a fresh perspective compared to typical narratives?
```

### CritiCS Detailed Creativity Rubric (from critics/setting/promptDesign/)

**Originality dimensions:**
- Unconventional Themes
- Unique Plot (unexpected twists, unconventional progression)
- Diverse Settings (unfamiliar locations/times)
- Authenticity

**Structure dimensions:**
- Non-linear timeline
- Shifting perspectives
- Intertextuality
- Metafiction

**Ending dimensions:**
- Unexpected Conclusions
- Humorous or Witty Conclusions
- Provocative or Intriguing Statements

**Text quality dimensions:**
- Image features: Insight, See, Hear, Feel, Body (sensory language)
- Voice features: Informal language, Unusual words, Noteworthy sentence structures, Authenticity

## For Judgment Subset

Uses human annotations from doc-storygen-v2 as ground truth.
No LLM-as-judge needed — this is an exact_match task (predict A or B).

### Human Annotation Protocol (doc-storygen-v2)

- Pairwise comparison: Plan A vs Plan B
- 6 dimensions (Q1-Q6):
  - Q1: Which plot is more interesting?
  - Q3: Which could make a more interesting book or movie?
  - Q4: Which has better suspense or surprise?
  - Q5: Which characters do you identify with more?
  - Q6: Which has a better ending?
- Response options: Plot A / Plot B / Both / Neither
- Judgment subset uses Q1 (interesting) as the primary ground truth label
- "Both" and "Neither" responses filtered out (no clear winner)

### Validation

- CritiCS paper reports "fair to moderate agreement" between GPT-4 judge and human annotators (Cohen's Kappa)
- Presentation order randomized to avoid position bias
