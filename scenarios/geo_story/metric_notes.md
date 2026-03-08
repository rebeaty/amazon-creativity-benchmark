# Metric Notes: Geo-Anchored Story Generation

Source: Paper Section 3; `src/measure_uniqueness.py`, `src/measure_informativeness.py`, `src/extract_emotions.py`

## Metrics from the Paper

### 1. Lexical Uniqueness (IDF-based)

Source: `src/measure_uniqueness.py`

For each word in a response, compute 1 / document_frequency (across all responses
in the corpus). Average across all words in the response, then multiply by
corpus size for normalization.

Higher scores = more unique vocabulary (less overlap with other responses).

This is computed corpus-wide, so it requires collecting all model outputs first,
then scoring as a batch. Not a per-instance metric.

### 2. Informativeness (NER entity count)

Source: `src/measure_informativeness.py`

Count distinct geographic named entities (LOC, FAC, GPE) in each response using
spaCy's transformer-based NER pipeline (`en_core_web_trf`).

Higher counts = more geographically specific and detailed responses.

### 3. Emotion Classification

Source: `src/extract_emotions.py`

GPT-4 classifies each story into 5 emotion categories:
- Joy
- Hardships
- Fear
- Sadness
- Serenity

Prompt: "Help me identify which of the following emotions: Joy, Hardships, Fear,
Sadness, Serenity; are recognized within the story given in the prompt. Only
output the names of the emotions found in the prompt."

Multiple emotions can be assigned per story. The paper analyzes the distribution
of emotions across geographic regions (e.g., Sub-Saharan Africa stories
disproportionately tagged with Hardship/Sadness).

## Key Analysis (Paper Table 1)

Pearson correlation with GDP per capita:
- Uniqueness: 0.27–0.39 (travel), suggesting richer countries get more diverse vocabulary
- Hardship emotion: -0.30 to -0.54 (stories), confirming poorer countries get more hardship narratives

## Notes for HELM Implementation

- All three metrics are post-hoc (computed after generation), not reference-based
- Uniqueness requires corpus-level computation (IDF across all outputs)
- Emotion classification requires an LLM call (GPT-4 in the paper)
- Standard HELM open_ended metrics (BLEU, ROUGE) are not meaningful here since
  there are no reference outputs — this is purely generative
- The most HELM-compatible approach would be LLM-as-judge for creative quality,
  supplemented by the custom IDF uniqueness and NER informativeness metrics
