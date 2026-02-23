# Annotator Notes: PunEval — Pun Generation Task

Source: arXiv:2404.13599; https://github.com/Zhijun-Xu/PunEval (Notebook 6)

## Original Evaluation Method (Generation Task)

The paper evaluates generated pun sentences on five dimensions:
1. **Pun Detection** — LLM-based: is the generated text a valid pun? (binary)
2. **Ambiguity** — word embedding similarity (GloVe/SkipGram); see metric_notes.md
3. **Distinctiveness** — LLM-based synonym-finding; see metric_notes.md
4. **Surprise** — embedding-based unusualness; see metric_notes.md
5. **Unusualness** — language model perplexity; see metric_notes.md

Only **Pun Detection** is implemented as an LLM judge here.
Metrics 2–5 require custom NLP implementations (see metric_notes.md).

## LLM-as-Judge Configuration (Pun Detection)

**Judge model:** GPT-4o (recommended; matches paper's model lineup)
**Evaluation type:** Binary validity check — is the generated sentence a valid pun?
**Dimensions:** pun_valid (binary: pun / non-pun)

## Judge Prompt Template

Verbatim from `evaluate_generation_by_detection()` in Notebook 6:

```
<*Definition*>
Puns are a form of wordplay exploiting different meanings of a word or similar-sounding words,
while non-puns are jokes or statements that don't rely on such linguistic ambiguities.

<*Instruction*>
Determine whether the given Text is a pun. You should either say "The given text is a pun"
or say "The given text is a non-pun". You must output the current status in a parsable JSON
format. An example output looks like:
{"Choice": "The given text is a XXX"}

<*Your Response*>
Text: {RESPONSE}
Output:
```

**Scoring:** Parse `Choice` field from JSON output.
- "The given text is a pun" → 1 (valid pun generated)
- "The given text is a non-pun" → 0 (invalid)

**Primary metric:** Pun Detection Rate = fraction of generated sentences classified as puns.

## Notes

- The paper calls the model TWICE per sample with side="pun" and side="non-pun" to check
  for consistency bias, but for HELM evaluation a single pass (side="pun") suffices.
- The human_text reference is included for BLEU/ROUGE as a secondary metric, but these are
  weak signals — many valid puns exist for each keyword that differ from the human reference.
- Paper Table 3 reports Pun Detection Rate across models:
  GPT-4o ≈ 0.70, Claude-3.5-sonnet ≈ 0.65 for direct generation (Method 1).
