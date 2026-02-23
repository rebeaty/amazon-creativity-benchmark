# Annotator Notes: Fann or Flop — Arabic Poetry Explanation

Source: arXiv:2505.18152; https://github.com/mbzuai-oryx/FannOrFlop (evaluation/judge_eval.py)

## Original Evaluation Method

The paper evaluates generated verse explanations using:
1. **LLM-as-Judge (GPT-4o)** — faithfulness, fluency, overall (1-5 per poem)
2. **BERTScore** — via AraBERT (semantic similarity vs. gold)
3. **BLEU + chrF++** — lexical overlap vs. gold `raw_explanation`
4. **Textual Entailment** — bidirectional NLI via mDeBERTa

Only the LLM judge and standard metrics (BLEU/BERTScore) are HELM-compatible.

## LLM-as-Judge Configuration

**Judge model:** `gpt-4o-2024-08-06`
**Temperature:** 0.0
**Evaluation type:** Poem-level holistic assessment
**Dimensions:** faithfulness, fluency, overall
**Scale:** 1–5 per dimension

## Judge System Prompt

From `evaluation/judge_eval.py` (verbatim rubric):

```
You are an expert Arabic linguist and literary critic. You will be given an Arabic poem
along with a ground-truth verse-by-verse explanation and a model-generated explanation.
Evaluate the generated explanation holistically across all verses and return a JSON object
with three scores (1-5 each):

- faithfulness_score: Does the generated explanation faithfully convey the meaning of
  each verse?
  5 = Deeply faithful, captures poetic imagery and precise meaning
  3 = Generally aligned but loses some nuance or imagery
  1 = Misinterprets verse meaning or invents content

- fluency_score: Is the generated Arabic well-formed Modern Standard Arabic?
  5 = Fluent, grammatically correct, natural MSA
  3 = Understandable but with minor grammatical issues
  1 = Awkward, incomplete, or ungrammatical

- overall_score: Holistic quality assessment combining faithfulness, fluency, and
  interpretive depth.
```

**Judge input format** (pairs ground-truth and generated explanations by verse index):

```
Ground Truth Explanations:
Verse 1: {gt_verse_explanation_1}
Verse 2: {gt_verse_explanation_2}
...

Generated Explanations:
Verse 1: {pred_verse_explanation_1}
Verse 2: {pred_verse_explanation_2}
...
```

**Note:** The judge compares at verse level but scores at poem level. Ground truth comes
from the `explanation` field (list of `{verse, explanation}` dicts). Generated output
should be parsed into the same verse-aligned structure before judging.

## Expected Output Format

For LLM judge evaluation, the model's response should ideally align with the verse
structure from the gold `explanation` list. If using `raw_explanation` as the reference
for BLEU/BERTScore, the model can output free-form Arabic text.

## Benchmark Results (Paper, Table — top models)

| Model | Faithfulness | Fluency | Overall | BERTScore | Entailment |
|-------|-------------|---------|---------|-----------|------------|
| Gemini-2.5-Flash | 4.25 | — | — | — | 0.7475 |
| GPT-4o | — | — | — | 0.6410 | — |
| Human (Interpretive Depth) | 7.52/10 | — | — | — | — |

## Notes

- The dataset has only a `train` split (no test labels); all 6,984 instances are used.
- Dataset on HuggingFace: `omkarthawakar/FannOrFlop` (publicly accessible)
- Arabic text uses full diacritical marks (tashkeel); judge should handle this.
- Judge uses `pydantic` structured output for reliable JSON parsing in the original code.
- Poetry spans 12 eras from Pre-Islamic (~6th c.) to Modern (19th c.–present).
