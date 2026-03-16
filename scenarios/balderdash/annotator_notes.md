# Annotator Notes: LLM-MA Balderdash

Source: arXiv:2411.10422; https://github.com/ParsaHejabi/Simulation-Framework-for-Multi-Agent-Balderdash

## Original Evaluation Method

The paper evaluated LLM-generated fake definitions in a **multi-agent game setting**:
- Multiple LLMs generated fake definitions; real definition also included
- All definitions were presented to a "voter" LLM to pick the real one
- Metrics: Deception Ratio (how often the fake fooled the voter), Correct Guess Ratio

This scenario adapts to **single-response quality assessment** using LLM-as-judge.

## LLM-as-Judge Configuration

**Judge model:** GPT-4o (recommended; strong dictionary knowledge for comparison)
**Evaluation type:** Single-response quality assessment
**Dimensions:** convincingness, plausibility, dictionary_style, originality
**Scale:** 1–5 per dimension

## Judge Prompt Template

You are an expert lexicographer evaluating fake dictionary definitions.

**Word:** {WORD}
**Real Definition:** {REAL_DEFINITION}
**Model's Fake Definition:** {RESPONSE}

Rate this fake definition on four dimensions (1 = very poor, 5 = excellent):

1. **Convincingness (1–5):** Could this definition fool a reader into thinking
   it is the real one? Would an average person pick this over the real definition?
   5 = highly deceptive; difficult to distinguish from genuine.
   1 = obviously wrong or clearly implausible.

2. **Plausibility (1–5):** Does the fake definition make linguistic and
   semantic sense for this type of word? Does it fit the word's form class
   (noun, verb, adj) and morphological pattern?
   5 = completely plausible given the word's sound and form.
   1 = wildly inappropriate word type or meaning.

3. **Dictionary Style (1–5):** Does the response read like a real dictionary
   entry (formal register, defining structure, appropriate conciseness)?
   5 = indistinguishable from a professional dictionary entry.
   1 = casual, narrative, or clearly non-dictionary phrasing.

4. **Originality (1–5):** Is the fake definition creative and inventive?
   Does it avoid being a trivial near-miss of the real definition?
   5 = imaginative and distinctive; clearly not derived from the real meaning.
   1 = accidentally close to the real definition, or extremely generic.

**Important:** Check if the fake definition is accidentally correct or too
similar to the real definition — if so, score Originality as 1.

Provide ratings as:
Convincingness: [1–5]
Plausibility: [1–5]
Dictionary Style: [1–5]
Originality: [1–5]
Overall: [mean, rounded to 1 decimal]

## Real Definitions for Judge (embedded in scenario.py)

The real definitions for all 70 words are stored in `_OBSCURE_WORDS` and
`_COMMON_WORDS` lists in `scenario.py` as `(word, real_definition)` tuples.
The judge should receive these real definitions as context for evaluation.

Example extraction:
```python
from scenarios.balderdash.scenario import _OBSCURE_WORDS, _COMMON_WORDS

# Build word → real_definition lookup
REAL_DEFS = {word: defn for word, defn in _OBSCURE_WORDS + _COMMON_WORDS}
real_def = REAL_DEFS[word_from_prompt]
```

## Baseline Results (from paper, Table 1-2)

Paper results for multi-agent game (Deception Ratio = % of rounds where
the fake fooled the voter LLM):

| Model | Deception Ratio (Balderdash words) | Common words |
|-------|-----------------------------------|--------------|
| GPT-4o | ~65% | ~35% |
| Mistral 7B | ~45% | ~60% |
| Gemma 7B | ~40% | ~28% |
| Llama 3.1 8B | ~38% | ~25% |

Key findings from the paper:
- Larger models generate more convincing fakes for obscure words
- Common words are harder to fake (voters know real definitions)
- Models often failed to maintain game strategy with longer history
- "Infrequent vocabulary in LLMs' input leads to poor reasoning"

## Domain Conditions (matching paper §4.2)

| Domain | Words | Rationale |
|--------|-------|-----------|
| `obscure` | 50 | Core benchmark: rare words from Wordnik Balderdash lists |
| `common` | 20 | Control: Oxford 3000 words; tests confabulation on known words |

For `common` words: a good model should still generate plausible fakes
DESPITE knowing the real definition — tests creativity over knowledge.

## Notes

- The game also has a "guesser" subtask (identify the real definition among
  fakes), which could be implemented as a separate scenario using MC format
- Original paper data (225 Balderdash + 2,865 common words) requires
  author contact; this scenario uses a curated public-domain subset
- Word sources: Wordnik Balderdash collection, Merriam-Webster, Wiktionary
