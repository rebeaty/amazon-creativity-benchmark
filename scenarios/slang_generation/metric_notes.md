# Metric Notes: Slang Generation Evaluative Framework

Source: https://github.com/siyangwu1/LLM-Slang-Dictionary/blob/main/code/novelty.py
        https://github.com/siyangwu1/LLM-Slang-Dictionary/blob/main/code/coinage_coherence.py

## Primary Metric: Semantic Novelty

Measures how semantically distant the generated slang definition is from
standard (non-slang) dictionary definitions of the same word.

**Implementation (novelty.py):**
1. Encode the generated slang definition with SentenceBERT (`all-mpnet-base-v2`)
2. Encode standard dictionary definitions for the generated word
   (e.g., from WordNet or a standard English dictionary)
3. Compute mean Euclidean distance between slang embedding and standard embeddings
4. Higher distance = more semantically novel / creative slang

**Interpretation:** A slang term is "novel" if its definition diverges from
the conventional meaning of that word. E.g., "sick" meaning "excellent" is
highly novel because its slang definition is distant from standard definitions.

## Secondary Metric: Morphological Coherence (Coinage mode only)

Measures whether a coined word is morphologically plausible (recognizable
sub-morphemes, not random strings).

**Implementation (coinage_coherence.py):**
1. Train a Morfessor model on a large English word list
2. Segment the generated word into morphemes
3. Score coherence based on morpheme recognizability and structure
4. Higher score = more morphologically coherent invented word

**Applicable only when prompting for coinage** (model asked to invent a new word).

## BLEU/ROUGE (Soft Proxy)

Standard HELM open_ended metrics (BLEU-1, BLEU-4, ROUGE-L) are computed
against the gold slang term from conv_slang.txt, but **should be interpreted
cautiously**: many valid slang terms exist for any definition, so low BLEU
against the gold does not indicate failure.

## Evaluation Notes

- Both primary metrics are **reference-free** — they do not compare to the
  gold slang term, and instead measure intrinsic properties of the output.
- The scenario stores the gold term as a CORRECT_TAG reference to enable
  standard HELM metric computation; semantic novelty is the authoritative metric.
- Model output should be parsed to extract the generated slang word (field 1)
  before running novelty and coherence metrics.
