# Metric Notes: Future Research Idea Generation Benchmark

Source: https://arxiv.org/abs/2409.06185 (Section 4 — Evaluation Metrics)
        https://github.com/sandeep82945/Future-Idea-Generation/tree/main/code

## Primary Metrics (proposed in the paper)

### 1. Idea Alignment Score (IAScore)

Measures semantic similarity between a generated idea and the author's
gold future work section.

**Implementation:**
1. Encode the generated ideas and gold future work using SentenceBERT
   (or similar sentence embedding model)
2. Compute cosine similarity between generated and reference embeddings
3. Score = mean cosine similarity across idea-reference pairs

Higher score = ideas are more aligned with what the authors intended.

### 2. Idea Distinctness Index (IDI)

Measures the diversity/novelty of the generated ideas — penalizes repetitive
or redundant outputs.

**Implementation:**
1. Encode each generated idea sentence with SentenceBERT
2. Compute pairwise cosine similarity between all generated idea pairs
3. IDI = 1 - mean(pairwise similarities)

Higher score = more diverse and distinct ideas generated.

## Standard HELM Metrics (open_ended)

BLEU-1, BLEU-4, ROUGE-L, and F1 are computed automatically against the
`Future_work` gold reference. These are soft proxies — IAScore and IDI
are the paper's authoritative metrics.

## Notes

- The paper also uses human evaluation on 3 dimensions:
  **novelty** (1–5), **relevance** (1–5), **feasibility** (1–5)
- Human evaluation was performed on 30 sampled outputs per model per domain
- The scenario stores the full `Future_work` text as CORRECT_TAG reference
  for BLEU/ROUGE; metric implementation should segment generated output
  into individual idea sentences before computing IAScore/IDI
- The `Response_Chat` column in the xlsx files contains GPT-4/Claude-2/
  Gemini-generated outputs from the paper's experiments — useful for
  calibrating expected score ranges but should NOT be used as references
