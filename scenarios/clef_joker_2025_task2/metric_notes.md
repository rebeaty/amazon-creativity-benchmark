# Metric Notes: CLEF JOKER 2025 Task 2 — Wordplay Translation

Source: CEUR-WS Vol-4038 paper_219 (Task 2 overview), arXiv:2507.06506 (participant paper)

## Official Evaluation (Task Organisers)

| Metric | Type | Notes |
|--------|------|-------|
| **BLEU** | Automated | Official metric; computed against up to 29 reference translations per pun |
| **BERTScore** | Automated | Used by participants (e.g., DUTH: F1 = 86.96); rewards semantic fidelity over surface form |
| **Human evaluation** | Manual | Expert French speaker scored 50 puns × N runs on equivalence and wordplay preservation |

## Why BLEU Undershoots for This Task

The overview paper explicitly notes: *"Traditional metrics like BLEU and BERTScore reward direct, literal translations, and therefore penalise outputs that include idiomatic language."* A model that successfully reconstructs wordplay using different French vocabulary may score lower on BLEU than a literal translation, even if the pun quality is higher.

HELM's `get_open_ended_generation_metric_specs()` includes BLEU-1, BLEU-4, ROUGE-L, and F1, which are sufficient for baseline comparison but may not fully capture creative wordplay quality.

## BERTScore Implementation

BERTScore is not included in HELM's built-in open_ended metrics. To add it:
- Use `sentence-transformers` or the `bert-score` PyTorch package
- Recommended model for multilingual FR/EN: `bert-base-multilingual-cased`
- Compute F1 of each generated translation against the full set of reference translations and take the maximum

## Human Evaluation Criteria (from overview paper)

Evaluators rated translations on:
1. **Meaning equivalence** — Does the French pun convey the same core idea?
2. **Wordplay preservation** — Is there an active double meaning in the French?
3. **Authenticity** — Does the pun feel natural in French (not like a foreign import)?
4. **Emotional resonance** — Does it land as funny to a native French speaker?

Scale: Binary or 1–5 Likert (exact scale not specified in publicly available docs).

## Top Benchmark Results (CLEF 2025 Task 2, 9 teams, 52 runs)

| System | BLEU | BERTScore F1 |
|--------|------|--------------|
| Lucie-7B + SFT (best) | 42.40 | — |
| DUTH hybrid NMT | 41.11 | 86.96 |
| arXiv:2507.06506 (multi-agent) | 1st/2nd place (manual eval) | — |

## Recommendation

For a complete evaluation pipeline, combine:
1. **BLEU-4** (HELM built-in, for comparability with published results)
2. **BERTScore F1** (custom implementation, multilingual BERT)
3. **LLM-as-judge** for wordplay quality if human evaluation is not feasible
   — judge prompt should assess both double-meaning presence and naturalness in French
