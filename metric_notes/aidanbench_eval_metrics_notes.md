# Metric Notes: AidanBench — Novel Idea Generation

Source: OpenReview fz969ahcvJ, GitHub github.com/aidanmclaughlin/AidanBench

## Original Multi-Turn Scoring (paper)

AidanBench's native protocol is iterative:

1. Ask the model the question → get response #1
2. Append response #1 to context, ask again → get response #2
3. Repeat until EITHER:
   - **Coherence** (LLM judge, o1-mini) ≤ 15 / 100
   - **Novelty** (cosine similarity to all prior responses via embeddings) ≤ 0.15
4. **AidanBench Score** = count of responses that passed both checks

Top published scores (from GitHub results):
| Model | Score |
|-------|-------|
| o1-preview | ~30+ |
| GPT-4o | ~22 |
| Claude 3.5 Sonnet | ~20 |
| GPT-4-turbo | ~18 |

## HELM Adaptation (batch prompting)

This scenario uses batch prompting (single-turn, N=30 responses at once) instead of the iterative protocol. This requires custom post-processing metrics:

### Metric 1: Self-BLEU (Diversity)

Measures how different the N generated responses are from each other.
Lower Self-BLEU = more diverse = better.

```python
from nltk.translate.bleu_score import sentence_bleu
# For each response i, compute BLEU against all other responses as references
# Self-BLEU = mean of per-response BLEU scores
```

Reference: Texygen (Zhu et al. 2018). Lower is better; 0 = completely distinct.

### Metric 2: Coherence (LLM-as-judge)

Rate each numbered response for quality and plausibility on 0–100.
Threshold from paper: responses scoring ≤ 15 are considered failures.

**Judge prompt template** (from AidanBench GitHub):
```
You are evaluating responses to the question: "{question}"

Rate the following response on a scale of 0-100 for coherence, quality,
and appropriateness. A score of 0 means completely incoherent or off-topic.
A score of 100 means a perfect, insightful, well-reasoned response.

Response: {response}

Score (0-100):
```

Judge model: o1-mini (original paper); any capable judge model acceptable.

### Metric 3: AidanBench Score (Approximate)

Simulates the original stopping-criterion count from batch output:

```python
def aidanbench_score(responses, coherence_scores, novelty_scores,
                     coherence_threshold=15, novelty_threshold=0.15):
    # Count responses passing both checks
    # Note: in batch mode, novelty is computed pairwise (not sequentially)
    score = 0
    for i, (resp, coh, nov) in enumerate(zip(responses, coherence_scores, novelty_scores)):
        if coh > coherence_threshold and nov > novelty_threshold:
            score += 1
    return score
```

### Metric 4: Novelty (Embedding Similarity)

For each response, compute mean cosine similarity to all other responses.
Novelty = 1 - mean_similarity. Threshold from paper: ≤ 0.15 = too repetitive.

Recommended embedding model: `text-embedding-3-small` (OpenAI) or
`sentence-transformers/all-MiniLM-L6-v2` (open-source).

## Notes on Batch vs Sequential Evaluation

The batch approach may overestimate diversity compared to the original protocol because:
- Models can plan N responses upfront (no pressure from sequential context)
- Novelty is computed pairwise, not sequentially against a growing history

The original iterative protocol is the gold standard. For full fidelity, implement
the multi-turn loop outside HELM using the GitHub evaluation harness directly.

## Published Thresholds (from paper/GitHub)
- Coherence threshold: ≤ 15/100 → response fails
- Novelty threshold: cosine similarity ≤ 0.15 → response fails (too repetitive)
- Embedding model used: OpenAI `text-embedding-ada-002` (original)
