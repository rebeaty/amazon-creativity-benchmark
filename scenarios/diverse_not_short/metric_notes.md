# Metric Notes: Diverse-not-Short

Source: Paper Section 4.2, Appendix B; `evaluation_scripts/` in codebase

## Task-Specific Evaluation

### RAT (Remote Associates Test)
- **Metric**: Accuracy (exact_match)
- Correct answer is labeled in the TSV (143 items)
- Paper baseline: Llama-3.1-8B-Instruct achieves ~0.35 accuracy

### AUT (Alternative Uses Test)
- **Metric**: LLM-as-judge creativity scoring
- Paper uses separate evaluator to rate creativity of generated uses
- Multi-seed (10 seeds): diversity measured across runs

### CWT (Creative Writing Task)
- **Metrics**: DSI (Divergent Semantic Integration), entropy, lexical diversity
  - DSI: BERT-based semantic diversity of generated text
  - Entropy: Distributional entropy of generated content
  - TTR / MTLD / HDD: Type-token ratio variants for lexical diversity
  - Flesch reading ease / Kincaid grade: Readability
  - Sentence length, dependency distance: Structural diversity
- Multi-seed (10 seeds): diversity measured across runs
- Paper's decile_map.csv provides reference distributions by word count

### DAT (Divergent Association Test)
- **Metric**: Semantic distance (GloVe-based average pairwise distance)
- Multi-seed (100 seeds): diversity of word lists across runs
- Related to dat_creative_writing scenario (different paper/prompt)

### C-DAT (Conditional Divergent Association Test)
- **Metric**: Semantic distance within generated words, relevance to cue
- 15 test words, single seed
- Related to cdat scenario (different paper/prompt, fewer cues)

### PGT (Persona Generation Task)
- **Metric**: Diversity of generated personas (unique names, cities, occupations)
- Multi-seed (100 seeds): diversity across repeated generations
- Evaluated by counting unique values in each JSON field

## Multi-Seed Diversity Evaluation

The paper's key contribution is measuring output diversity across multiple
generations with different seeds:
- PGT/DAT: 100 seeds each (1 prompt × 100 runs)
- CWT/AUT: 10 seeds each (N prompts × 10 runs)
- RAT/C-DAT: 1 seed (deterministic tasks)

In HELM, this can be approximated by running the same scenario multiple times
with different generation seeds, or by using `num_instances` duplication
(similar to dat_creative_writing pattern).

## Reference Baselines (Paper Table 2)

| Model | RAT Acc | AUT Score | CWT DSI | DAT Dist |
|-------|---------|-----------|---------|----------|
| Llama-3.1-8B-Instruct | ~0.35 | - | - | - |
| + Diverse-NS DPO | improved diversity without length bias |
