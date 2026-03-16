# Evaluation Metrics: Cue-word-based Creative Story Generation

## Standard NLG Metrics (Implemented in HELM)

Use `get_open_ended_generation_metric_specs()` for:
- **BLEU-1, BLEU-4**: N-gram overlap with human reference stories
- **ROUGE-L**: Longest common subsequence
- **F1**: Token-level F1 score

## LLM-as-Judge Evaluation (Optional Enhancement)

The paper (Section 4) evaluated stories on 4 creativity dimensions. This could be implemented using HELM's `LLMAsJuryAnnotator`.

### Dimensions from Paper

1. **Creativity** (1-5 scale)
   - How creative is the story overall?

2. **Originality** (1-5 scale)
   - How original and unique are the ideas?

3. **Surprise** (1-5 scale)
   - How surprising or unexpected is the story?

4. **Effectiveness/Value** (1-5 scale)
   - How effective is the story at being creative?

### Judge Prompt Template

```
Rate the following short story on {dimension} from 1 to 5, where:
1 = Very low {dimension}
5 = Very high {dimension}

Cue words: {cue_words}
Story: {STORY}

Provide your rating as a single number (1-5).
```

### Implementation Notes

- **Judge model**: GPT-4 or Claude (paper used human experts and non-experts)
- **Human correlation**: Paper achieved inter-rater reliability but exact correlation not reported
- **Dataset notes**: Original dataset includes expert (2 raters) and non-expert (5 raters) ratings for 479 pre-generated stories

## Automated Metrics from Paper

The paper also used several automated metrics:

### Diversity Metrics
- N-gram diversity (n=1,2,3,4,5)
- Inverse homogenization

### Novelty Metrics
- Semantic distance of dominant terms from cue words

### Surprise Metrics
- Average semantic distances between consecutive sentences

### Linguistic Complexity
- Lexical: word count, average word length, Flesch-Kincaid readability
- Syntactic: POS tag ratios, dependency path depths, constituency tree depth

These would require custom metric implementations beyond HELM's standard metrics.

## Recommendation

For initial HELM integration:
1. Use standard open-ended generation metrics (BLEU, ROUGE, F1)
2. Optionally implement LLM-as-judge for the 4 creativity dimensions
3. Consider custom metrics (diversity, novelty, surprise) as future enhancement
