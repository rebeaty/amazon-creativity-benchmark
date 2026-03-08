# Annotator Requirements: Pun2Pun

Source: Official evaluation code (eval/aacc_pun.py, eval/hit.py, eval/ovl.py)

## Configuration for LLMAsJuryAnnotator

**Judge model**: GPT-4 or similar (original paper uses various models)

**Metrics**: Two custom metrics used in combination

### 1. Hit Metric (Binary)
Determines if the translation contains a pun at all.

**Judge Prompt Template**:
```
You are a helpful assistant that determines if the model prediction covers the annotation.
You should output only 'yes' if it does, and 'no' if not.
Focus only on content and semantics, ignore the style. Minor differences or extended
explanations are acceptable if it does hit the annotation.

Please compare these two answers:
Prediction: {MODEL_OUTPUT}
Annotation: {REFERENCE_PUN_WORD}

Does the prediction cover the annotation? Output only 'yes' or 'no'.
```

**Scale**: Binary (yes/no)

### 2. Overlap (Ovl) Metric
Measures semantic similarity between source and translated pun using embeddings.

**Implementation**:
- Uses text embedding models (e.g., text-embedding-3-small)
- Computes cosine similarity between source pun and translation
- Range: 0.0 to 1.0

## Alternative: Standard HELM Metrics

Since this is open-ended generation, standard HELM metrics could also apply:
- **BLEU**: Would require reference translations (not available)
- **ROUGE**: Would require reference translations (not available)
- **Semantic similarity**: Cosine similarity with embeddings (similar to Ovl)

## Implementation Notes

For HELM integration, two approaches:

1. **Custom Annotator**: Implement LLM-as-judge following the Hit metric pattern
   - Binary decision: Does translation preserve pun?
   - Could expand to multi-dimensional (pun preservation, fluency, creativity)

2. **Embedding-based metric**: Implement Overlap metric using HELM's embedding support
   - More objective than LLM-judge
   - Measures semantic preservation

3. **Hybrid**: Combine both metrics
   - Hit for pun preservation detection
   - Overlap for semantic similarity
   - Report both scores

## Human Evaluation Details (from paper)

The original paper used human evaluation alongside automated metrics:
- Evaluators judged whether pun effect was preserved
- Rated translation quality
- Assessed creativity of pun recreation

For automated evaluation to approximate human judgment, the Hit metric showed reasonable correlation.
