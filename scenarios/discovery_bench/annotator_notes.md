# Annotator Requirements: DiscoveryBench

Source: Paper Section 4.2, https://arxiv.org/abs/2407.01725

## Primary Metric: Hypothesis Matching Score (HMS)

HMS is a faceted, decomposable metric that evaluates generated hypotheses by breaking them into three components:

### Decomposition (via LLM)

A judge model decomposes both the gold and predicted hypotheses into sub-hypotheses, each with:
- **Context**: Boundary conditions (e.g., "in 1989 data", "for Hispanic men")
- **Variables**: Concepts that interact (e.g., ["time preference", "BMI"])
- **Relationship**: Nature of interaction (e.g., "positive correlation", "coefficient 0.36")

### Scoring Formula

```
HMS = ctxF1 × (1/|M|) × Σ(varF1 × rel_acc)

Where:
  ctxF1     = F1 score for context alignment between gold and predicted
  varF1     = F1 score for variable overlap per matched sub-hypothesis
  rel_acc   = Relationship accuracy:
                100 = exact match
                 50 = broader but encompassing relationship
                  0 = incorrect
  |M|       = Number of matched sub-hypothesis pairs
```

Result: Score between 0-100.

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4 (paper uses GPT-4 for decomposition)

### Decomposition Prompt

Given a hypothesis, decompose it into sub-hypotheses. For each, extract:
1. Context (boundary conditions)
2. Variables (list of concepts)
3. Relationship (how variables interact)

### Matching Prompt

Given a gold hypothesis decomposition and a predicted hypothesis decomposition:
1. Match sub-hypotheses by context similarity
2. For each matched pair, score variable overlap (F1)
3. For each matched pair, score relationship accuracy (exact=100, partial=50, wrong=0)

## Simplified Alternative

For HELM open_ended evaluation, BLEU/ROUGE against the gold hypothesis provides a coarser but automatic signal. The HMS metric would require a custom metric implementation wrapping the LLM decomposition step.

## Notes

- Paper reports best model achieves ~25% HMS on real tasks (Reflexion with Oracle)
- Human performance not reported (hypotheses are from published papers)
- The original benchmark gives models code execution access to analyze CSVs;
  text-only evaluation tests reasoning from metadata alone, which is a harder task
- Implementation reference: https://github.com/allenai/discoverybench/blob/main/discovery_eval.py
