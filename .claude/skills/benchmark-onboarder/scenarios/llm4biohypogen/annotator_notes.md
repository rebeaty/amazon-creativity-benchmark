# Annotator Requirements: LLM4BioHypoGen

Source: Section 4.2 and Appendix from https://arxiv.org/abs/2407.08940

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4
Temperature: Not specified (likely 0 for deterministic scoring)
Task: Evaluate generated hypotheses on 4 dimensions
Scale: 0-3 for each dimension

## Evaluation Dimensions

### 1. Novelty (0-3)
**Question:** Does the hypothesis introduce new information or perspectives?
- 0: Not novel at all
- 1: Minimally novel
- 2: Moderately novel
- 3: Highly novel

### 2. Relevance (0-3)
**Question:** Is the hypothesis aligned with the research background?
- 0: Not relevant
- 1: Minimally relevant
- 2: Moderately relevant
- 3: Highly relevant

### 3. Significance (0-3)
**Question:** Does the hypothesis have potential scientific impact?
- 0: Not significant
- 1: Minimally significant
- 2: Moderately significant
- 3: Highly significant

### 4. Verifiability (0-3)
**Question:** Can the hypothesis be tested using existing methods or data?
- 0: Not verifiable
- 1: Minimally verifiable
- 2: Moderately verifiable
- 3: Highly verifiable

## Judge Prompt Template

Based on the paper's evaluation protocol:

```
You are evaluating a scientific hypothesis generated based on a research background.

Research Background:
{BACKGROUND}

Generated Hypothesis:
{HYPOTHESIS}

Please evaluate the hypothesis on the following dimension:

{DIMENSION}: {DIMENSION_QUESTION}

Provide your rating on a scale of 0-3:
- 0: Not {dimension_adjective} at all
- 1: Minimally {dimension_adjective}
- 2: Moderately {dimension_adjective}
- 3: Highly {dimension_adjective}

Rating:
```

Where:
- DIMENSION ∈ {Novelty, Relevance, Significance, Verifiability}
- DIMENSION_QUESTION is the corresponding question above
- dimension_adjective ∈ {novel, relevant, significant, verifiable}

## Additional Metrics

### Automatic Reference-Based Metrics
For test sets with ground truth hypotheses:
1. **BLEU**: Word overlap with reference hypothesis
2. **ROUGE-L**: Longest common subsequence with reference
3. **SelfBLEU**: Measures diversity/uncertainty across multiple generations

### Human Evaluation Protocol
From the paper:
- 3 biomedical experts evaluated 100 examples (5% of dataset)
- Used same 4-dimensional scoring (0-3 scale)
- Calculated Pearson and Spearman correlations with GPT-4 scores
- Results showed strong correlation, validating GPT-4 as judge

## Implementation Notes

1. **Multi-dimensional scoring**: Each hypothesis receives 4 separate scores
2. **Aggregate score**: Can be computed as average or sum of 4 dimensions
3. **Reference-free evaluation**: Novelty, Significance, Verifiability don't require ground truth
4. **Reference-based evaluation**: Relevance can benefit from ground truth for comparison
5. **Background context**: All 4 metrics require the original research background for evaluation

## Example Evaluation

**Background:** (1) Fatty acid species with a maximum chain length of 16-18 carbon atoms account for >90% of total fatty acids in most mammalian tissues...

**Generated Hypothesis:** (1) Substrate specificity of fatty acyltransferases determines the distribution bias of VLCFA between sphingolipids and glycerolipids...

**Scores:**
- Novelty: 2 (Introduces new perspective on enzyme specificity)
- Relevance: 3 (Directly addresses VLCFA distribution mentioned in background)
- Significance: 2 (Could impact understanding of lipid metabolism)
- Verifiability: 3 (Can be tested with enzyme assays and lipid analysis)

## Human-AI Agreement

According to the paper:
- Pearson correlation: ~0.7-0.8 across dimensions
- Spearman correlation: ~0.65-0.75 across dimensions
- GPT-4 tends to be slightly more lenient than human experts
- Agreement is highest for Relevance, lowest for Novelty

## Notes

- Multi-agent framework (Analyst-Scientist-Critic) can improve scores
- Few-shot learning doesn't always improve performance
- Unseen test sets show models can generate novel hypotheses on new literature
- Tool use (PubMed, ArXiv) can enhance hypothesis quality
