# Annotator Requirements: IDRBench

Source: Adapted from IDRBench paper (arXiv:2601.06676) evaluation methodology

## Overview

IDRBench requires LLM-as-judge evaluation to assess the quality of generated research reports. The original benchmark uses an "interaction-aware evaluation suite," but this adaptation simplifies to single-turn generation with four key dimensions.

## Configuration for LLMAsJuryAnnotator

**Judge Model:** GPT-4 (or GPT-4-turbo)

**Evaluation Dimensions:**
1. **Accuracy** - Factual correctness based on provided documents
2. **Completeness** - Addresses all aspects of the research question
3. **Coherence** - Logical organization and clear flow
4. **Citations** - Proper attribution and use of source documents

**Scale:** 1-5 Likert scale per dimension
- 1 = Poor
- 2 = Below Average
- 3 = Average
- 4 = Good
- 5 = Excellent

## Judge Prompt Templates

### Dimension 1: Accuracy

```
You are evaluating a research report for factual accuracy.

Research Question: {RESEARCH_QUESTION}

Supporting Documents:
{DOCUMENTS}

Generated Report:
{GENERATED_REPORT}

Rate the report's ACCURACY on a scale of 1-5:

Criteria:
- All claims are supported by the provided documents
- No hallucinated or invented information
- Correct interpretation of data and facts
- Appropriate caveats and qualifications

5 = Excellent: All claims accurately reflect source documents, no errors
4 = Good: Mostly accurate with minor interpretation issues
3 = Average: Some inaccuracies or unsupported claims
2 = Below Average: Multiple factual errors or unsupported statements
1 = Poor: Significant fabrication or misrepresentation of sources

Provide your rating as a single number (1-5):
```

### Dimension 2: Completeness

```
You are evaluating a research report for completeness.

Research Question: {RESEARCH_QUESTION}

Context:
- Industry: {INDUSTRY}
- Domain: {DOMAIN}

Generated Report:
{GENERATED_REPORT}

Rate the report's COMPLETENESS on a scale of 1-5:

Criteria:
- Addresses all aspects of the research question
- Covers key considerations for the industry/domain
- Provides sufficient depth of analysis
- Includes actionable insights or recommendations

5 = Excellent: Comprehensive coverage of all aspects
4 = Good: Addresses main points with minor gaps
3 = Average: Covers some aspects but misses important elements
2 = Below Average: Significant gaps in addressing the question
1 = Poor: Fails to adequately address the research question

Provide your rating as a single number (1-5):
```

### Dimension 3: Coherence

```
You are evaluating a research report for coherence and organization.

Generated Report:
{GENERATED_REPORT}

Rate the report's COHERENCE on a scale of 1-5:

Criteria:
- Logical flow and structure
- Clear organization of ideas
- Smooth transitions between points
- Professional writing quality
- Easy to follow and understand

5 = Excellent: Exceptionally well-organized and clear
4 = Good: Well-structured with good flow
3 = Average: Adequate organization but could be clearer
2 = Below Average: Disorganized or hard to follow
1 = Poor: Incoherent or confusing structure

Provide your rating as a single number (1-5):
```

### Dimension 4: Citations

```
You are evaluating a research report for proper source attribution.

Supporting Documents:
{DOCUMENTS}

Generated Report:
{GENERATED_REPORT}

Rate the report's use of CITATIONS on a scale of 1-5:

Criteria:
- Claims are properly attributed to specific documents
- Document IDs are referenced when making claims
- Appropriate level of citation (not over- or under-cited)
- Clear connection between claims and sources

5 = Excellent: Excellent citation practices throughout
4 = Good: Most claims properly attributed
3 = Average: Some citations present but inconsistent
2 = Below Average: Few citations or poor attribution
1 = Poor: No citations or misattribution of sources

Provide your rating as a single number (1-5):
```

## Aggregation

**Overall Score:** Average of the four dimension scores

**Interpretation:**
- 4.0-5.0: Excellent research report
- 3.0-3.9: Good research report
- 2.0-2.9: Acceptable but needs improvement
- 1.0-1.9: Poor research report

## Implementation Notes

### For HELM Integration

1. **Create Custom Annotator**: Extend `LLMAsJuryAnnotator` or similar
2. **Sequential Evaluation**: Evaluate each dimension separately to avoid bias
3. **Prompt Engineering**: Include full context (question, documents, report) in each prompt
4. **Response Parsing**: Extract numeric rating from LLM response
5. **Fallback**: If LLM returns non-numeric, retry or use default score of 3

### Example Usage

```python
from helm.benchmark.annotation.annotator import LLMAsJuryAnnotator

annotator = LLMAsJuryAnnotator(
    model="gpt-4",
    dimensions=["accuracy", "completeness", "coherence", "citations"],
    scale=(1, 5),
    prompt_templates=IDRBENCH_PROMPTS
)

scores = annotator.annotate(
    question=instance.input.text,
    response=model_output,
    context=supporting_docs
)
```

## Validation and Reliability

**Human Correlation:** Original IDRBench paper reports strong correlation between
automated metrics and human judgments for deep research tasks.

**Inter-Rater Reliability:** GPT-4 has shown consistent scoring behavior across
similar research synthesis tasks.

**Recommended Safeguards:**
1. Sample manual review of 10-20% of evaluations
2. Flag extreme scores (all 1s or all 5s) for review
3. Compare aggregate statistics across models for sanity check

## Alternative Evaluation Approaches

If LLM-as-judge is not available, consider:

1. **ROUGE-L**: Measure similarity to reference reports (if available)
2. **Fact Verification**: Binary check if key facts from documents appear in report
3. **Citation Count**: Simple metric counting document ID references
4. **Length**: Basic sanity check (reports should be 200-500 tokens)

## Notes

- This is an **adapted version** of IDRBench for single-turn evaluation
- Original IDRBench includes interaction costs (turns, tokens) which this adaptation does not measure
- Focus is on report quality rather than interactive research process
- Small dataset (15 questions) means this is a challenging, high-quality eval set
