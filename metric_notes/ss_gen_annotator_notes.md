# Annotator Requirements: SS-GEN

Source: Paper Section 4.2, Appendix A2

## Configuration for LLMAsJuryAnnotator

**Judge model:** GPT-4 (paper used GPT-4 for automated quality assessment)

**Evaluation dimensions:** 5 dimensions on 1-5 Likert scale
- **Coherence**: Logical flow and consistency of the narrative
- **Descriptiveness**: Richness of detail and descriptive language (vs. coaching)
- **Empathy**: Appropriateness and sensitivity for children with autism
- **Grammaticality**: Correctness of grammar and syntax
- **Relevance**: Alignment with the given title and social story purpose

**Sample size:** Paper evaluated 200 randomly sampled test stories with GPT-4

## Judge Prompt Template

The paper does not provide the exact GPT-4 evaluation prompt. A suitable template would be:

```
Evaluate the following Social Story for children with autism on {dimension} using a scale of 1-5.

Title: {TITLE}
Generated Story: {GENERATED_STORY}

Rate the {dimension} of this story from 1 (poor) to 5 (excellent).

Provide your rating as a single number.
```

## Human Evaluation Criteria

The paper also describes human evaluation with **Quality Assessment Criteria**:

### 1. Structural Clarity (1-5 scale)
Assesses whether the story has:
- A clear title
- An introduction identifying the topic positively
- A detailed main body
- A conclusion that reinforces the message

### 2. Descriptive Orientation (Binary: Yes/No)
Checks the **GR-Eight formula**: Descriptive sentences must exceed coaching sentences by a 2:1 ratio

### 3. Situational Safety (Binary: Yes/No)
Evaluates:
- Appropriate perspective (first/third person only, never second person)
- Positive and patient tone
- Literal accuracy (appropriate for autism-spectrum audience)
- Vocabulary accuracy

## Evaluation Setup Details

- **Human evaluation sample:** 50 randomly selected test stories
- **GPT-4 evaluation sample:** 200 randomly selected test stories
- **Human annotators:** Received training on the Quality Assessment Criteria
- **Annotation interface:** Custom web-based interface with checklists

## Notes

- The paper reports that GPT-4 evaluation showed high consistency with human judgments
- Human evaluation focused on constraint compliance (structural, descriptive ratio, safety)
- GPT-4 evaluation assessed broader quality dimensions (coherence, empathy, etc.)
- Both evaluation approaches complement the traditional BLEU/ROUGE metrics
- The paper does not report specific inter-annotator agreement scores
