# Annotator Requirements: TinyStories

Source: Paper Section 3 (Evaluation), arXiv:2305.07759

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4
Dimensions: Grammar, Creativity, Consistency
Scale: 1-10 for each dimension
Additional output: Age group estimation (A: 3 or under, B: 4-5, C: 6-7, D: 8-9, E: 10-12)

## Evaluation Framework

The evaluation uses GPT-4 to grade story completions "as if those were stories written by students and graded by a (human) teacher."

**Setup:**
1. The model receives a story beginning (from Evaluation_prompts.yaml)
2. The model generates a story completion
3. GPT-4 grades the completion on three dimensions

**Framing:**
- Present as a student exercise: "The student is given a beginning of a story. The student needs to complete it into a full story. The exercise tests the student's language abilities and creativity."

## Judge Prompt Template

The exact prompt text should be extracted from the paper PDF (Section 3 or Appendix). The prompt instructs GPT-4 to:

1. Grade the student's completion in terms of:
   - **Grammar**: Correctness of language use
   - **Creativity**: Originality and imaginative quality of the story
   - **Consistency**: How well the completion aligns with the story beginning
   - **Plot coherence**: Whether the story makes logical sense

2. Estimate the age group of the hypothetical student writer

**Expected output format:**
```
Grammar: 8/10, Creativity: 7/10, Consistency: 7/10, Age group: E (10-12)
```

## Evaluation Procedure

From the paper:
1. Use 44 test stories from Evaluation_prompts.yaml (story beginnings)
2. Generate completions for each prompt
3. Submit each completion to GPT-4 for grading
4. Average scores across all test stories for final metrics

## Notes

- The paper noted that "creativity benefits from further scale increases" among the evaluated dimensions
- The evaluation framework was designed to overcome "the flaws of standard benchmarks which often requires the model's output to be very structured"
- This approach allows for more nuanced assessment of creative language generation
- The stories use simple vocabulary appropriate for 3-4 year-olds

## Implementation Requirements

For HELM's `LLMAsJuryAnnotator`:
- Judge model: `gpt-4` (or `gpt-4-turbo` for cost efficiency)
- Parse numeric scores from GPT-4's response (X/10 format)
- Extract three separate scores: grammar_score, creativity_score, consistency_score
- Optionally track age_group predictions for analysis
- Average scores across all 44 test instances

## Alternative Metrics

While the paper uses GPT-4 as judge, standard automatic metrics could complement evaluation:
- ROUGE-L (lexical overlap with reference stories from validation set)
- Perplexity (fluency measure)
- Self-BLEU (diversity across generated completions)
