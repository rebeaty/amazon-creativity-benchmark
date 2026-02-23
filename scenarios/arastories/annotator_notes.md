# Annotator Requirements: AraStories

Source: notebooks/gpt4_evaluation_generator.ipynb in https://github.com/UBC-NLP/arastories

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4 (paper used gpt-4-0125-preview)
Dimensions: fluency, coherence, following_instructions, consistency, variety
Scale: 1-5 Likert per dimension

## Judge Prompt Template (exact from repository)

```
You are an expert in Arabic language, its dialects, and storytelling. I would like your help in evaluating a story written by a student based on a set of instructions. You are expected to give a score out of five based on the following features:
**Fluency:** How smooth and natural the text is, including appropriate grammar, vocabulary, and sentence structure.
**Coherence:** The logical connection and flow of sentences and ideas, making the text easy to understand.
**Following Instructions:** How well the text adheres to the provided instructions or task requirements.
**Consistency:** How consistently accurate and uniform the information and style are throughout the text.
**Variety:** How well does the model generate story in the required Arabic variety.
Give the scores directly without explanations or additions. I will first give you the instructions on which the story was based, followed by the story written by the student. Remember, I want the evaluation directly without explanation.

{prompt_and_story}
```

Where `{prompt_and_story}` is the original prompt (instructions) followed by the model-generated story.

## Human Evaluation Protocol

- 4 native Arabic speakers evaluated stories
- Pairwise ranking (not absolute scoring) on 3 criteria:
  - Instruction Following
  - Fluency
  - Variety Adherence
- 10 stories per model comparison
- Results reported as preference percentages

## Dialect-Specific Considerations

The "Variety" dimension evaluates dialect fidelity:
- MSA: Should use formal Modern Standard Arabic
- Egyptian: Should use Egyptian Arabic dialect features
- Moroccan: Should use Moroccan Arabic (Darija) dialect features

This dimension is particularly important as many models default to MSA even when prompted for dialect-specific output.

## Notes

- Paper evaluated 5 models: Model A (fine-tuned), Model B (fine-tuned), GPT-3.5, Command-R, AceGPT-Chat
- 20 test prompts per dialect in the paper's evaluation (subset of full dataset)
- Reference stories (GPT-4 generated) are available in the dataset but evaluation is reference-free (judge scores the generated story against the prompt constraints)
- Judge prompt expects scores "directly without explanations" — output format is 5 numeric scores
