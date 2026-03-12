# Annotator Requirements: TinyFabulist

Source: tinyfabulist/conf/evaluator.yaml in https://github.com/klusai/tinyfabulist

## Configuration for LLMAsJuryAnnotator

Judge model: o3-mini-2025-01-31 (temperature 0.0, max 350 tokens)
Dimensions: grammar, creativity, moral_clarity, adherence_to_prompt
Scale: 1-10 per dimension
Additional: best_age_group classification (A-E)

## System Prompt (exact from evaluator.yaml)

```
You are an expert literary critic specializing in fables and moral tales. Your evaluations should be objective, consistent, and based on established literary standards. Age-appropriateness is a key consideration in your assessment. Provide your assessment in valid, properly-formatted JSON only. Do not include any text outside the JSON object. Your response must be parseable by a JSON parser with no preprocessing. Balance critical analysis with constructive feedback, focusing on both strengths and weaknesses.
```

## Evaluation Prompt Template (exact from evaluator.yaml)

```
Evaluate the following fable according to these specific criteria:

1. **Grammar & Style (1-10)**:
   • 1-3: Significant errors that impede understanding
   • 4-6: Some errors but generally readable
   • 7-10: Clean, polished writing with appropriate language and style for a fable

2. **Creativity & Originality (1-10)**:
   • 1-3: Derivative, predictable, or clichéd
   • 4-6: Contains some original elements but follows familiar patterns
   • 7-10: Fresh perspective, innovative approach while maintaining classic fable structure

3. **Moral Clarity (1-10)**:
   • 1-3: Moral absent, confused, or contradictory
   • 4-6: Moral present but underdeveloped or lacking impact
   • 7-10: Clear, meaningful moral that provides genuine insight

4. **Adherence to Prompt (1-10)**:
   • 1-3: Missing multiple required elements from the prompt
   • 4-6: Incorporates main elements but overlooks some instructions
   • 7-10: Thoroughly addresses all prompt requirements while maintaining narrative cohesion

5. **Age Group Fit**:
   Determine which age group this fable is most appropriate for based on:
   • Vocabulary complexity and sentence structure
   • Conceptual difficulty of the moral lesson
   • Story length and complexity
   • Content appropriateness

Age groups are defined as:
  - A: 3 years or under
  - B: 4-7 years
  - C: 8-11 years
  - D: 12-15 years
  - E: 16 years or above

Format your response as valid JSON with this structure:
{
    "type": "Fable Evaluation",
    "grammar": <integer 1-10>,
    "creativity": <integer 1-10>,
    "moral_clarity": <integer 1-10>,
    "adherence_to_prompt": <integer 1-10>,
    "best_age_group": "<letter: A, B, C, D, or E>",
    "explanation": [
        "<One sentence explaining grammar & style score>",
        "<One sentence explaining creativity & originality score>",
        "<One sentence explaining moral clarity score>",
        "<One sentence explaining adherence to prompt score>",
        "<One sentence explaining why this fable best fits the chosen age group>"
    ]
}

Be critical but fair. Ensure your entire evaluation is concise yet informative.

Original Prompt:
{prompt}

Fable:
{fable}
```

## Published Baselines (Table 1, 100 benchmark prompts, judge: o3-mini)

| Model | Grammar | Creativity | Moral Clarity | Adherence | Mean |
|-------|---------|------------|---------------|-----------|------|
| Llama-3.1-Tulu-3-8B | 8.32 | 6.97 | 8.50 | 7.69 | 7.87 |
| Llama-3.1-8B-Instruct | 8.42 | 6.59 | 8.21 | 8.18 | 7.85 |
| Llama-3.1-8B-Instruct-Scp | 8.33 | 6.44 | 8.16 | 7.59 | 7.63 |
| Falcon3-7B-Instruct | 8.29 | 6.56 | 8.27 | 7.06 | 7.54 |
| Qwen2.5-7B-Instruct | 8.28 | 6.21 | 8.02 | 6.81 | 7.33 |
| Mistral-7B-Instruct-v0.3 | 8.12 | 6.31 | 8.05 | 6.58 | 7.26 |
| Phi-3-mini-4k-instruct | 8.10 | 6.28 | 7.87 | 6.61 | 7.21 |
| deepseek-llm-7b-chat | 8.04 | 6.08 | 7.88 | 5.72 | 6.93 |
| aya-23-8B | 7.78 | 5.75 | 7.24 | 5.12 | 6.47 |
| SmolLM2-1.7B-Instruct | 7.79 | 5.40 | 6.98 | 4.81 | 6.25 |
| Llama-3.2-1B-Instruct | 7.87 | 5.41 | 6.56 | 4.98 | 6.21 |

## Corpus-Level Metrics (reference-free)

In addition to LLM-as-judge, the paper uses three corpus-level metrics:
- **Self-BLEU**: Intra-set redundancy (lower = greater diversity)
- **Distinct-1**: Proportion of unique unigrams (lexical richness)
- **Flesch Reading Ease**: Readability score via sentence/syllable counts

These are computed across the full set of generated fables, not per-instance.

## Notes

- Paper primarily evaluates small open-weight models (1B-8B parameters)
- 100 benchmark prompts are combinatorially generated from 100+ characters,
  100+ traits, 100+ settings, 100+ conflicts, 100+ resolutions, 100+ morals
- All prompts target age group B (4-7 years) with simple vocabulary constraints
- Judge expects JSON-only output (no explanation text outside JSON)
- Evaluation is reference-free; no gold-standard fables exist
