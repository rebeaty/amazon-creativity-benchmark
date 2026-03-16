# Annotator Requirements: Showerthoughts Generation

Source: *SEM 2024 paper, Section 4.2 (Survey of Human Preferences)
Paper: https://aclanthology.org/2024.starsem-1.23.pdf

## Background

The paper evaluates Showerthoughts generation quality through human ratings on multiple dimensions. This task requires LLM-as-jury evaluation to replicate the human evaluation methodology.

## Configuration for LLMAsJuryAnnotator

**Judge model:** GPT-4 (or equivalent)

**Evaluation dimensions:** 5 dimensions, each rated independently

1. **Logical Validity** - "It makes a true/valid/logical statement"
2. **Creativity** - "It is creative"
3. **Humor** - "It is funny"
4. **Cleverness** - "It is clever"
5. **General Score** - "I like this Showerthought" (overall quality)

**Scale:** 6-point Likert scale (1 = strongly disagree / lowest, 6 = strongly agree / highest)

## Judge Prompt Templates

### System Prompt
```
You are an evaluator for creative text generation. You will rate Showerthoughts - creative,
witty observations about everyday life from Reddit's r/Showerthoughts community.

A Showerthought is "a miniature epiphany about everyday life - the type of thought you might
have while carrying out a routine task like showering, driving, or daydreaming. At their best,
Showerthoughts are universally relatable and find the amusing/interesting within the mundane."

Rate each Showerthought on the specified dimension using a 6-point scale where:
1 = Strongly disagree / Very poor
2 = Disagree / Poor
3 = Somewhat disagree / Below average
4 = Somewhat agree / Above average
5 = Agree / Good
6 = Strongly agree / Excellent
```

### Dimension-Specific Prompts

**Logical Validity:**
```
Rate the following Showerthought on logical validity - does it make a true, valid,
or logical statement? Does it make sense and hold up to scrutiny?

Showerthought: {RESPONSE}

Rating (1-6):
```

**Creativity:**
```
Rate the following Showerthought on creativity - is it creative, original, and novel?
Does it present an unexpected or unique perspective?

Showerthought: {RESPONSE}

Rating (1-6):
```

**Humor:**
```
Rate the following Showerthought on humor - is it funny or amusing?

Showerthought: {RESPONSE}

Rating (1-6):
```

**Cleverness:**
```
Rate the following Showerthought on cleverness - is it clever or witty? Does it show
intellectual playfulness or insightfulness?

Showerthought: {RESPONSE}

Rating (1-6):
```

**General Score:**
```
Rate the following Showerthought on overall quality - do you like this Showerthought?
Consider all aspects including creativity, wit, relatability, and entertainment value.

Showerthought: {RESPONSE}

Rating (1-6):
```

## Notes from Paper

- **Survey participants:** 56 human evaluators (25 in Group A, 31 in Group B)
- **Evaluation setup:** Each participant rated 45 Showerthoughts (15 genuine, 30 AI-generated from 3 models)
- **Demographics:** 89.4% had ML experience, ~47% visit Reddit regularly
- **Key findings:**
  - Human-written Showerthoughts scored highest on all dimensions
  - GPT-Neo (2.7B fine-tuned) performed best among models on logical validity and cleverness
  - ChatGPT (GPT-3.5-turbo zero-shot) performed best on creativity and humor
  - Smaller GPT-2 model performed ~30% worse than human-written
  - GPT-Neo and ChatGPT achieved 6-7% below human level overall

## Human Performance Baselines

Mean scores from paper Table 2 (6-point scale):

| Model    | General | Logical Validity | Creativity | Humor | Cleverness |
|----------|---------|------------------|------------|-------|------------|
| Genuine  | 3.71    | 4.20             | 3.63       | 3.18  | 3.41       |
| GPT-Neo  | 3.40    | 3.96             | 3.23       | 2.74  | 3.15       |
| ChatGPT  | 3.23    | 3.55             | 3.45       | 2.85  | 3.07       |
| GPT-2    | 2.42    | 3.10             | 2.42       | 2.10  | 2.19       |

## Implementation Notes

- Each dimension should be evaluated independently (not influenced by others)
- The task requires subjective judgment; inter-rater agreement may be moderate
- Consider aggregating scores across multiple judge invocations for reliability
- Humor generation is particularly challenging for LLMs (paper finding)
