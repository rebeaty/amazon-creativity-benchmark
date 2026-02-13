# Annotator Requirements: POLLUX Creativity Evaluation

**Source:** Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX
**Paper:** [arXiv:2505.24616](https://arxiv.org/abs/2505.24616)
**Dataset:** [ai-forever/POLLUX](https://huggingface.co/datasets/ai-forever/POLLUX)

## Overview

POLLUX uses **LLM-as-a-Judge** evaluation with detailed criteria-based assessment. The benchmark includes 161,076 evaluation samples from expert annotations across 2,115 unique instructions and 7 different models.

**Key Innovation:** Multi-criteria evaluation with detailed rubrics and scoring protocols for each task type, enabling transparent assessment beyond simple preference comparisons.

## Evaluation Methodology

### Scoring Scale

**0-4 point scale** for each criterion:
- **0**: Does not meet requirements at all
- **1**: Minimally meets requirements
- **2**: Partially meets requirements
- **3**: Mostly meets requirements
- **4**: Fully meets requirements

### Judge Models

The paper releases two judge models:
- **pollux-judge-7b-r**: 7B parameter evaluator
- **pollux-judge-32b-r**: 32B parameter evaluator (more accurate)

Available on HuggingFace: `ai-forever/pollux-judge-7b-r` and `ai-forever/pollux-judge-32b-r`

## Creativity-Specific Evaluation Criteria

### 1. Creativity (Креативность)

**Definition:** Originality and inventiveness of the response

**Rubric:**
- **4**: Highly original, innovative ideas that go beyond standard approaches
- **3**: Creative elements present, shows some originality
- **2**: Conventional approach with minor creative touches
- **1**: Derivative, formulaic content
- **0**: No creativity, purely mechanical response

**Applies to:** Literary text, creative brainstorming, style transfer

### 2. Dramaturgy (Драматургия)

**Definition:** Narrative structure, plot development, and story arc

**Rubric:**
- **4**: Well-structured narrative with compelling arc, proper pacing
- **3**: Coherent structure with minor pacing issues
- **2**: Basic structure present but uneven development
- **1**: Weak structure, disjointed narrative
- **0**: No discernible narrative structure

**Applies to:** Literary text generation (stories, narratives)

### 3. Dialogue Expressiveness (Выразительность диалога)

**Definition:** Emotional depth, character voice, and natural conversation flow

**Rubric:**
- **4**: Rich emotional nuance, distinct character voices, natural flow
- **3**: Good expressiveness with occasional flat moments
- **2**: Adequate but somewhat generic dialogue
- **1**: Stilted, unnatural dialogue
- **0**: No dialogue or completely inappropriate

**Applies to:** Literary text, AI as a character

**Key Challenge (from paper):** Human evaluators frequently criticized model responses for "lack of soul" or emotional depth

### 4. Genre Appropriateness (Попадание в жанр)

**Definition:** Adherence to conventions and expectations of the specified genre

**Rubric:**
- **4**: Perfect genre fit, all conventions respected
- **3**: Mostly appropriate with minor deviations
- **2**: Recognizable but inconsistent genre adherence
- **1**: Weak genre markers, mostly generic
- **0**: Does not match specified genre

**Applies to:** Literary text across different genres (15 literary movements, 93 substyles/genres)

### 5. Rhyme Quality (Качество рифмы)

**Definition:** Prosodic correctness and quality of rhyming in poetry

**Rubric:**
- **4**: Excellent rhymes, creative and natural
- **3**: Good rhymes with occasional imperfections
- **2**: Basic rhyming but forced or predictable
- **1**: Poor rhymes, many errors
- **0**: No rhyming or completely incorrect

**Applies to:** Poetry generation tasks (146 samples in dataset)

### 6. Verse Meter (Размер стиха)

**Definition:** Rhythmic structure and metrical pattern correctness

**Rubric:**
- **4**: Perfect meter throughout
- **3**: Mostly correct with minor deviations
- **2**: Recognizable meter but inconsistent
- **1**: Weak meter, frequent errors
- **0**: No discernible meter

**Applies to:** Poetry generation (153 samples in dataset)

### 7. Literary Accents (Литературные акценты)

**Definition:** Stylistic flourishes, literary devices, and linguistic artistry

**Rubric:**
- **4**: Rich use of metaphors, epithets, and literary devices
- **3**: Good use of some literary devices
- **2**: Basic devices, somewhat generic
- **1**: Minimal literary style
- **0**: No literary devices, purely functional text

**Note (from paper):** Only 4.9% of instructions incorporated stylistic devices like metaphors and epithets

### 8. Character Adherence (Соблюдение образа персонажа)

**Definition:** Consistency with specified character persona and role

**Rubric:**
- **4**: Perfect character consistency throughout
- **3**: Mostly consistent with minor lapses
- **2**: Recognizable but inconsistent characterization
- **1**: Weak character adherence
- **0**: Does not match specified character

**Applies to:** AI as a character tasks (6,025+ samples)

## General Evaluation Criteria

These apply across creative and non-creative tasks:

### Linguistic Competence (Лингвистическая компетентность)

- Grammar, spelling, punctuation correctness
- Appropriate register and style for task type
- Natural Russian language usage

### Formatting Quality

- Proper text structure and organization
- Appropriate use of paragraphs, sections
- Readability and visual presentation

### Proactivity (Проактивность)

- Engagement with the task
- Going beyond minimum requirements
- Providing helpful additional context

### Artifact Absence

- No generation artifacts or errors
- No hallucinations or factual errors
- No inappropriate content

## Judge Prompt Template

```
Оцените ответ модели по следующим критериям:

Задание: {instruction}
Ответ модели: {model_output}

Критерий оценки: {criteria_name}
Описание критерия: {criteria_description}

Рубрика:
{rubrics}

Пример применения:
{rubrics_example}

Оцените ответ по шкале от 0 до 4 и объясните вашу оценку.

Оценка:
```

**Translation:**
```
Evaluate the model's response according to the following criteria:

Task: {instruction}
Model response: {model_output}

Evaluation criterion: {criteria_name}
Criterion description: {criteria_description}

Rubric:
{rubrics}

Application example:
{rubrics_example}

Rate the response on a scale from 0 to 4 and explain your rating.

Score:
```

## Implementation for HELM

### Option 1: Use Existing Annotations

The POLLUX dataset already contains expert annotations with scores (0-4) for each criterion. These can be used directly:

```python
# From instance.extra_data
expected_score = instance.extra_data['expected_score']  # Expert annotation
criteria_name = instance.extra_data['criteria_name']
```

### Option 2: Use LLM-as-a-Judge

Implement `LLMAsJuryAnnotator` using the POLLUX judge models or GPT-4:

```python
class POLLUXCreativityAnnotator(Annotator):
    def annotate(self, request: Request, completion: str) -> Annotation:
        # Load criteria from instance metadata
        criteria_name = request.instance.extra_data['criteria_name']
        criteria_description = request.instance.extra_data['criteria_description']
        rubrics = request.instance.extra_data['rubrics']

        # Build judge prompt in Russian
        judge_prompt = f"""
        Оцените ответ модели по критерию "{criteria_name}".

        Задание: {request.instance.input.text}
        Ответ модели: {completion}

        Описание критерия: {criteria_description}
        Рубрика: {rubrics}

        Оцените по шкале от 0 до 4.
        """

        # Query judge model
        judge_response = self.query_judge(judge_prompt)

        # Parse score and justification
        score = self.extract_score(judge_response)

        return Annotation(
            score=score,
            metadata={'judge_response': judge_response}
        )
```

## Dataset Statistics

**Human Annotation Effort:**
- Total hours: 24,447 hours
- Cost: $262,316
- Number of annotators: Multiple expert annotators per sample

**Sample Distribution:**
- Literary Text: 6,764 samples
- Creative Brainstorming: 5,558 samples
- Text Interpretation (subjective): 6,081 samples
- Style Transfer: 5,977 samples
- AI as Character: 6,025+ samples

**Model Performance (from paper):**
- Top models (Claude 3.5 Sonnet, OpenAI o1) still lag behind humans on creativity tasks
- Particular weakness in "emotional nuance" and "soul"

## Key Findings on Creativity Evaluation

1. **Subjectivity Challenge:** Discrepancies in ratings predominantly relate to emotional nuance
2. **Human-AI Gap:** Even SOTA models struggle with true creativity
3. **Emotional Depth:** Models criticized for lacking "soul" in creative outputs
4. **Cultural Context:** Russian literary traditions add complexity (15 movements, 17 canonical authors)

## Notes for Implementation

- **Language:** All evaluation must be in Russian
- **Cultural Context:** Familiarity with Russian literary traditions helpful
- **Multiple Criteria:** Each task may have multiple evaluation criteria
- **Judge Model:** Use `ai-forever/pollux-judge-32b-r` for best results, or GPT-4 with Russian prompts
- **Ground Truth:** Expert annotations available as baseline comparisons

## References

- Paper: [arXiv:2505.24616](https://arxiv.org/abs/2505.24616)
- Dataset: [HuggingFace](https://huggingface.co/datasets/ai-forever/POLLUX)
- Judge Models: [pollux-judge-7b-r](https://huggingface.co/ai-forever/pollux-judge-7b-r), [pollux-judge-32b-r](https://huggingface.co/ai-forever/pollux-judge-32b-r)
