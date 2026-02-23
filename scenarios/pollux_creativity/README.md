# POLLUX: Russian Creativity Evaluation Benchmark

**Paper:** Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX
**Venue:** December 2025
**arXiv:** [2505.24616](https://arxiv.org/abs/2505.24616)
**Dataset:** [ai-forever/POLLUX](https://huggingface.co/datasets/ai-forever/POLLUX)
**Judge Models:** [pollux-judge-7b-r](https://huggingface.co/ai-forever/pollux-judge-7b-r), [pollux-judge-32b-r](https://huggingface.co/ai-forever/pollux-judge-32b-r)

## Overview

POLLUX is a comprehensive benchmark for evaluating the generative capabilities of Russian-language LLMs. The full benchmark contains **2,115 expert-authored prompts** across **35 task types** with **161,076 evaluation samples** from 7 different models.

This scenario focuses on the **creativity-related tasks** from POLLUX, which test literary generation, creative brainstorming, style transfer, and subjective interpretation.

## Key Finding from Paper

> "Even top-tier models like Claude 3.5 Sonnet and OpenAI o1 still lag behind human experts in tasks that heavily rely on creativity."

Human evaluators frequently criticized model outputs for a perceived **"lack of soul"** or emotional depth in creative tasks.

---

## Dataset Overview

### Total POLLUX Dataset
- **161,076 evaluation samples** (7 models × ~23K prompts)
- **2,115 unique instructions** (expert-authored, all in Russian)
- **35 task types** covering code, creative writing, analysis, etc.
- **Cost:** $262,316 for 24,447 hours of expert annotation

### Creativity Tasks (This Scenario)

| Task Type (Russian) | English Translation | Examples | Description |
|---------------------|---------------------|----------|-------------|
| Написать художественный текст | Write Literary Text | 6,764 | Stories, poems, narratives across 15 literary movements and 93 genres |
| Творческий брейншторминг | Creative Brainstorming | 5,558 | Generate novel ideas and creative solutions |
| Интерпретация текста (субъективная) | Text Interpretation (Subjective) | 6,081 | Analytical creativity, subjective analysis |
| Стайл-трансфер | Style Transfer | 5,977 | Adapt text across registers and styles |
| ИИ как персонаж | AI as Character | 6,025 | Role-play as specified characters |
| Прикладной брейншторминг | Applied Brainstorming | 6,452 | Practical ideation and problem-solving |

**Total Creativity Examples:** ~36,000 (filtered from 161K total)
**Unique Instructions (after deduplication):** ~10,070

---

## Task Categories Detail

### 1. Literary Text Generation (Написать художественный текст)

**Examples:** 6,764
**Difficulty:** Easy, Medium, Hard

Write original literary works including:
- **Poems:** 15 literary movements, rhyme and meter requirements
- **Stories:** 93 substyles and genres
- **Narratives:** Character-driven plots with dramatic structure
- **Canonical Style Imitation:** 17 canonical Russian authors

**Evaluation Criteria:**
- Creativity (originality, inventiveness)
- Dramaturgy (narrative structure, plot arc)
- Dialogue Expressiveness (emotional depth, character voice)
- Genre Appropriateness (adherence to literary conventions)
- Rhyme Quality (for poetry)
- Verse Meter (for poetry)
- Literary Accents (stylistic flourishes, metaphors, epithets)

**Example Instruction:**
> "Хочу написать философское стихотворение о выборе. Напиши первое четверостишие..."
>
> *(I want to write a philosophical poem about choice. Write the first quatrain...)*

### 2. Creative Brainstorming (Творческий брейншторминг)

**Examples:** 5,558

Generate innovative ideas for creative problems:
- Novel concepts for projects
- Unique approaches to challenges
- Original perspectives on topics
- Imaginative solutions

**Evaluation Criteria:**
- Creativity (originality, innovation)
- Diversity of ideas
- Practical applicability
- Depth of exploration

### 3. Style Transfer (Стайл-трансфер)

**Examples:** 5,977

Transform text across different:
- Literary registers (formal ↔ informal)
- Genres (prose ↔ poetry)
- Time periods (modern ↔ classical)
- Authorial voices (Shakespeare → Chekhov)

**Evaluation Criteria:**
- Target style adherence
- Content preservation
- Linguistic appropriateness
- Creative interpretation

### 4. Text Interpretation (Subjective) (Интерпретация текста)

**Examples:** 6,081

Provide creative, subjective analysis of texts:
- Literary criticism
- Thematic interpretation
- Symbolic analysis
- Cultural contextualization

**Evaluation Criteria:**
- Depth of insight
- Originality of interpretation
- Evidence-based reasoning
- Clarity of expression

### 5. AI as a Character (ИИ как персонаж)

**Examples:** 6,025

Role-play as specific characters with:
- Consistent persona
- Appropriate knowledge and attitudes
- Character-appropriate language
- Emotional authenticity

**Evaluation Criteria:**
- Character adherence
- Dialogue expressiveness
- Situational appropriateness
- Emotional authenticity

### 6. Applied Brainstorming (Прикладной брейншторминг)

**Examples:** 6,452

Practical creative problem-solving:
- Business ideation
- Product development
- Process improvement
- Strategic planning

**Evaluation Criteria:**
- Creativity with practical constraints
- Feasibility of ideas
- Problem-solution fit
- Innovative thinking

---

## Evaluation Methodology

### LLM-as-a-Judge with Detailed Criteria

**Scoring Scale:** 0-4 points per criterion
- **4:** Fully meets requirements
- **3:** Mostly meets requirements
- **2:** Partially meets requirements
- **1:** Minimally meets requirements
- **0:** Does not meet requirements

### Creativity-Specific Criteria

1. **Creativity (Креативность)**
   - Originality and inventiveness
   - Going beyond standard approaches
   - Innovative ideas

2. **Dramaturgy (Драматургия)**
   - Narrative structure and arc
   - Plot development and pacing
   - Story coherence

3. **Dialogue Expressiveness (Выразительность диалога)**
   - Emotional depth and nuance
   - Character voice distinction
   - Natural conversation flow

4. **Genre Appropriateness (Попадание в жанр)**
   - Adherence to genre conventions
   - Style consistency
   - Audience expectations

5. **Literary Accents (Литературные акценты)**
   - Use of metaphors, epithets
   - Stylistic flourishes
   - Linguistic artistry

6. **Character Adherence (Соблюдение образа персонажа)**
   - Persona consistency
   - Role-appropriate behavior
   - Voice authenticity

7. **Rhyme Quality (Качество рифмы)**
   - Prosodic correctness
   - Creative and natural rhyming
   - (Poetry tasks only)

8. **Verse Meter (Размер стиха)**
   - Rhythmic structure correctness
   - Metrical pattern adherence
   - (Poetry tasks only)

### Judge Models

**Official POLLUX Judges:**
- `ai-forever/pollux-judge-7b-r` (7B parameters)
- `ai-forever/pollux-judge-32b-r` (32B parameters, more accurate)

**Alternative:** GPT-4 with Russian prompts

---

## Dataset Structure

### Fields per Example

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | Text | Task prompt in Russian |
| `reference_answer` | Text | Reference (often empty for creative tasks) |
| `answer` | Text | Model-generated output |
| `model_id` | Text | Model that generated the answer |
| `task_type` | Text | Main task category (35 types) |
| `task_subtype` | Text | Subcategory (88 subtypes) |
| `task_subsubtype` | Text | Fine-grained type (238 types) |
| `difficulty` | Text | Easy/Medium/Hard |
| `domain` | Text | Subject domain (6 domains) |
| `is_provocative` | Boolean | Provocative content flag |
| `criteria_name` | Text | Evaluation criterion (58 types) |
| `criteria_description` | Text | Criterion definition |
| `rubrics` | Text | Evaluation rubric (125 variations) |
| `rubrics_example` | Text | Example application |
| `annotations` | List | Evaluator comments and scores |
| `criteria_score` | Float | Expert annotation (0-4) |

### Data Distribution

**Total Dataset:** 161,076 examples
- 7 models × ~2,115 instructions × ~10 criteria = ~150K evaluations
- Plus additional samples

**Creativity Subset (This Scenario):**
- ~36,000 examples before deduplication
- ~10,070 unique instructions after deduplication

---

## Implementation Details

### Scenario Features

```python
# Load all creative tasks
scenario = POLLUXCreativityScenario(include_all_creative=True)

# Load specific task type
scenario = POLLUXCreativityScenario(
    task_type="Написать художественный текст"  # Literary text only
)
```

### Deduplication Logic

Since the dataset contains multiple model outputs per instruction:
1. Filter for creative task types
2. Deduplicate by unique `instruction`
3. Keep one example per instruction with criteria metadata
4. Result: ~10K unique creative prompts

### Metadata in Instance.extra_data

Each instance includes:
```python
instance.extra_data = {
    'task_type': 'Написать художественный текст',
    'task_subtype': 'Поэзия',
    'difficulty': 'Medium',
    'domain': 'Художественный',
    'criteria_name': 'Креативность',
    'criteria_description': '...',
    'rubrics': '...',
    'expected_score': 3.5,  # Expert annotation
    'is_provocative': False,
}
```

### Evaluation Options

**Option 1: Use Existing Annotations**
- Expert scores already in `expected_score`
- Compare model output score to expert annotation
- Cost: $0 (annotations included)

**Option 2: Use LLM-as-a-Judge**
- Query `ai-forever/pollux-judge-32b-r` or GPT-4
- Provide criteria, rubrics, and examples
- Get 0-4 score with justification
- Cost: API calls

See `annotator_notes.md` for detailed implementation.

---

## Language Considerations

**⚠️ Russian Language Required**

All instructions, outputs, and evaluation must be in Russian:
- **Models must support Russian** (GPT-4, Claude, Gemini, Russian LLMs)
- **Judge prompts are in Russian**
- **Criteria descriptions are in Russian**
- **Cultural context matters** (15 literary movements, 17 canonical authors)

---

## Key Findings from Paper

### 1. SOTA Models Lag on Creativity

| Model | Creativity Score | Human Expert |
|-------|------------------|---------------|
| Claude 3.5 Sonnet | Lower | Baseline |
| OpenAI o1 | Lower | Baseline |
| GPT-4 | Lower | Baseline |

Even the best models underperform humans on creativity-heavy tasks.

### 2. "Lack of Soul" Problem

Human annotators frequently noted that model outputs lacked:
- **Emotional depth** and nuance
- **Authentic voice** and personality
- **Genuine creativity** vs. pattern matching
- **Cultural understanding** of Russian literature

### 3. Subjective Assessment Challenges

Discrepancies in ratings predominantly related to:
- Emotional nuance evaluation
- Interpretation of "creativity"
- Cultural and literary knowledge
- Personal aesthetic preferences

---

## Python Syntax Validation

```bash
python -m py_compile scenarios/pollux_creativity/scenario.py
# ✓ Valid syntax
```

---

## Comparison to Other Benchmarks

| Aspect | POLLUX Creativity | Standard Benchmarks |
|--------|-------------------|---------------------|
| **Language** | Russian only | Mostly English |
| **Task Scope** | Creativity-focused | General capability |
| **Evaluation** | LLM-as-a-Judge (0-4) | Accuracy/F1 |
| **Criteria** | 8 creativity-specific | Task-specific |
| **Reference** | Often none (creative) | Fixed correct answers |
| **Cost** | $262K for annotations | Varies |
| **Cultural Context** | Deep (literary movements) | Usually minimal |

---

## Usage Example

```python
from helm.benchmark.scenarios.scenario import Scenario
from scenarios.pollux_creativity.scenario import POLLUXCreativityScenario

# Load all creativity tasks
scenario = POLLUXCreativityScenario(include_all_creative=True)
instances = scenario.get_instances(output_path="./data")

# Example instance
instance = instances[0]
print(instance.input.text)  # Russian instruction
print(instance.extra_data['task_type'])  # Task category
print(instance.extra_data['expected_score'])  # Expert annotation

# Evaluate with judge model
# (see annotator_notes.md for implementation)
```

---

## Citation

```bibtex
@article{pollux2025,
    title={Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX},
    author={AI Forever Team},
    year={2025},
    journal={arXiv preprint arXiv:2505.24616},
}
```

---

## Resources

- 📄 [Paper (arXiv)](https://arxiv.org/abs/2505.24616)
- 🤗 [Dataset (HuggingFace)](https://huggingface.co/datasets/ai-forever/POLLUX)
- 🤖 [Judge Model 7B](https://huggingface.co/ai-forever/pollux-judge-7b-r)
- 🤖 [Judge Model 32B](https://huggingface.co/ai-forever/pollux-judge-32b-r)
- 📋 [Annotator Notes](./annotator_notes.md)

---

## Future Work

- **Multilingual Extension:** Adapt evaluation framework to other languages
- **Human Correlation:** Study correlation between LLM judges and human experts
- **Cross-Cultural:** Compare Russian vs. English creativity evaluation
- **Fine-tuning:** Train models specifically on Russian creative writing
