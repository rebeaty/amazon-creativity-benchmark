# Annotator Requirements: CPers (Creativity in Persian)

Source: Paper Section 3.2-3.4 (https://arxiv.org/abs/2509.18401)

## Overview

CPers evaluation uses an LLM-as-judge approach based on the Torrance Tests of Creative Thinking (TTCT) framework, adapted for Persian literary text generation. The automated judge evaluates four creativity dimensions plus rhetorical device usage.

## Configuration for LLMAsJuryAnnotator

**Judge Model**: Claude 3.7 Sonnet (selected after validation study)
- **Validation**: Intraclass Correlation Coefficient (ICC) showed "strong agreement" with human annotators
- **Human Baseline**: 2 annotators rated 200 texts (100 human-written, 100 model-generated)
- **Temperature**: Not specified (likely 0 for consistency)

**Language**: Persian (Farsi)

**Evaluation Framework**: Four TTCT dimensions, each with 3 questions on 1-5 scale

---

## TTCT Dimensions and Rating Questions

### 1. Originality (اصالت)

Evaluates creative novelty and uniqueness.

**Questions (1-5 scale):**
1. Does the text demonstrate creativity and originality in its expression?
2. Does the text avoid clichés and overused expressions?
3. Does the text employ literary devices (simile, metaphor, hyperbole, antithesis)?

### 2. Fluency (روانی)

Assesses linguistic quality and naturalness.

**Questions (1-5 scale):**
1. Is the text grammatically correct?
2. Does the text sound natural to Persian readers?
3. Is the text appropriate for literary or conversational use?

### 3. Flexibility (انعطاف‌پذیری)

Measures perspective diversity and stylistic variety.

**Questions (1-5 scale):**
1. Does the text present multiple perspectives on the topic?
2. Does the text employ varied stylistic approaches?
3. Does the text show adaptability in expressing the theme?

### 4. Elaboration (بسط)

Evaluates richness and depth of expression.

**Questions (1-5 scale):**
1. Does the text use rich and diverse vocabulary?
2. Does the text evoke mental imagery for the reader?
3. Does the text effectively convey emotions?

### Overall Creativity Score

**Calculation**: Average of all four dimension scores
```
creativity_score = (originality + fluency + flexibility + elaboration) / 4
```

**Scale**: 1-5 (continuous, averaged across 12 questions)

---

## Rhetorical Device Analysis

In addition to TTCT scoring, evaluate presence and effectiveness of four core Persian literary devices:

### Device Types

1. **Simile (تشبیه)**: Explicit comparison using "like" or "as"
   - Example: "عشق مثل آتش است" (Love is like fire)

2. **Metaphor (استعاره)**: Implicit comparison without comparison words
   - Example: "عشق آتشی در دل من است" (Love is a fire in my heart)

3. **Hyperbole (اغراق)**: Intentional exaggeration for emphasis
   - Example: "هزار بار برایت مُردم" (I died a thousand times for you)

4. **Antithesis (تضاد)**: Juxtaposition of contrasting ideas
   - Example: "در شب تاریک، ستاره‌ای می‌درخشد" (In the dark night, a star shines)

### Device Detection Task

For each generated text:
- **Binary labels**: Presence (1) or absence (0) of each device
- **Distribution analysis**: Track which devices are overused vs underused
- **Balance assessment**: Compare to human baseline distribution

**Paper Finding**: Models overuse simile and metaphor, underuse hyperbole and antithesis

---

## Judge Prompt Template

### For TTCT Scoring

```
You are evaluating the creativity of a Persian literary text based on the Torrance Tests of Creative Thinking (TTCT) framework.

Topic: {TOPIC}
Generated Text: {GENERATED_TEXT}
Reference Text (Human): {REFERENCE_TEXT}

Please rate the generated text on the following dimensions, answering each question on a scale of 1-5 (1=Poor, 5=Excellent):

**Originality:**
1. Does the text demonstrate creativity and originality in its expression?
2. Does the text avoid clichés and overused expressions?
3. Does the text employ literary devices (simile, metaphor, hyperbole, antithesis)?

**Fluency:**
1. Is the text grammatically correct?
2. Does the text sound natural to Persian readers?
3. Is the text appropriate for literary or conversational use?

**Flexibility:**
1. Does the text present multiple perspectives on the topic?
2. Does the text employ varied stylistic approaches?
3. Does the text show adaptability in expressing the theme?

**Elaboration:**
1. Does the text use rich and diverse vocabulary?
2. Does the text evoke mental imagery for the reader?
3. Does the text effectively convey emotions?

Provide your ratings as a JSON object:
{
  "originality": {
    "q1": <score>,
    "q2": <score>,
    "q3": <score>,
    "average": <avg>
  },
  "fluency": {
    "q1": <score>,
    "q2": <score>,
    "q3": <score>,
    "average": <avg>
  },
  "flexibility": {
    "q1": <score>,
    "q2": <score>,
    "q3": <score>,
    "average": <avg>
  },
  "elaboration": {
    "q1": <score>,
    "q2": <score>,
    "q3": <score>,
    "average": <avg>
  },
  "overall_creativity": <average of all 4 dimensions>
}
```

### For Rhetorical Device Detection

```
Analyze the following Persian literary text for the presence of four rhetorical devices:

Text: {GENERATED_TEXT}

Devices to identify:
1. Simile (تشبیه): Explicit comparison using "like" or "as"
2. Metaphor (استعاره): Implicit comparison without comparison words
3. Hyperbole (اغراق): Intentional exaggeration for emphasis
4. Antithesis (تضاد): Juxtaposition of contrasting ideas

For each device, indicate whether it is present (1) or absent (0) in the text, and provide a brief justification.

Response format:
{
  "simile": {"present": <0 or 1>, "justification": "<explanation>"},
  "metaphor": {"present": <0 or 1>, "justification": "<explanation>"},
  "hyperbole": {"present": <0 or 1>, "justification": "<explanation>"},
  "antithesis": {"present": <0 or 1>, "justification": "<explanation>"}
}
```

---

## Validation and Reliability

### Human-LLM Agreement

**Dataset**: 200 texts (100 human, 100 model-generated)
**Annotators**: 2 human experts + Claude 3.7 Sonnet
**Metric**: Intraclass Correlation Coefficient (ICC)
**Result**: "Strong agreement" between human and LLM ratings

### Inter-Annotator Reliability

Human annotators showed high consistency (ICC values reported in paper Table X)

### Model Selection

**Models Tested as Judges**: Multiple LLMs evaluated
**Selected**: Claude 3.7 Sonnet (best ICC with human ratings)
**Rejected**: Models with lower agreement (not specified in abstract)

---

## Implementation Notes

### For HELM Integration

1. **Annotator Class**: Implement `TTCTAnnotator` extending `LLMAsJuryAnnotator`
2. **Language Support**: Ensure judge model supports Persian text evaluation
3. **Prompt Engineering**: Use exact questions from paper (translate to English if needed for non-Persian judge models)
4. **Output Parsing**: Extract structured scores from judge responses
5. **Aggregation**: Compute dimension averages and overall creativity score

### Alternative Evaluation Approaches

If Claude 3.7 Sonnet is unavailable:

1. **Other LLMs**: Test GPT-4, Gemini Pro for Persian evaluation capability
2. **Validate**: Run on 200-text annotated subset to compute ICC
3. **Threshold**: Only use if ICC shows "moderate" or "strong" agreement (>0.6)

### Performance Considerations

- **Latency**: Each text requires 12 questions + device detection (2 API calls)
- **Cost**: 4,371 texts × 2 calls = 8,742 LLM evaluations
- **Batching**: Consider evaluating multiple dimensions in single prompt
- **Caching**: Store judge responses to avoid re-evaluation

---

## References

### Paper

- **Title**: Evaluating the Creativity of LLMs in Persian Literary Text Generation
- **Authors**: Armin Tourajmehr, Mohammad Reza Modarres
- **Venue**: arXiv preprint (Sep 2025)
- **URL**: https://arxiv.org/abs/2509.18401
- **Paper ID**: 26c397ae35fa0d521a6b30e578924861a06d8cdd

### Dataset

- **Name**: CPers (Creativity in Persian)
- **Size**: 4,371 literary texts across 20 topics
- **Annotated Subset**: 200 texts with TTCT scores and rhetorical device labels
- **Location**: https://huggingface.co/datasets/teias-ai/CPers
- **License**: MIT

### TTCT Framework

- **Original**: Torrance Tests of Creative Thinking (1966, 1974)
- **Adaptation**: Culturally-grounded for Persian literary context
- **Dimensions**: Originality, Fluency, Flexibility, Elaboration
- **Scale**: 1-5 Likert (3 questions per dimension)
