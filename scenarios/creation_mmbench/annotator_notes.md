# Annotator Requirements: Creation-MMBench

Source: Paper Section 3.3, GitHub repo (VLMEvalKit integration)

## Configuration for LLMAsJuryAnnotator

**Judge model:** GPT-4o (specifically `gpt-4o-0806` recommended by authors)

**Evaluation approach:** Instance-specific criteria with dual evaluation

**Dimensions:** Two evaluation aspects per instance
- **Subjective requirement**: Task-specific quality criteria (coherence, creativity, etc.)
- **Groundtruth alignment**: Visual factuality and accuracy with respect to image content

**Scale:** 1-10 for Visual Factuality Score (VFS)

**Output metrics:**
- Visual Factuality Score (VFS): 1-10 scale
- Reward: Normalized score from -100 to +100

## Evaluation Methodology

### Dual Evaluation

To reduce position bias, the paper uses "dual evaluation":
1. **First pass**: Evaluate response against reference in standard order
2. **Second pass**: Swap positions and re-evaluate
3. **Aggregate**: Combine scores from both passes

This approach is enabled by default in the VLMEvalKit framework.

### Instance-Specific Criteria

Each test case has customized evaluation criteria stored in the dataset's `criteria` field (Python dict format):

```python
{
    'subjective requirement': '<detailed quality criteria>',
    'groundtruth alignment': '<visual factuality requirements>'
}
```

**Example subjective requirements** (story continuation task):
- Ensure continuation is cohesive and logically connected
- Use vivid descriptions and realistic dialogue
- Maintain original tone and style
- Keep characters consistent and evolving
- Introduce new challenges that drive plot forward

## Judge Prompt Structure

The paper mentions using "carefully crafted instance-specific criteria" but does not provide exact prompt templates in the published paper. The evaluation is implemented through VLMEvalKit framework.

**General structure** (inferred from methodology):
```
<image(s)>

Query: {question}

Response: {model_response}

Evaluate the response based on:
1. Subjective Requirement: {subjective_requirement_from_criteria}
2. Groundtruth Alignment: {groundtruth_alignment_from_criteria}

Provide a Visual Factuality Score from 1 to 10.
```

## Reference Answers

The dataset includes two types of references:
- **reference_answer_by_gpt4o**: GPT-4o generated responses (746/765 examples)
- **ground_truth**: Human-provided answers (356/765 examples)

When ground_truth is available, it takes precedence for evaluation.

## Task Categories

Evaluation criteria vary by task category:

### Literary Writing (8 tasks, ~120 examples)
- Narrative coherence, creativity, character development
- Descriptive language, dialogue quality
- Story structure and plot progression

### Common Functional Writing (18 tasks, ~270 examples)
- Clarity, tone appropriateness
- Practical utility for intended purpose
- Audience-appropriate language

### Professional Functional Writing (19 tasks, ~285 examples)
- Domain expertise demonstration
- Professional formatting and structure
- Actionable and practical content

### Creative Multimodal Understanding (6 tasks, ~90 examples)
- Accurate interpretation of visual content
- Creative insight and analysis
- Integration of visual and textual information

## Implementation Notes

- **VLMEvalKit Integration**: The authors use the VLMEvalKit framework for evaluation
- **Cost consideration**: Dual evaluation approximately doubles API costs
- **Reproducibility**: Paper reports results across multiple models (GPT-4V, GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, Qwen-VL-Max, InternVL2-40B, LLaVA-OneVision-72B, etc.)

## Human Correlation

The paper states that instance-specific criteria enable assessment of "both general response quality and visual-factual alignment," but specific human correlation coefficients are not reported in the main paper.

## Citation

```bibtex
@inproceedings{fang2025creationmmbench,
  title={Creation-MMBench: Assessing Context-Aware Creative Intelligence in MLLM},
  author={Fang, Huiyu and others},
  booktitle={ICCV},
  year={2025}
}
```
