# Evaluation Notes: DataNarrative

Source: DataNarrative paper (EMNLP 2024), Figures 18-26

## Multi-Agent Framework

The paper proposes a multi-stage iterative refinement framework with two LLM agents:
- **Narration Agent**: Generates content (reflection, outline, narration)
- **Critic Agent**: Reviews and creates revision plans

### Generation Pipeline

1. **Reflection Stage** (Figures 18-20)
   - Generate: Systematic examination of data tables, identify key insights
   - Critique: Check factual accuracy, identify discrepancies
   - Revise: Apply corrections based on revision plan

2. **Outline Stage** (Figures 21-23)
   - Generate: Create linear narrative structure (intro, middle, conclusion)
   - Critique: Verify theme consistency and factual accuracy
   - Revise: Adjust narrative flow and ensure theme alignment

3. **Narration Stage** (Figures 24-26)
   - Generate: Write final data story with paragraph headers and visualization placeholders
   - Critique: Check consistency with outline and data tables
   - Revise: Final refinement for coherence and accuracy

## Evaluation Methodology

### Model-Based Evaluation (GPT-4 Judge)

The paper uses GPT-4 for automated quality assessment on multiple dimensions:

**Evaluation Dimensions:**
- **Factual Accuracy**: Correctness of data descriptions and statistics
- **Coherence**: Logical flow and consistency of narrative
- **Comprehensiveness**: Coverage of important data insights
- **Theme Consistency**: Alignment with stated intent/topic

**Judge Configuration:**
- Model: GPT-4 (specific version not stated in figures)
- Scale: Not specified in provided figures (likely Likert scale or binary)
- Comparison: Likely pairwise comparison between generated stories

### Human Evaluation

The paper mentions "model-based and human evaluations" but specific details are not provided in Figures 18-26.

### Traditional Metrics

For HELM integration, standard open-ended generation metrics are applicable:
- **BLEU-1, BLEU-4**: N-gram overlap with reference narratives
- **ROUGE-L**: Longest common subsequence
- **F1**: Token-level precision and recall

## Key Evaluation Criteria (from Prompts)

### Factual Accuracy (Critical)
- Numerical data must be correct
- Contextual interpretations must align with data
- No overlooked or misrepresented details
- Specific data points must be accurately cited

### Narrative Quality
- **Linear Structure**: Clear introduction, development, conclusion
- **Theme Consistency**: Maintain focus on stated intent throughout
- **Engagement**: Balance technical accuracy with accessibility
- **Synthesis**: Connect data points into coherent whole

### Visualization Integration
- Include visualization placeholders where appropriate
- Provide sufficient specifications (chart type, axes, data values)
- Ensure visualizations support narrative elements
- Keep chart types simple (line, bar, pie, scatter)

## Implementation Notes for HELM

**For Standard HELM Evaluation:**
- Use `open_ended` metrics (BLEU, ROUGE, F1)
- Compare generated narratives to reference paragraphs

**For LLM-as-Judge Extension:**
- Would require implementing GPT-4 judge with prompts similar to Figures 19, 22, 25
- Judge should evaluate: factual accuracy, coherence, comprehensiveness, theme consistency
- Could use multi-dimensional scoring or pairwise comparison

**For Multi-Agent Framework:**
- The full iterative pipeline (6 stages) is not implemented in basic scenario
- Could be extended by chaining multiple instances through the refinement cycle
- Would require storing intermediate outputs (reflection, outline) between stages

## Dataset-Specific Notes

**Sources have different characteristics:**
- **GapMinder**: Global demographic/economic trends, temporal data
- **Pew Research**: Social/political topics, survey data
- **Tableau**: Mixed domains, various visualization types

**Evaluation may need source-specific considerations:**
- GapMinder: Emphasis on trend identification and temporal patterns
- Pew: Focus on survey interpretation and demographic breakdowns
- Tableau: Diverse data types require flexible narrative approaches
