# CS4 Benchmark - Annotator Notes

## Overview

CS4 (Comparing the Skill of Creating Stories by Controlling the Synthesized Constraint Specificity) evaluates LLM creativity in story generation by systematically varying the number of constraints in the prompt.

## Evaluation Metrics

The benchmark uses **automatic evaluation** (no human annotation required for basic metrics):

### 1. Constraint Satisfaction (Primary Metric)
- **Method**: GPT-4-based evaluation
- **Process**: For each constraint, GPT-4 determines if it's satisfied (yes/no) with explanation
- **Scoring**: Number of constraints satisfied / Total number of constraints
- **Strictness**: Mark as "yes" only if completely satisfied; partial satisfaction = "no"

### 2. Coherence
- **Purpose**: Evaluate overall narrative coherence
- **Method**: Automatic coherence scoring (likely using language models)

### 3. Diversity
- **Purpose**: Assess story originality
- **Method**: N-gram diversity calculation
- **Rationale**: Higher diversity indicates less reliance on training data patterns

### 4. Perplexity
- **Purpose**: Measure text predictability and fluency
- **Method**: Standard language model perplexity calculation
- **Interpretation**: Balance needed - too low = formulaic, too high = incoherent

### 5. QUC (Quality Under Constraints)
- **Purpose**: Custom metric for creative quality when constrained
- **Details**: Specific to CS4 benchmark (see paper for formula)

### 6. RCS (Relative Creativity Score)
- **Purpose**: Measure creativity relative to constraint level
- **Details**: Custom metric (see paper for formula)

## Constraint Satisfaction Evaluation Details

### GPT-4 Prompt Format
```
System: You are an expert reader. I will give you a story followed by a set of constraints.
Your task is to carefully read both of them and tell how many constraints are being satisfied in the story.

Output format:
1. [yes/no] - [Constraint text]
   - If yes: Quote the sentence/line from the story where it's satisfied
   - If no: Explain how it's being violated

...repeat for all constraints...

Number of constraints satisfied: [number]
```

### Evaluation Criteria
- **Complete satisfaction required**: Constraint must be fully satisfied, not partially
- **Evidence-based**: Must cite specific text from story when marking "yes"
- **Strict marking**: Any doubt or partial satisfaction = "no"

## Key Research Findings

1. **Constraint Specificity Effect**: LLMs struggle significantly when prompts have 31-39 constraints
2. **Trade-off Challenge**: Models find it difficult to balance:
   - Constraint satisfaction
   - Narrative coherence
   - Creative originality
3. **Learning from Human Feedback**: Helps models select better stories from training data but has limited impact on generating genuinely novel creative stories

## Dataset Structure

### Two Dataset Types

1. **Instruction-based** (Realistic Fiction)
   - 50 base instructions
   - 250 total instances (50 × 5 constraint levels)
   - Example: "Write about two characters struggling to shift their priorities..."

2. **Story-based** (Writing Prompts)
   - 50 base prompts
   - 250 total instances (50 × 5 constraint levels)
   - Derived from r/WritingPrompts dataset

### Constraint Levels
- **7 constraints**: Low specificity, high creative freedom
- **15 constraints**: Moderate specificity
- **23 constraints**: High specificity
- **31 constraints**: Very high specificity
- **39 constraints**: Extreme specificity

## Recommended Evaluation Approach for HELM

For HELM implementation, we recommend:

1. **Primary Metric**: Constraint Satisfaction Ratio
   - Use LLM-as-judge (GPT-4 or similar)
   - Report both overall satisfaction and per-constraint-level breakdown

2. **Secondary Metrics**:
   - Diversity (n-gram based - can be computed automatically)
   - Generation length (compare to typical story length)

3. **Tertiary Analysis**:
   - Stratify results by constraint level (7, 15, 23, 31, 39)
   - Analyze performance degradation as constraints increase
   - Report trade-offs between constraint satisfaction and other quality metrics

## Example Instance

**Instruction**: Write about two characters struggling to shift their priorities and keep their relationship intact as they age.

**Constraints (7-constraint version)**:
1. The story must be set in a small coastal town where the characters have lived their whole lives.
2. Both characters must have had successful careers in different fields—one in science and the other in the arts.
3. Each character must face a unique health issue that influences their priorities.
4. Introduce a supporting character who acts as a catalyst for change in their relationship.
5. The narrative must include flashbacks to their youth, showing how their dreams and aspirations have changed over time.
6. The characters must attempt to learn a new skill together, which they find challenging.
7. Incorporate a significant scene that takes place in a local cafe that holds sentimental value to the characters.

**Expected Output**: A creative story that satisfies all 7 constraints while maintaining narrative coherence and originality.

## Citation

```bibtex
@article{lakkaraju2024cs4,
  title={CS4: Measuring the Creativity of Large Language Models Automatically by Controlling the Number of Story-Writing Constraints},
  author={Lakkaraju, Anirudh and Atmakuru, Anirudh and Nainani, Jatin and Bheemreddy, Rohith Siddhartha Reddy and Yao, Zonghai and Zamani, Hamed and Chang, Haw-Shiuan},
  journal={arXiv preprint arXiv:2410.04197},
  year={2024}
}
```
