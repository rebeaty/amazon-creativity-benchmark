---
name: benchmark-onboarder
description: Onboard AI creativity benchmarks into HELM-compliant Scenario implementations. Use when asked to onboard a benchmark, create a HELM scenario, process a paper into evaluation code, convert a dataset to HELM format, or when the user mentions "creativity benchmark", "task.py", or "scenario.py".
allowed-tools: Read, Write, WebFetch, Grep, Glob, Bash, Task
user-invocable: true
---

# HELM Benchmark Onboarder

Transform creativity benchmark datasets into HELM Scenario implementations.

## Core Principles

1. **Dataset-first** - The dataset defines what's possible; the paper provides context
2. **Skip model outputs** - We want original prompts/stimuli, not what GPT-4 generated
3. **Verify everything** - Every field must map to real data

## Resources

- For HELM code patterns, see [helm-template.md](helm-template.md)
- For working examples, see [examples/](examples/)
  - [brainteaser.py](examples/brainteaser.py) - Multiple choice with distractor answers
  - [analobench.py](examples/analobench.py) - Analogical reasoning MC task
  - [riddlesense.py](examples/riddlesense.py) - Riddle QA with CommonsenseQA format
- For benchmark queue, see [benchmarks.json](benchmarks.json)

## Tool Usage

- **WebFetch**: Read dataset documentation, READMEs, and papers
- **Read**: Examine dataset files, existing scenarios, and example code
- **Bash**: Load dataset samples to inspect field structure
- **Glob**: Find relevant files in benchmark repos
- **Task (Explore)**: For complex benchmarks, explore dataset structure first
- **Write**: Generate the final scenario.py

## When to Use

User says things like:
- "Onboard the BRAINTEASER benchmark"
- "Create a HELM scenario for this paper"
- "Process this benchmark into evaluation code"

## Workflow

### Step 1: Qualify the Benchmark

Before doing any work, check if this benchmark is suitable:

**Ask yourself:**
- Is this a creativity benchmark?
- Is there a publicly available dataset?
- Is there an evaluation method? (accuracy, human ratings, metrics)

If the answer to any is NO, tell the user why this benchmark doesn't qualify and stop.

**Focus on primary creativity tasks.** Some papers include multiple tasks or secondary experiments (e.g., MMLU baselines, general QA comparisons). Identify and implement only the core creativity evaluation tasks described in the paper.

### Handling Multi-Task Benchmarks

Some papers contain multiple tasks, not all creativity-related. When onboarding:

1. **Identify the primary creativity task(s)** - Read the paper abstract/introduction to understand the main contribution
2. **Skip secondary experimental tasks** - Papers often include baseline comparisons (e.g., MMLU, general QA) that aren't creativity benchmarks
3. **When unclear, ask the user** - If you can't determine which tasks are the main creativity evaluations, ask before proceeding

### Step 2: Examine the Dataset

Load a few examples and understand the structure:

```python
from datasets import load_dataset
ds = load_dataset("org/dataset-name", split="test")
print(ds[0])  # See field names and structure
```

**Identify fields:**
- **Stimulus/Question**: The input text (question, prompt, story, etc.)
- **Choices**: For MC tasks, the answer options
- **Answer/Label**: The correct answer or ground truth
- **Metadata**: Split info, IDs, categories

**Use judgment on fields:** Avoid fields that contain model-generated outputs (e.g., fields named "gpt_response" or containing experiment results). Long text fields may be legitimate inputs for evaluation tasks (e.g., stories to analyze, passages to read). When uncertain, check the paper to understand what each field represents.

### Step 3: Check for Task Instructions

Some benchmarks have specific prompt wording. Check:

1. **Dataset README** on HuggingFace/GitHub - often has example prompts
2. **Paper Methods/Appendix** - if specific instructions were given to annotators/models

**Priority order:**
1. Explicit instructions from paper/README → use exactly as written
2. Standard format for task type → note "Standard MC format" in header
3. Unclear → ask user

Many MC tasks work fine with standard formatting ("Question: X\n\nA. ... B. ..."), but when papers DO specify exact wording, use it.

### Step 4: Generate the HELM Scenario

Follow this structure:

```python
"""
HELM Scenario: BENCHMARK_NAME

Prompt source: [Paper Section X / Dataset README / Standard MC format]
Fields used: field1, field2, field3
Fields skipped: none / field4 (model outputs)

Paper: [URL or citation]
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)

class BenchmarkScenario(Scenario):
    name = "benchmark_name"  # lowercase, underscores
    description = "org/dataset-name"  # data source, not task description
    tags = ["creativity", "relevant_tag"]

    def get_instances(self, output_path):
        dataset = load_dataset("org/dataset-name", split="test")

        instances = []
        for item in dataset:
            # Build prompt from dataset fields
            prompt = f"Question: {item['question']}\n"

            # Build references (all choices for MC, correct answer tagged)
            references = [...]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))
        return instances
```

**HELM conventions:**
- For multiple choice: ALL choices become References, only correct one gets CORRECT_TAG
- For binary (yes/no): Both options are References
- For open-ended: Reference can be empty or contain gold response
- `description` field = data source, not task description

### Step 5: Verify Before Finishing

Check these before delivering:

- [ ] Dataset loads without errors
- [ ] All template variables map to real data fields
- [ ] No model output fields used as inputs
- [ ] References have non-empty text (for MC/closed tasks)
- [ ] Correct split used (test if labels available, validation otherwise)
- [ ] Test with a few examples to confirm formatting

## Output

Create `scenario.py` with:
- Header comment noting prompt source, fields used/skipped, paper reference
- Clean, minimal code following HELM patterns

### Auto-Capture Notes to CLAUDE.md

When you encounter issues or discover patterns, append directly to CLAUDE.md:

1. Read the current CLAUDE.md file
2. Add your note to the appropriate section (Dataset Quirks table, Common Patterns, etc.)
3. Write the updated file

**Always capture notes for:**
- Split issues (test has no labels, etc.)
- Field name mismatches (docs say X, actual field is Y)
- Special loading requirements (trust_remote_code, config names)
- Skipped tasks (secondary experiments, non-creativity baselines, etc.)

This ensures team knowledge is captured immediately without manual intervention.

## Common Issues

| Problem | Solution |
|---------|----------|
| Test split has no labels | Use validation split instead (see riddlesense.py) |
| Dataset requires special loading | Add `trust_remote_code=True` to load_dataset() |
| Field names don't match docs | Print `ds[0]` to see actual field names |
| Empty references | Wrong answer field - check the schema |
| Very long prompts | Might be using wrong field (model outputs) |
| Multimodal benchmark | Include if there's a text evaluation component; note modality in tags |
| No explicit prompt in paper | Use standard formatting, note in header |

## Complex Benchmarks

For benchmarks with multiple subsets or complex evaluation:

1. Use **Task tool with Explore agent** to understand the dataset structure first
2. Break into sub-tasks if multiple scenario files are needed
3. Use `/compact` between benchmarks if onboarding multiple in one session

## Examples

### Example A: Paper specifies instructions (ANALOBENCH)

1. **Qualify**: Creativity benchmark (analogical reasoning)? Yes. Dataset? Yes. Eval? Yes (accuracy).

2. **Examine dataset**:
   ```python
   ds = load_dataset("jhu-clsp/AnaloBench", "T1S1-Subset", split="train")
   print(ds[0].keys())  # ['Index', 'Sentence', 'Options', 'Label']
   ```

3. **Check instructions**: Paper Section 3 specifies "Which of the following is the most analogous story?" → use exactly

4. **Generate scenario**: Paper instruction + dataset fields, MC pattern

5. **Verify**: 340 examples, Labels are A/B/C/D

### Example B: Standard format (RiddleSense)

1. **Qualify**: Creativity benchmark (riddles + commonsense)? Yes. Dataset? Yes. Eval? Yes.

2. **Examine dataset**: Fields are `question`, `choices`, `answerKey`

3. **Check instructions**: No specific wording in paper → use standard MC format, note in header

4. **Generate scenario**: Standard "Question: X\n\nA. ..." format

5. **Verify**: Test split has no labels → switch to validation split
