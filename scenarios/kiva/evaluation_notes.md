# Evaluation Notes: KiVA (Kid-inspired Visual Analogies)

Source: Paper "KiVA: Kid-inspired Visual Analogies for Testing Large Multimodal Models" (ICLR 2025)
Dataset: https://github.com/ey242/KiVA/releases/tag/0.1

## Evaluation Type: Exact Match

KiVA uses **standard multiple-choice evaluation** with exact string matching. No custom annotators or LLM judges are required.

### Task Format

**Input**:
- Visual puzzle showing a transformation in the top row
- Three candidate transformations in the bottom row (A, B, C)
- Text question asking which bottom transformation matches the top

**Expected Output**:
- Single letter answer: `(A)`, `(B)`, or `(C)`
- Alternative acceptable formats: `A`, `B`, `C` (without parentheses)

**Correct Answer**:
- Stored in JSON metadata for each trial
- Example: `"correct": "(A)"` means option A is the right answer

## Evaluation Metrics

### Primary Metric: Accuracy

```
Accuracy = (Number of Correct Answers) / (Total Instances)
```

**Implementation**: Use HELM's `get_exact_match_metric_specs()`

**What counts as correct**:
- Model output exactly matches the correct answer: `(A)` = `(A)` ✓
- May need normalization for variations: `A` → `(A)`, ` (A) ` → `(A)`

### Per-Domain Breakdown

Compute accuracy separately for each transformation domain:

| Domain | Instances | Unique Trials | Human Baseline | GPT-o1 Performance |
|--------|-----------|---------------|----------------|-------------------|
| 2DRotation | 450 (150) | 150 | ~85% | ~65% |
| Colour | 450 (150) | 150 | ~90% | ~75% |
| Counting | 600 (200) | 200 | ~80% | ~60% |
| Reflect | 300 (100) | 100 | ~85% | ~62% |
| Resize | 300 (100) | 100 | ~88% | ~68% |

*(Numbers in parentheses are deduplicated counts)*

### Deduplication Consideration

**Important**: Each trial is repeated 3 times with randomized answer positions (A/B/C).

**Two evaluation modes**:

1. **All instances (2,100)**:
   - Includes all 3 repetitions
   - May introduce position bias
   - Higher sample size

2. **Deduplicated (700)**:
   - Only first repetition of each trial
   - Eliminates position bias
   - Recommended for fair comparison

**Recommendation**: Use `deduplicate=True` in the scenario for unbiased evaluation.

## Evaluation Protocol

### 1. Run Scenario

```python
from kiva_scenario import KiVAScenario

# Recommended: deduplicated for unbiased results
scenario = KiVAScenario(domain="all", deduplicate=True)
instances = scenario.get_instances(output_path)
```

### 2. Model Inference

For each instance:
- Show image + text prompt to vision-language model
- Model generates answer: `(A)`, `(B)`, or `(C)`
- Extract answer from model output

**Prompt format used**:
```
Which one of three left-to-right object transformations shown in the
bottom row is the same as the transformation shown in the top row?
Answer with the correct letter surrounded by parentheses.

Choices:
(A)
(B)
(C)

Answer:
```

### 3. Answer Extraction

**Challenge**: Models may generate verbose responses.

**Examples**:
- ✓ `(A)` → Extract `A`
- ✓ `The answer is (B) because...` → Extract `B`
- ✓ `I believe option C is correct` → Extract `C`
- ✗ `Both A and B seem plausible` → Unclear, mark as incorrect

**Extraction logic**:
```python
import re

def extract_answer(model_output):
    # Look for (A), (B), or (C) pattern
    match = re.search(r'\(([ABC])\)', model_output)
    if match:
        return match.group(1)

    # Look for standalone A, B, or C
    match = re.search(r'\b([ABC])\b', model_output)
    if match:
        return match.group(1)

    return None  # Extraction failed
```

### 4. Compute Metrics

```python
correct_count = 0
total_count = len(instances)

for instance in instances:
    correct_answer = get_correct_answer(instance)  # e.g., "A"
    model_answer = extract_answer(model_output)

    if model_answer == correct_answer:
        correct_count += 1

accuracy = correct_count / total_count
print(f"Overall Accuracy: {accuracy:.2%}")
```

### 5. Per-Domain Analysis

```python
# Group by domain
results_by_domain = {}

for instance in instances:
    domain = instance.id.split('_')[0]  # e.g., "Colour"

    if domain not in results_by_domain:
        results_by_domain[domain] = {'correct': 0, 'total': 0}

    results_by_domain[domain]['total'] += 1
    if is_correct(instance):
        results_by_domain[domain]['correct'] += 1

# Print per-domain results
for domain, stats in results_by_domain.items():
    acc = stats['correct'] / stats['total']
    print(f"{domain}: {acc:.2%} ({stats['correct']}/{stats['total']})")
```

## Expected Performance Ranges

Based on the paper's findings:

| Model Class | Expected Accuracy |
|-------------|-------------------|
| Random Baseline | 33.3% (1/3 chance) |
| Human Children (3-5 years) | 80-90% |
| GPT-4V | ~55-60% |
| GPT-o1 | ~65-70% |
| LLaVA-1.5 | ~40-45% |
| MANTIS | ~50-55% |

**Key findings**:
- Models struggle more with "how" transformations changed (quantification)
- Best at identifying "what" changed (classification)
- Children outperform state-of-the-art models

## Common Issues and Solutions

### Issue 1: Position Bias

**Problem**: Model consistently chooses option A or B regardless of content.

**Solution**: Use `deduplicate=True` to eliminate position bias from repeated trials.

### Issue 2: Verbose Responses

**Problem**: Model generates explanations instead of just the answer.

**Solution**: Improve answer extraction to handle verbose outputs. Consider few-shot prompting.

### Issue 3: Image Loading Failures

**Problem**: Images not found or corrupted.

**Solution**: Ensure data downloaded correctly. Re-download if needed:
```bash
curl -L https://github.com/ey242/KiVA/releases/download/0.1/single_image.zip -o single_image.zip
unzip single_image.zip
```

### Issue 4: Domain Imbalance

**Problem**: Counting domain has more instances (600 vs 300-450).

**Solution**: Report both overall accuracy and per-domain accuracy. Consider weighted average:
```python
# Weight by human-level difficulty
weights = {
    '2DRotation': 1.0,
    'Colour': 0.8,  # Easier for humans
    'Counting': 1.2, # Harder for humans
    'Reflect': 1.0,
    'Resize': 1.0
}
```

## RunSpec Configuration

In HELM's RunSpec system, configure as:

```python
def get_kiva_metric_specs():
    return get_exact_match_metric_specs()
```

**Additional metrics to consider**:
- Per-domain accuracy breakdown
- Position bias analysis (if using non-deduplicated data)
- Error analysis by transformation type

## Comparison to Human Performance

The paper includes human baselines:

**Children (ages 3-5)**:
- Tested on same visual puzzles
- ~80-90% accuracy overall
- Slightly better on color transformations
- Slightly worse on counting/resize

**Adults**:
- Tested on KiVA-adults (harder version, not included in this scenario)
- ~95% accuracy on standard KiVA

**Interpretation**: Models significantly underperform even young children, indicating visual analogical reasoning remains a challenge for current LMMs.

## Citation

When reporting results on KiVA, cite:

```bibtex
@inproceedings{yiu2025kiva,
  title={KiVA: Kid-inspired Visual Analogies for Testing Large Multimodal Models},
  author={Yiu, Eunice and Qraitem, Maha and et al.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
