# FLUTE (Filtered): Rhetorical Language Understanding

**Original FLUTE Paper:** FLUTE: Figurative Language Understanding through Textual Explanations (EMNLP 2022)
**Filtering Paper:** Rhetorical Text-to-Image Generation via Two-layer Diffusion Policy Optimization (May 2025)
**arXiv:** [2205.12404](https://arxiv.org/abs/2205.12404) (FLUTE), [2505.22792](https://arxiv.org/abs/2505.22792) (Rhet2Pix)
**Dataset:** [ColumbiaNLP/FLUTE](https://huggingface.co/datasets/ColumbiaNLP/FLUTE)
**Code:** [Rhet2Pix GitHub](https://github.com/zyxxxxx-39/Rhet2Pix)

## Overview

FLUTE (Filtered) is a subset of the original FLUTE dataset, focusing on **metaphors and similes** - the two types of figurative language with high rhetorical clarity and visual interpretability.

The Rhet2Pix paper uses this filtered version for training text-to-image models to understand rhetorical language, selecting these examples because they can be effectively visualized while maintaining clear figurative meaning.

---

## Original FLUTE Dataset

**FLUTE** (Figurative Language Understanding through Textual Explanations) is a comprehensive dataset for evaluating understanding of figurative language.

### Full FLUTE Statistics
- **Total examples:** 7,534
- **Task:** Figurative Natural Language Inference (NLI) with explanations
- **Format:** Premise-Hypothesis pairs with labels and explanations

### Figurative Types in Full FLUTE
| Type | Examples | Description |
|------|----------|-------------|
| **Metaphor** | 1,250 | Figurative comparisons ("My soul was a lampless sea") |
| **Simile** | 1,250 | Explicit comparisons ("works like a charm") |
| Sarcasm | ~1,500 | Ironic/contradictory statements |
| Idiom | ~1,500 | Fixed expressions with non-literal meaning |
| CreativeParaphrase | ~2,000 | Unconventional rewordings |

---

## FLUTE (Filtered) Subset

### Filtering Criteria (from Rhet2Pix Paper)

> "We adopt the FLUTE dataset and apply an additional filtering step to select high-quality metaphor and simile samples, ensuring both rhetorical clarity and visual interpretability."

**Included Types:**
- ✅ **Metaphor** (1,250 examples)
- ✅ **Simile** (1,250 examples)

**Excluded Types:**
- ❌ Sarcasm (less visually interpretable)
- ❌ Idiom (culturally specific, literal imagery)
- ❌ CreativeParaphrase (focus on wording not imagery)

**Total Filtered Dataset:** 2,500 examples

---

## Task Description

### Task Format: Figurative NLI with Explanations

**Input:** Premise-Hypothesis pair containing figurative language

**Output:**
1. **Classification:** Entailment or Contradiction
2. **Explanation:** Natural language reasoning for the inference

### Example 1: Metaphor

**Premise:**
> "My soul was a complete mess and she was the cause of it."

**Hypothesis:**
> "My soul was a lampless sea and she was the tempest."

**Label:** Entailment

**Explanation:**
> "A complete mess and a tempest both convey chaos and disorder. A soul being described as a lampless sea implies darkness and emptiness, similar to being a complete mess."

### Example 2: Simile

**Premise:**
> "This new software update works perfectly for me."

**Hypothesis:**
> "This new software update works like a charm for me."

**Label:** Entailment

**Explanation:**
> "'Works like a charm' is an idiomatic expression meaning something works perfectly or smoothly, which directly matches the premise."

---

## Semantic Dimensions (Rhet2Pix Framework)

For downstream rhetorical text-to-image generation, the Rhet2Pix paper extracts **7 semantic dimensions** from each figurative expression:

1. **Rhetorical Device:** metaphor or simile
2. **Literal Subject:** The actual topic being discussed
3. **Metaphorical Vehicle:** The figurative comparison used
4. **Theme:** Overall conceptual theme
5. **Emotional Tone:** Sentiment and mood
6. **Subject Keywords:** Key terms for the literal subject
7. **Vehicle Keywords:** Key terms for the figurative vehicle

### Extraction Process

The paper uses GPT-4o with a **generate-verify-retry loop** to extract these dimensions with high coherence and consistency.

**Example Extraction:**

| Dimension | Value |
|-----------|-------|
| Rhetorical Device | Metaphor |
| Literal Subject | Soul/emotional state |
| Metaphorical Vehicle | Lampless sea |
| Theme | Chaos, darkness, emptiness |
| Emotional Tone | Despair, turmoil |
| Subject Keywords | soul, mess, emotional state |
| Vehicle Keywords | sea, lampless, dark, tempest |

---

## Evaluation Metrics

### 1. Classification Accuracy

**Task:** Predict Entailment vs. Contradiction

**Metric:** Exact match accuracy

**Baseline Performance (from original FLUTE paper):**
- T5-base fine-tuned: ~85% accuracy
- GPT-3 few-shot: ~70% accuracy

### 2. Explanation Quality

**Task:** Generate natural language explanations

**Metrics:**
- BLEU (n-gram overlap)
- ROUGE-L (longest common subsequence)
- METEOR (synonym-aware matching)
- BERTScore (semantic similarity)

### 3. Combined Evaluation

Both classification correctness AND explanation quality matter. A model that predicts the correct label but provides poor reasoning is less useful than one that explains its inference clearly.

---

## Why Metaphors and Similes?

The Rhet2Pix paper filters for these types because they have unique properties for vision-language tasks:

### Advantages of Metaphors/Similes

1. **Visual Interpretability**
   - Can be rendered as images
   - Concrete vehicles (sea, tempest, lamp) are visualizable
   - Support text-to-image generation tasks

2. **Rhetorical Clarity**
   - Clear figurative structure
   - Explicit tenor-vehicle mapping
   - Easier to extract semantic dimensions

3. **Creative Understanding**
   - Test genuine figurative language comprehension
   - Require conceptual mapping between domains
   - Go beyond literal text understanding

### Contrast with Excluded Types

| Type | Issue for Vision Tasks |
|------|------------------------|
| **Sarcasm** | No visual component, requires tone/context |
| **Idiom** | Often has literal imagery that doesn't match meaning ("kick the bucket") |
| **CreativeParaphrase** | Focuses on wording, not visual metaphor |

---

## Dataset Structure

### Fields per Example

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique identifier |
| `premise` | string | Base statement with figurative language |
| `hypothesis` | string | Statement to evaluate against premise |
| `label` | string | "Entailment" or "Contradiction" |
| `explanation` | string | Natural language reasoning (0-368 chars) |
| `type` | string | "Metaphor" or "Simile" (filtered subset) |
| `idiom` | string/null | (Not applicable for metaphor/simile) |
| `split` | string | "train" (all examples in single split) |

### Data Distribution

**Metaphors:** 1,250 examples
- Various source domains (nature, machines, emotions, etc.)
- Diverse target concepts
- Range of complexity

**Similes:** 1,250 examples
- Explicit comparison markers ("like", "as")
- Often idiomatic ("works like a charm")
- Varying levels of conventionality

**Total:** 2,500 examples (33% of original FLUTE)

---

## Implementation Details

### Scenario Features

```python
# Load with both classification and explanation evaluation
scenario = FLUTEFilteredScenario(include_explanations=True)

# Load for classification only (faster evaluation)
scenario = FLUTEFilteredScenario(include_explanations=False)
```

### Metadata in Instance.extra_data

Each instance includes:
```python
instance.extra_data = {
    'type': 'Metaphor',  # or 'Simile'
    'premise': '...',
    'hypothesis': '...',
    'label': 'Entailment',
    'explanation': '...',
    'original_id': 42,
}
```

### Output Format

**With explanations:**
```
Entailment

Explanation: [reasoning text]
```

**Classification only:**
```
Entailment
```

---

## Key Findings

### Original FLUTE Paper (2022)

1. **Models struggle with figurative language**
   - Pre-trained models show limited understanding
   - Fine-tuning helps but doesn't solve the problem
   - Explanations improve model performance

2. **Human-AI collaboration effective**
   - Model-in-the-loop data collection with GPT-3
   - Crowd workers + expert annotators ensure quality
   - Scales creation of complex linguistic datasets

### Rhet2Pix Paper (2025)

1. **Metaphors/similes enable vision-language tasks**
   - Successfully generate rhetorical images
   - Outperform GPT-4o and other MLLMs
   - Multi-step reasoning improves results

2. **Semantic decomposition helps**
   - Breaking down into 7 dimensions improves understanding
   - Sequential prompt enrichment works better than single-shot
   - Verification loops ensure quality

---

## Comparison to Other Benchmarks

| Aspect | FLUTE (Filtered) | Traditional NLI |
|--------|------------------|-----------------|
| **Language** | Figurative | Literal |
| **Task** | Entailment + Explanation | Entailment only |
| **Types** | Metaphor, Simile | Factual statements |
| **Evaluation** | Accuracy + Explanation quality | Accuracy |
| **Downstream** | Text-to-image generation | General NLU |
| **Size** | 2,500 examples | 10K-550K examples |

---

## Usage Example

```python
from helm.benchmark.scenarios.scenario import Scenario
from scenarios.flute_filtered.scenario import FLUTEFilteredScenario

# Load filtered dataset
scenario = FLUTEFilteredScenario(include_explanations=True)
instances = scenario.get_instances(output_path="./data")

# Example instance
instance = instances[0]
print(instance.input.text)
print(instance.references[0].output)
print(instance.extra_data['type'])  # "Metaphor" or "Simile"
```

---

## Citation

**Original FLUTE:**
```bibtex
@inproceedings{chakrabarty2022flute,
    title={FLUTE: Figurative Language Understanding through Textual Explanations},
    author={Chakrabarty, Tuhin and Saakyan, Arkadiy and Muresan, Smaranda},
    booktitle={EMNLP},
    year={2022}
}
```

**Rhet2Pix (Filtering Application):**
```bibtex
@article{rhet2pix2025,
    title={Rhetorical Text-to-Image Generation via Two-layer Diffusion Policy Optimization},
    author={[Authors]},
    journal={arXiv preprint arXiv:2505.22792},
    year={2025}
}
```

---

## Resources

- 📄 [Original FLUTE Paper (EMNLP 2022)](https://arxiv.org/abs/2205.12404)
- 📄 [Rhet2Pix Paper (May 2025)](https://arxiv.org/abs/2505.22792)
- 🤗 [FLUTE Dataset (HuggingFace)](https://huggingface.co/datasets/ColumbiaNLP/FLUTE)
- 💻 [Rhet2Pix Code (GitHub)](https://github.com/zyxxxxx-39/Rhet2Pix)
- 💻 [Original FLUTE Code](https://github.com/tuhinjubcse/model-in-the-loop-fig-lang)

---

## Future Work

- **Multimodal Extension:** Combine with image generation evaluation
- **Cross-lingual:** Extend to other languages
- **Fine-grained Types:** Distinguish conventional vs. novel metaphors
- **Difficulty Levels:** Stratify by complexity of figurative mapping
- **Semantic Extraction:** Automate the 7-dimension extraction pipeline
