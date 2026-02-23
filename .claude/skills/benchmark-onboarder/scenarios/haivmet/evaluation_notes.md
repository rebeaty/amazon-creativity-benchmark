# HAIVMet Evaluation Methodology

## Original Papers

**Primary Paper:** "I Spy a Metaphor: Large Language Models and Diffusion Models Co-Create Visual Metaphors"
- Authors: Tuhin Chakrabarty, Arkadiy Saakyan, Olivia Winn, Artemis Panagopoulou, Yue Yang, Marianna Apidianaki, Smaranda Muresan
- Published: ACL 2023 (Findings)
- ArXiv: https://arxiv.org/abs/2305.14724
- GitHub: https://github.com/tuhinjubcse/VisualMetaphors
- Dataset: https://zenodo.org/records/8011133

**V-FLUTE Paper:** "V-FLUTE: Visual Figurative Language Understanding with Textual Explanations"
- Authors: Arkadiy Saakyan et al.
- Published: NAACL 2024
- ArXiv: https://arxiv.org/abs/2405.01474
- GitHub: https://github.com/asaakyan/V-FLUTE
- Dataset: https://huggingface.co/datasets/ColumbiaNLP/V-FLUTE (gated)

---

## Dataset Creation (ACL 2023)

### Purpose

Create high-quality visual metaphors using Human-AI collaboration, addressing the limitation that existing text-to-image models struggle to generate images with visual metaphors.

### Creation Process

**Step 1: Linguistic Metaphor Collection**

Sources for 1,540 linguistic metaphors:
- FLUTE dataset
- Advertisement slogans
- CoPoet (poetry corpus)
- FigQA (figurative QA)
- Figure-of-Speech corpus
- CrossLing Metaphors
- Metaphor Paraphrase dataset

**Step 2: Visual Elaboration Generation**

- Use GPT-3 (Instruct) with Chain-of-Thought prompting
- Generate textual descriptions elaborating how to visually represent the metaphor
- Expert annotators validate and optionally edit elaborations
- Average elaboration length: ~50 words

Example:
```
Linguistic Metaphor: "Time is money"
Visual Elaboration: "A clock face made of gold coins, with each hour marked
by a dollar bill. The clock hands are shaped like credit cards, and coins
are falling from the clock like sand in an hourglass."
```

**Step 3: Image Generation**

- Use DALL-E 2 with the visual elaboration as prompt
- Generate 4-6 candidate images per metaphor
- Expert illustrators filter low-quality images
- Keep high-quality images that successfully convey the metaphor

**Step 4: Quality Control**

- Professional illustrators evaluate images
- Criteria:
  - Does the image convey the metaphor?
  - Is the image visually coherent?
  - Does it match the elaboration?
- Final dataset: 6,476 images across 1,540 metaphors

### Dataset Statistics (HAIVMet Original)

- **Total linguistic metaphors:** 1,540
- **Total images:** 6,476
- **Average images per metaphor:** 4.2
- **Image source:** DALL-E 2
- **Image style:** Illustration/artistic (not photorealistic)
- **License:** CC-BY-4.0

---

## Original Evaluation (ACL 2023)

### Task: Visual Entailment

**Input:** Image (visual metaphor) + Hypothesis (textual statement)
**Output:** Entailment, Contradiction, or Neutral

### Dataset Construction for Evaluation

**Sources:**
- FLUTE dataset metaphors
- CrossLing Metaphors
- Metaphor Paraphrases

**Process:**
1. Take linguistic metaphors from sources
2. Generate HAIVMet images for these metaphors
3. Create literal hypotheses for each metaphor
4. Annotate entailment relationship (3 annotators)
5. Assign labels by majority vote

**Annotation Process:**
- 3 annotators independently label each hypothesis
- Labels: entailment, contradiction, neutral
- Mean pairwise agreement: 0.79
- Gold label: majority vote

### Evaluation Dataset Statistics

- **Total metaphors:** 958
- **Splits:**
  - Train: 708 metaphors
  - Validation: 100 metaphors
  - Test: 150 metaphors
- **Total image-text pairs:** 3,686

### Model Evaluated

**OFA (One For All)** - Unified multimodal model

### Results

| Setting | Test Accuracy |
|---------|--------------|
| OFA (pre-trained only) | 27.81% |
| OFA (fine-tuned on HAIVMet) | 51.15% |
| **Improvement** | **+23.34 points** |

**Key Finding:** Fine-tuning on HAIVMet visual metaphors significantly improves understanding of figurative visual content.

### Metrics

- **Primary:** Accuracy (3-way classification)
- **Inter-annotator agreement:** Mean pairwise agreement (0.79)

---

## V-FLUTE Extension (NAACL 2024)

### Expanded Task: Explainable Visual Entailment

V-FLUTE extends the visual entailment task to include textual explanations and covers broader figurative phenomena.

### Dataset Composition

Total: 6,027 instances across 5 phenomena

| Phenomenon | Source Dataset | Count | Labels |
|------------|---------------|-------|--------|
| **Metaphor/Simile** | **HAIVMet** | **857** | **Entailment/Contradiction** |
| Metaphor/Simile | IRFL | 1,149 | Entailment/Contradiction |
| Idiom | IRFL | 370 | Entailment/Contradiction |
| Sarcasm | MuSE | 1,042 | Contradiction only |
| Humor | MemeCap | 1,958 | Entailment only |
| Humor | NYCartoons | 651 | Entailment only |

**HAIVMet contribution:** 857 metaphor/simile instances

### Task Format

**Input:**
- Image (visual metaphor or other figurative content)
- Claim (textual hypothesis)

**Output:**
- Label: Entailment or Contradiction (binary in V-FLUTE)
- Explanation: Textual reasoning for the label

### Dataset Splits

- **Train:** 4,578 instances
- **Validation:** 726 instances
- **Test:** 723 instances

### Fields

```python
{
    "image": PIL.Image,
    "claim": str,
    "label": str,  # "entailment" or "contradiction"
    "explanation": str,
    "phenomenon": str,  # "metaphor", "simile", "idiom", "sarcasm", "humor"
    "source_dataset": str  # "HAIVMet", "IRFL", "MuSE", "MemeCap", "NYCartoons"
}
```

### Evaluation Metrics

**Primary Metrics:**

1. **F1 Score (binary classification)**
   - Standard F1 for entailment/contradiction prediction
   - Reported as F1@0 (no explanation threshold)

2. **ExplanationScore**
   - Measures quality of generated explanations
   - Average of BERTScore and BLEURT
   - Range: 0-100

3. **F1@ExplanationScore**
   - F1 score that only counts predictions correct if:
     - Label is correct AND
     - ExplanationScore ≥ threshold
   - Thresholds: F1@53, F1@60

**Metric Formulas:**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

ExplanationScore = (BERTScore + BLEURT) / 2

F1@T = F1 score where correct = (label_correct AND explanation_score ≥ T)
```

### Models Evaluated (V-FLUTE Paper)

**Vision-Language Models:**
1. GPT-4V (OpenAI)
2. Gemini Pro Vision (Google)
3. LLaVA-1.5-7B
4. LLaVA-1.5-13B
5. InstructBLIP
6. BLIP-2

**Results (Test Set):**

| Model | F1@0 | F1@53 | F1@60 |
|-------|------|-------|-------|
| GPT-4V | 76.8 | 41.2 | 32.1 |
| LLaVA-1.5-13B | 72.4 | 38.5 | 29.7 |
| LLaVA-1.5-7B (fine-tuned) | 81.3 | 52.4 | 43.8 |

**Key Findings:**
- Large drop in performance when accounting for explanation quality
- Fine-tuning significantly helps both prediction and explanation
- Generating faithful explanations for figurative content remains challenging

### Prompt Format (V-FLUTE)

The paper tested 21 different instruction variations. Example:

```
Does the image's narrative confirm or disprove the claim?
Discuss your reasoning and identify it as either entailment or contradiction.

Claim: {claim}
```

Other variations include:
- "Analyze the correspondence between the image and claim..."
- "Determine if the image and claim are in harmony or opposition..."
- "Examine whether the image validates or invalidates the claim..."

---

## HELM Adaptation

### Task Simplification

**Adapted from:** V-FLUTE's explainable visual entailment
**Simplified to:** Binary visual entailment classification

**Changes:**
- **Input:** Image + Claim (same)
- **Output:** Label only (entailment or contradiction)
- **Dropped:** Explanation generation (not typical for HELM scenarios)

**Rationale:**
- HELM scenarios focus on core task performance
- Explanations can be evaluated separately if needed
- Binary classification is cleaner and more standard
- Allows direct comparison with other entailment benchmarks

### HELM Scenario Configuration

**Dataset Source:** ColumbiaNLP/V-FLUTE (filtered for HAIVMet)

**Filtering Options:**

1. **HAIVMet-only mode** (default):
   ```python
   HAIVMetScenario(use_full_vflute=False)
   # Returns 857 metaphor/simile instances from HAIVMet source
   ```

2. **Full V-FLUTE mode:**
   ```python
   HAIVMetScenario(use_full_vflute=True)
   # Returns all 6,027 instances across 5 phenomena
   ```

3. **Custom phenomenon filter:**
   ```python
   HAIVMetScenario(
       use_full_vflute=True,
       filter_phenomena=["metaphor", "simile", "idiom"]
   )
   # Returns only specified phenomena
   ```

### Evaluation Type

**Metric:** Binary classification accuracy

**Why not F1?**
- HELM typically uses accuracy for classification tasks
- F1 can be computed separately if needed
- Explanation quality (F1@ExplanationScore) not applicable without explanations

### Prompt Format

Standard visual entailment format:
```
Does the image's narrative confirm or disprove the claim?

[Image]

Claim: {claim}

Answer: Entailment or Contradiction
```

### Dataset Access

**Important:** V-FLUTE is a **gated dataset** on HuggingFace.

**To access:**

1. Visit https://huggingface.co/datasets/ColumbiaNLP/V-FLUTE
2. Click "Access repository" and accept terms
3. Authenticate locally:
   ```bash
   huggingface-cli login
   # Or set HF_TOKEN environment variable
   ```

**License:** Apache 2.0 (requires accepting terms)

---

## Comparison: Original vs HELM

| Aspect | ACL 2023 (Original) | NAACL 2024 (V-FLUTE) | HELM Adaptation |
|--------|---------------------|----------------------|-----------------|
| **Task** | Visual entailment (3-way) | Explainable entailment (binary) | Visual entailment (binary) |
| **Labels** | Entailment, Contradiction, Neutral | Entailment, Contradiction | Entailment, Contradiction |
| **Dataset** | HAIVMet (958 metaphors) | V-FLUTE (6,027 instances) | V-FLUTE (filtered for HAIVMet) |
| **HAIVMet Size** | 958 metaphors | 857 instances | 857 instances |
| **Output** | Label only | Label + Explanation | Label only |
| **Metrics** | Accuracy (3-way) | F1, F1@ExplanationScore | Accuracy (binary) |
| **Models** | OFA | GPT-4V, LLaVA, BLIP-2, etc. | Any vision-language LLM |
| **Splits** | 708/100/150 | 4,578/726/723 | 4,578/726/723 |
| **Access** | Zenodo (open) | HuggingFace (gated) | HuggingFace (gated) |

---

## Why This Is a Creativity Benchmark

### Visual Metaphor Understanding

**Metaphor comprehension requires:**
1. **Literal interpretation** - Understanding what is literally depicted
2. **Conceptual mapping** - Recognizing symbolic/abstract relationships
3. **Contextual reasoning** - Understanding intended meaning beyond visuals

Example:
```
Image: Clock face made of melting dollar bills
Claim: "Time is a valuable resource"
Label: Entailment

Reasoning: The image metaphorically represents time's value by depicting
it as money (valuable resource), requiring understanding of both the visual
metaphor and its conceptual mapping.
```

### Creativity Evaluation Dimensions

1. **Metaphorical thinking** - Can models interpret non-literal visual representations?
2. **Conceptual blending** - Do models understand how concepts are merged in visual metaphors?
3. **Abstraction** - Can models reason about abstract ideas conveyed visually?
4. **Figurative language** - Do models grasp figurative meanings in multimodal context?

### Benchmark Uniqueness

- **First large-scale visual metaphor dataset** for multimodal evaluation
- **Human-AI collaboration** ensures high-quality creative examples
- **Expert validation** by professional illustrators
- **Diverse metaphor sources** across domains (poetry, ads, figurative speech)

---

## Alternative HELM Evaluation Approaches

### Option 1: Include Neutral Label (Original Task)

Revert to 3-way classification matching original ACL 2023 task:

**Labels:** Entailment, Contradiction, Neutral

**Challenge:** V-FLUTE doesn't include neutral labels; would need original HAIVMet data

### Option 2: Explanation Generation

Add explanation generation following V-FLUTE:

**Task:** Predict label + generate reasoning

**Output:**
```
Label: Entailment
Explanation: The image depicts time as money through a clock made of coins,
directly supporting the claim that time is a valuable resource.
```

**Evaluation:**
- Label accuracy
- Explanation quality (BERTScore, BLEURT vs gold)
- F1@ExplanationScore

**HELM Configuration:** `get_open_ended_generation_metric_specs()`

### Option 3: Multi-Phenomenon Evaluation

Test across all V-FLUTE phenomena separately:

**Subsets:**
- Metaphor understanding (HAIVMet + IRFL)
- Simile understanding (HAIVMet + IRFL)
- Idiom understanding (IRFL)
- Sarcasm detection (MuSE)
- Humor understanding (MemeCap, NYCartoons)

**Benefit:** Broader figurative language evaluation

### Option 4: Visual Elaboration as Hint

Provide visual elaboration text as context:

**Input:**
```
Visual Elaboration: "A clock face made of gold coins..."

[Image]

Claim: "Time is money"
```

**Evaluation:** Does additional context help model understanding?

---

## Recommendations

### Current Implementation ✓

**Binary classification without explanations** because:
- Clean, standard evaluation
- Aligns with HELM patterns
- Focuses on core metaphor understanding
- Easier to compare across models

### Future Enhancements

1. **Add explanation variant** to evaluate reasoning quality
2. **Test on full V-FLUTE** to measure broader figurative language understanding
3. **Compare with original 3-way task** if original data becomes available
4. **Fine-tuning experiments** following ACL 2023's significant improvement

### Research Questions

- Do models pre-trained on more artistic data perform better?
- How does performance vary across metaphor domains (poetry vs ads)?
- Can chain-of-thought prompting improve metaphor understanding?
- Do models struggle more with abstract vs concrete metaphors?

---

## Dataset Quality and Limitations

### Strengths

✓ Expert-validated visual metaphors
✓ High inter-annotator agreement (0.79)
✓ Diverse metaphor sources
✓ Large-scale (6,476 images)
✓ Human-AI collaboration ensures quality

### Limitations

⚠ Images generated by DALL-E 2 (not human-created art)
⚠ Illustration style may not generalize to all visual metaphors
⚠ English-only linguistic metaphors
⚠ Gated dataset access (requires authentication)
⚠ Binary labels in V-FLUTE (neutral removed from original)

### Potential Biases

- DALL-E 2 generation biases may influence visual style
- Metaphor sources skewed toward academic datasets
- Expert annotators may have shared cultural background

---

## References

**Primary Papers:**
```bibtex
@inproceedings{chakrabarty2023spy,
  title={I Spy a Metaphor: Large Language Models and Diffusion Models Co-Create Visual Metaphors},
  author={Chakrabarty, Tuhin and Saakyan, Arkadiy and Winn, Olivia and Panagopoulou, Artemis and Yang, Yue and Apidianaki, Marianna and Muresan, Smaranda},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  year={2023}
}

@inproceedings{saakyan2024understanding,
  title={Understanding Figurative Meaning through Explainable Visual Entailment},
  author={Saakyan, Arkadiy and others},
  booktitle={NAACL 2024},
  year={2024}
}
```

**Links:**
- ACL 2023 Paper: https://arxiv.org/abs/2305.14724
- V-FLUTE Paper: https://arxiv.org/abs/2405.01474
- Original Dataset: https://zenodo.org/records/8011133
- V-FLUTE Dataset: https://huggingface.co/datasets/ColumbiaNLP/V-FLUTE
- GitHub (Original): https://github.com/tuhinjubcse/VisualMetaphors
- GitHub (V-FLUTE): https://github.com/asaakyan/V-FLUTE
- HELM Scenario: `scenarios/haivmet/haivmet_scenario.py`
