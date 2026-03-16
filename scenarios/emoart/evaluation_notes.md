# EmoArt Evaluation Methodology

## Original Paper Evaluation (Text-to-Image Generation)

**Paper:** EmoArt: A Multidimensional Dataset for Emotion-Aware Artistic Generation
**Published:** ACM Multimedia 2025
**ArXiv:** https://arxiv.org/abs/2506.03652
**Project:** https://zhiliangzhang.github.io/EmoArt-130k/

### Task: Emotion-Aware Artistic Image Generation

The original paper evaluates **text-to-image diffusion models** on their ability to generate emotionally expressive and stylistically accurate artistic images.

**Input Prompt Format:**
```
Style + Arousal + Valence + Description

Example:
"Abstract Art, Low arousal, Positive valence: A playful arrangement of colored
dots and abstract shapes scattered across a neutral background. The use of
vibrant oranges, blues, and blacks offers a lively, rhythmic pattern."
```

### Models Evaluated

Seven state-of-the-art text-to-image diffusion models:

1. **FLUX.1-dev** - Base model
2. **FLUX.1-schnell** - Fast variant
3. **SDXL** - Stable Diffusion XL
4. **SD3.5** - Stable Diffusion 3.5
5. **PixArt-sigma**
6. **Playground**
7. **Openjourney**
8. **FLUX.1-dev-finetuned** - Fine-tuned on EmoArt dataset

### Evaluation Metrics

#### 1. Standard Image Generation Metrics

| Metric | Description | Better Value |
|--------|-------------|--------------|
| **FID** (Fréchet Inception Distance) | Measures distributional similarity between generated and real images | Lower |
| **SSIM** (Structural Similarity) | Measures structural and visual similarity | Higher |
| **PSNR** (Peak Signal-to-Noise Ratio) | Measures reconstruction quality | Higher |
| **LPIPS** (Learned Perceptual Image Patch Similarity) | Measures perceptual similarity using deep features | Lower |

#### 2. Attributes Alignment (Custom Metric)

**Purpose:** Evaluates semantic fidelity across five artistic attributes

**Attributes Evaluated:**
- Brushwork
- Composition
- Color
- Line quality
- Light and shadow

**Implementation:**
- Uses MiniCPM-V-2.6 fine-tuned on EmoArt dataset
- Computes similarity to ground-truth text in CLIP embedding space
- Provides per-attribute scores (0-1 scale)

**Sample Results (FLUX.1-dev-finetuned):**
- Brushstroke: 0.6388
- Color: 0.6974
- Composition: 0.6698
- Overall Quality: 0.6604

#### 3. Emotion Alignment

**Dimensions:**
- **Arousal:** High vs. Low
- **Valence:** Positive vs. Negative
- **Emotion Category:** One of 12 emotions (Alarmed, Annoyed, Aroused, Bored, Calm, Contentment, Excited, Frustrated, Glad, Happy, Sad, Tired)

**Evaluation:** Measured through Attributes Alignment metric and qualitative assessment

### Human Evaluation Protocol

**Annotators:** 10 trained human annotators

**Task:** Independently assess generated images on:
1. Description accuracy (does image match textual description?)
2. Visual attributes quality (brushwork, composition, color, line, light)
3. Emotional content (arousal, valence, dominant emotion)

**Dataset:** 5,600 images evaluated

**Metrics Used:**
- **Percent Agreement:** Overall agreement rate
- **Positive Agreement:** Agreement on positive cases
- **Gwet's AC1:** Inter-rater reliability coefficient (handles prevalence issues)
- **McNemar's Test:** Statistical test for paired nominal data

**Agreement Threshold:** >85% agreement required for annotations

### Key Findings from Original Paper

1. **Fine-tuning helps:** FLUX.1-dev-finetuned achieved highest scores across all attributes
2. **Emotion is hard:** Traditional metrics (FID, SSIM) showed moderate performance but didn't capture emotional nuances well
3. **Attributes matter:** Custom Attributes Alignment metric revealed more detailed insights into artistic quality
4. **Style diversity:** 56 different artistic styles tested model versatility

---

## HELM Adaptation: Vision-Language Classification

**Our Implementation:** `scenarios/emoart_scenario.py`

### Task Adaptation

We adapted EmoArt from **image generation** → **vision-language understanding**

**New Task:** Given an artwork image, classify the dominant emotion it evokes

**Rationale:**
- HELM primarily evaluates language/multimodal LLMs, not image generators
- Dataset has high-quality emotion annotations (GPT-4o + human verified >85%)
- Tests models' understanding of emotional content in visual art
- Enables evaluation of vision-language models (GPT-4V, Claude, Gemini)

### HELM Evaluation Method

**Evaluation Type:** `exact_match`

**Format:** Multiple choice classification
```
Question: What emotion does this artwork primarily evoke?

[Image]

A) Alarmed  B) Annoyed  C) Aroused  D) Bored
E) Calm  F) Contentment  G) Excited  H) Frustrated
I) Glad  J) Happy  K) Sad  L) Tired

Answer: [Model selects letter]
```

**Metric:** Simple accuracy (correct predictions / total predictions)

**Ground Truth:** `description.third_section.dominant_emotion` from dataset

### Differences from Original Paper

| Aspect | Original Paper | HELM Adaptation |
|--------|---------------|-----------------|
| **Task** | Text-to-image generation | Image-to-emotion classification |
| **Models** | Diffusion models (FLUX, SD, etc.) | Vision-language LLMs |
| **Input** | Text prompt → Generate image | Image + Question → Select emotion |
| **Output** | Generated artistic image | Letter choice (A-L) |
| **Metrics** | FID, SSIM, LPIPS, Attributes Alignment | Exact match accuracy |
| **Evaluation** | Compare generated vs. real images | Compare prediction vs. label |

---

## Alternative HELM Evaluation Approaches

### Option 1: Open-Ended Emotion Description

**Task:** Generate free-form description of artwork's emotional qualities

**Prompt:**
```
Analyze the emotional content of this artwork. Describe:
1. The dominant emotion evoked
2. The arousal level (high/low energy)
3. The emotional valence (positive/negative)
4. Visual elements contributing to the emotional impact

[Image]

Analysis:
```

**Evaluation Metrics:**
- BLEU-1, BLEU-4 (against reference descriptions)
- ROUGE-L (longest common subsequence)
- BERTScore (semantic similarity)
- F1 (token overlap)

**HELM Configuration:** `get_open_ended_generation_metric_specs()`

**Pros:**
- Captures nuanced understanding
- Tests artistic reasoning
- More creative/generative

**Cons:**
- Harder to evaluate objectively
- Reference-dependent
- May not align well with emotion labels

### Option 2: LLM-as-Judge Evaluation

**Task:** Generate emotion analysis, have GPT-4 judge quality

**Judge Prompt Template:**
```
You are an expert art critic evaluating AI-generated emotional analysis of artwork.

Ground Truth:
- Dominant Emotion: {GROUND_TRUTH_EMOTION}
- Arousal: {AROUSAL}
- Valence: {VALENCE}
- Visual Attributes: {VISUAL_ATTRIBUTES}

Model's Analysis:
{MODEL_RESPONSE}

Evaluate the model's analysis on a 1-5 scale for:
1. Emotion Accuracy: Does it correctly identify the dominant emotion?
2. Artistic Insight: Does it demonstrate understanding of visual elements?
3. Detail Quality: Does it provide specific, relevant observations?

Provide scores as:
Emotion Accuracy: [1-5]
Artistic Insight: [1-5]
Detail Quality: [1-5]
```

**Judge Model:** GPT-4-turbo or Claude-3.5-Sonnet

**Aggregation:** Average scores across dimensions

**HELM Implementation:** Requires `LLMAsJuryAnnotator` configuration

**Pros:**
- Evaluates reasoning quality, not just labels
- Captures artistic understanding
- Flexible, nuanced assessment

**Cons:**
- Expensive (extra API calls)
- Judge model may have biases
- Less reproducible than exact match

### Option 3: Multi-Aspect Classification

**Task:** Predict multiple attributes per image

**Targets:**
1. Dominant emotion (12 classes)
2. Arousal level (2 classes: High/Low)
3. Valence (2 classes: Positive/Negative)
4. Visual attributes (5 attributes with free-form descriptions)

**Evaluation:**
- Emotion accuracy (exact match)
- Arousal accuracy (exact match)
- Valence accuracy (exact match)
- Attributes quality (BLEU/ROUGE against ground truth)

**Pros:**
- More comprehensive evaluation
- Tests multiple aspects of understanding
- Aligns with original paper's multi-dimensional approach

**Cons:**
- More complex implementation
- Multiple metrics to track
- Requires more careful prompt engineering

---

## Recommendations

### For Current Use
**Keep exact_match classification** because:
- Simple, objective, reproducible
- Allows direct model comparison
- Tests genuine emotion understanding
- Validated ground truth (>85% human agreement)

### For Future Enhancement
**Consider LLM-as-judge variant** that:
1. Asks models to explain emotion choice
2. Evaluates artistic reasoning quality
3. Provides richer signal about creativity understanding

### For Research
**Implement multi-aspect evaluation** to:
- Align more closely with original paper's methodology
- Test arousal/valence prediction separately
- Evaluate visual attribute understanding
- Enable comparison with text-to-image models' attribute alignment scores

---

## Dataset Ground Truth Quality

**Annotation Process:**
1. GPT-4o generates initial annotations for all fields
2. Human annotators verify on 5,600 sample images
3. Agreement measured: >85% across all dimensions
4. High-quality consensus labels used as ground truth

**Confidence Level:** High - dual verification (AI + human)

**Implication:** Exact match evaluation is well-justified given annotation quality

---

## References

- Paper: https://arxiv.org/abs/2506.03652
- Project Page: https://zhiliangzhang.github.io/EmoArt-130k/
- GitHub: https://github.com/ZHILIANGZHANG/EmoArt-130k
- Dataset: https://huggingface.co/datasets/printblue/EmoArt-130k
- HELM Scenario: `scenarios/emoart_scenario.py`
