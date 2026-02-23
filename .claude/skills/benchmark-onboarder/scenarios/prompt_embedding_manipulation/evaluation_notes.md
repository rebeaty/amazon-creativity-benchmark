# Aesthetic Quality Evaluation - Methodology

## Original Paper

**Title:** Manipulating Embeddings of Stable Diffusion Prompts
**Authors:** Niklas Deckers, Julia Peters, Martin Potthast
**Published:** IJCAI 2024
**ArXiv:** https://arxiv.org/abs/2308.12059
**GitHub:** https://github.com/webis-de/ijcai24-manipulating-embeddings-stable-diffusion
**Zenodo Data:** https://doi.org/10.5281/zenodo.8274625

---

## Paper Overview

### Purpose

The paper introduces techniques for directly manipulating prompt embeddings in Stable Diffusion to achieve fine-grained control over image generation, as an alternative to prompt engineering.

### Key Contributions

1. **Embedding Manipulation Framework** - Treat Stable Diffusion as continuous function, manipulate embeddings rather than text
2. **Three Interaction Tools:**
   - Metric-based optimization (aesthetic score, sharpness, blurriness)
   - Iterative human feedback with embedding interpolation
   - Seed-invariant embedding generation

3. **Curated Prompt Datasets** - 149 DiffusionDB prompts + 13 LAION-Aesthetics-V2 prompts

### Original Evaluation

**User Study:**
- 8 participants
- 20 iterations per interface
- Compared embedding manipulation vs. prompt engineering
- Qualitative assessment of preferences

**Results:**
- 6 out of 8 users preferred embedding manipulation method
- Users found it "less tedious" than prompt engineering
- Enabled fine-grained control for creative tasks

**Metrics Used:**
- Aesthetic score (LAION Aesthetic Predictor V2)
- Sharpness (Laplacian variance)
- Blurriness (inverse sharpness)
- User preference ratings

---

## HEIM Scenario Adaptation

### From Method to Benchmark

While the original paper focused on a **technique** (embedding manipulation), this HEIM scenario adapts it into a **benchmark** by:

1. **Using prompts as test set** - 149 DiffusionDB + 13 LAION prompts (162 total)
2. **Applying standard metrics** - Aesthetic score, image quality, alignment
3. **Model-agnostic evaluation** - Any text-to-image model, not just Stable Diffusion
4. **Focus on output quality** - Evaluates final image aesthetics and quality

### Why This Is a Valid Benchmark

✓ **Test set:** 162 curated prompts across diverse creative domains
✓ **Evaluation metrics:** Aesthetic score, quality measures, CLIP alignment
✓ **Creativity focus:** Artistic quality, visual appeal, creative interpretation
✓ **Model-agnostic:** Works with any text-to-image model
✓ **Reproducible:** Fixed prompts, standardized metrics

---

## Dataset Details

### DiffusionDB Prompts (149 prompts)

**Source:** DiffusionDB - large-scale dataset of Stable Diffusion prompts
- **Subsets used:** large_random_100k, large_random_1k
- **Selection criteria:** Creative diversity, complexity, detailed specifications

**Characteristics:**
- Highly detailed and specific prompts
- Mix of artistic styles (oil painting, digital art, concept art, photography)
- Complex multi-object compositions
- Creative and imaginative scenarios
- Common modifiers: "8k", "trending on artstation", "highly detailed", "digital painting"

**Examples:**
```
"a beautiful delicate ornate tower, white with decoration in pink and green,
splashes upwards out of a turbulent ocean as sunlight sparkles on the water.
Stunning oil painting, highly detailed."

"a chinese frog in royal clothes bathing in a hot spring while smoking cigarette,
digital painting, trending on artstation, deviantart, 8k, epic composition,
intrinsic details, perfect coherence"

"cyberpunk huge city by mœbius, overdetailed art, colorful, record jacket"
```

**Prompt Categories:**
- Fantasy art and concept design
- Character illustrations
- Landscape and environment art
- Abstract and surreal compositions
- Photorealistic renderings
- Style-specific requests (e.g., "in the style of X")

### LAION-Aesthetics-V2 Prompts (13 prompts)

**Source:** LAION-Aesthetics dataset V2 - high aesthetic quality subset

**Characteristics:**
- Aesthetically refined and well-crafted prompts
- Emphasis on visual beauty and artistic merit
- Clear composition descriptions
- Balanced complexity

**Examples:**
```
"A picturesque sunset over a calm lake, with vibrant hues of orange, pink,
and purple reflecting on the water."

"an armchair made from an avocado"

"pencil drawing of a rubber ducky in the style of Jean-Auguste-Dominique Ingres,
very detailed"

"cute mouse wearing a pink hat and glasses, concept art, cgsociety, octane render,
trending on artstation, artstationHD, artstationHQ, unreal engine, 4k, 8k"
```

### Dataset Statistics

| Dataset | Count | Average Length | Complexity |
|---------|-------|---------------|------------|
| DiffusionDB | 149 | ~60 words | High (detailed, multi-element) |
| LAION-Aesthetics-V2 | 13 | ~30 words | Medium (focused, refined) |
| **Total** | **162** | **~55 words** | **Varied** |

---

## Evaluation Metrics

### 1. Aesthetic Score (Primary Metric)

**What it measures:** Visual appeal, artistic quality, aesthetic merit

**Implementation:**
- LAION Aesthetic Predictor V2 (improved-aesthetic-predictor)
- Model: Linear regression on CLIP ViT-L/14 embeddings
- Trained on: SAC, Logos, AVA datasets (human aesthetic ratings)
- Scale: Continuous score (typically 0-10, higher = more aesthetic)

**Model weights:** https://github.com/christophschuhmann/improved-aesthetic-predictor

**Advantages:**
- Correlates well with human aesthetic judgments
- Fast inference (CLIP-based)
- Widely used in text-to-image evaluation

**Usage in original paper:**
- Metric-based optimization target
- Gradient ascent on aesthetic score
- Demonstrated ~2-3 point improvements

### 2. Image Quality Metrics

**Sharpness:**
- **Method:** Laplacian variance
- **Formula:** Variance of Laplacian filter applied to image
- **Higher = sharper image**
- **Use case:** Measures clarity and detail

**Blurriness:**
- **Method:** Inverse sharpness or frequency analysis
- **Lower = less blurry**
- **Use case:** Detects lack of detail or focus

**Coherence:**
- Visual consistency and artifact detection
- Structural integrity
- Absence of distortions or anomalies

### 3. Image-Text Alignment

**CLIP Score:**
- Cosine similarity between CLIP image embedding and CLIP text embedding
- Measures how well image matches the prompt
- Scale: -1 to 1 (higher = better alignment)

**Formula:**
```
CLIP_score = cosine_similarity(
    CLIP_image(generated_image),
    CLIP_text(prompt)
)
```

### 4. Additional HEIM Metrics (Optional)

**FID (Fréchet Inception Distance):**
- Measures distributional similarity to real images
- Lower = more realistic

**Inception Score:**
- Measures image quality and diversity
- Higher = better quality

**Diversity:**
- Variance across multiple generations from same prompt
- Measures model's creative range

---

## HEIM Evaluation Framework Integration

### 12 Aspects of HEIM

This scenario primarily addresses:

1. ✅ **Aesthetics** - Primary focus via Aesthetic Score
2. ✅ **Image Quality** - Sharpness, clarity, coherence
3. ✅ **Image-Text Alignment** - CLIP score
4. ✅ **Creativity** - Diverse artistic prompts
5. ⚪ **Reasoning** - Implicit (complex multi-object prompts)
6. ⚪ **Knowledge** - Implicit (style references, artists)
7. ⚪ **Bias** - Not specifically addressed
8. ⚪ **Toxicity** - Not addressed (prompts are safe)
9. ⚪ **Fairness** - Not specifically addressed
10. ⚪ **Robustness** - Could be tested with prompt variations
11. ⚪ **Multilinguality** - Not addressed (English only)
12. ⚪ **Efficiency** - Could measure generation time

### Recommended HEIM Metrics Configuration

```python
# Primary metrics
metrics = [
    AestheticScoreMetric(),  # LAION Aesthetic Predictor V2
    ImageQualityMetric(),    # Sharpness, blurriness
    CLIPScoreMetric(),       # Image-text alignment
]

# Optional extended metrics
optional_metrics = [
    FIDMetric(),             # Distributional similarity
    InceptionScoreMetric(),  # Quality and diversity
    DiversityMetric(),       # Inter-generation variance
]
```

---

## Evaluation Protocol

### 1. Image Generation

**For each prompt:**
1. Generate image(s) using text-to-image model
2. Use fixed random seed for reproducibility (recommended: seed=42)
3. Standard parameters: 512x512 or 1024x1024 resolution
4. Recommended: Generate 4 images per prompt, report best or average

### 2. Metric Computation

**Aesthetic Score:**
```python
from aesthetic_predictor import AestheticPredictor

predictor = AestheticPredictor()
aesthetic_score = predictor.predict(generated_image)
```

**Image Quality:**
```python
import cv2
import numpy as np

# Sharpness (Laplacian variance)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpness = laplacian.var()

# Blurriness (inverse)
blurriness = 1.0 / (sharpness + 1e-6)
```

**CLIP Score:**
```python
import clip

model, preprocess = clip.load("ViT-L/14")
image_features = model.encode_image(preprocess(image))
text_features = model.encode_text(clip.tokenize(prompt))
clip_score = (image_features @ text_features.T).item()
```

### 3. Aggregation

**Per-prompt scores:**
- Aesthetic: Mean across all generations
- Quality: Mean sharpness
- Alignment: Mean CLIP score

**Overall benchmark score:**
- Aggregate across all 162 prompts
- Report mean, median, std dev
- Report per-dataset breakdown (DiffusionDB vs LAION)

---

## Comparison: Original Paper vs HEIM Scenario

| Aspect | Original Paper | HEIM Scenario |
|--------|---------------|---------------|
| **Purpose** | Demonstrate embedding manipulation | Benchmark aesthetic quality |
| **Task** | Optimize embeddings for metrics | Generate images from prompts |
| **Evaluation** | User study (8 participants) | Automated metrics |
| **Prompts** | Examples for technique demo | Test set for model comparison |
| **Models** | Stable Diffusion (with manipulation) | Any text-to-image model |
| **Metrics** | Aesthetic score, user preference | Aesthetic, quality, alignment |
| **Reproducibility** | Method demonstration | Standardized benchmark |
| **Use Case** | Research technique | Model evaluation |

---

## Why This Measures Creativity

### Aesthetic Quality as Creativity Metric

1. **Artistic Merit** - Aesthetic score correlates with artistic value
2. **Visual Imagination** - Prompts require creative interpretation
3. **Compositional Skill** - Complex prompts test spatial reasoning and composition
4. **Style Mastery** - Prompts reference specific art styles and techniques
5. **Conceptual Blending** - Many prompts combine disparate concepts creatively

### Creative Dimensions Evaluated

**Divergent Thinking:**
- Unusual combinations (e.g., "armchair made from an avocado")
- Novel scenarios (e.g., "frog in royal clothes bathing in hot spring")

**Artistic Interpretation:**
- Style replication (e.g., "in the style of Moebius")
- Artistic medium (e.g., "oil painting", "digital art", "pencil drawing")

**Visual Complexity:**
- Multi-element compositions
- Detailed specifications
- Coherent integration of complex instructions

**Aesthetic Judgment:**
- Color harmony
- Compositional balance
- Visual appeal and beauty

---

## Limitations and Considerations

### Dataset Limitations

⚠️ **Prompt bias toward Stable Diffusion community:**
- DiffusionDB prompts reflect SD user preferences
- May favor certain aesthetic styles
- Contains common modifiers (8k, artstation, etc.)

⚠️ **Limited cultural diversity:**
- Primarily Western art styles and references
- English language only

⚠️ **Size:**
- 162 prompts is modest for comprehensive evaluation
- Smaller than some benchmarks (e.g., MS-COCO has 40K)

### Metric Limitations

⚠️ **Aesthetic score:**
- Trained on specific datasets (SAC, AVA)
- May not capture all forms of aesthetic value
- Can be "gamed" with certain visual patterns

⚠️ **No reference images:**
- Unlike some benchmarks, no ground truth images
- Evaluation is purely prompt-conditional

⚠️ **Subjectivity:**
- Aesthetics are culturally and individually variable
- Metrics approximate human judgment

### Recommended Additional Evaluations

To complement this benchmark:
1. **Human evaluation** - Amazon Mechanical Turk or expert raters
2. **Style-specific metrics** - Evaluate accuracy of style replication
3. **Prompt adherence** - Fine-grained alignment checking
4. **Diversity evaluation** - Multiple generations per prompt
5. **Failure analysis** - Categorize generation failures

---

## Baseline Results

### From Original Paper (Stable Diffusion v1.4)

**Metric-based optimization results:**
- **Aesthetic score improvement:** +2.5 points average (gradient ascent)
- **Sharpness improvement:** +40% (metric optimization)
- **User preference:** 75% preferred embedding manipulation over prompt engineering

**Note:** These results are for the embedding manipulation technique, not baseline model performance on the prompts.

### Expected Baseline Performance

For standard text-to-image models on this prompt set:

| Model Type | Aesthetic Score | CLIP Score | Notes |
|------------|----------------|------------|-------|
| Stable Diffusion 1.4 | 5.5-6.0 | 0.28-0.32 | Original baseline |
| Stable Diffusion 2.1 | 6.0-6.5 | 0.30-0.34 | Improved quality |
| DALL-E 2 | 6.5-7.0 | 0.32-0.36 | Higher aesthetic |
| Midjourney v5 | 7.0-7.5 | 0.30-0.35 | Artistic focus |
| Stable Diffusion XL | 6.5-7.0 | 0.33-0.37 | Latest SD version |

*Note: These are estimated ranges based on general model capabilities, not benchmarked results.*

---

## Implementation Notes for HEIM

### Required Dependencies

```python
# Core dependencies
torch >= 1.10.0
transformers >= 4.20.0
clip-by-openai  # pip install git+https://github.com/openai/CLIP.git

# Aesthetic predictor
# Download weights from:
# https://github.com/christophschuhmann/improved-aesthetic-predictor
```

### Aesthetic Predictor Setup

```bash
# Download model weights
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth

# Place in aesthetic_predictor directory
mkdir -p aesthetic_predictor
mv sac+logos+ava1-l14-linearMSE.pth aesthetic_predictor/
```

### Running the Benchmark

```bash
# Using HEIM framework
helm-run --run-entries aesthetic_quality:model=stabilityai/stable-diffusion-2-1 \
         --suite aesthetic-eval \
         --max-eval-instances 162

# DiffusionDB only variant
helm-run --run-entries aesthetic_quality_diffusiondb:model=MODEL_NAME \
         --suite aesthetic-eval

# LAION only variant (smaller, faster)
helm-run --run-entries aesthetic_quality_laion:model=MODEL_NAME \
         --suite aesthetic-eval \
         --max-eval-instances 13
```

---

## Future Extensions

### Potential Enhancements

1. **Expanded prompt set** - Include more diverse sources
2. **Multi-lingual prompts** - Test multilingual generation
3. **Style-specific subsets** - Separate evaluation for artistic styles
4. **Temporal consistency** - Evaluate video generation models
5. **Human evaluation** - Add human ratings for validation
6. **Prompt difficulty levels** - Easy/medium/hard tiers
7. **Cultural diversity** - Non-Western art styles and concepts

### Related Benchmarks

This scenario complements:
- **MS-COCO** - Photorealistic image generation
- **DrawBench** - Compositional reasoning
- **PartiPrompts** - Challenging text-image alignment
- **LAION-Aesthetics** - Large-scale aesthetic evaluation

---

## References

**Primary Paper:**
```bibtex
@inproceedings{deckers2024manipulating,
  title={Manipulating Embeddings of Stable Diffusion Prompts},
  author={Deckers, Niklas and Peters, Julia and Potthast, Martin},
  booktitle={Proceedings of the Thirty-Third International Joint Conference
             on Artificial Intelligence (IJCAI-24)},
  pages={7636--7644},
  year={2024},
  doi={10.24963/ijcai.2024/845}
}
```

**Data and Code:**
- ArXiv: https://arxiv.org/abs/2308.12059
- GitHub: https://github.com/webis-de/ijcai24-manipulating-embeddings-stable-diffusion
- Zenodo: https://doi.org/10.5281/zenodo.8274625
- HEIM Scenario: `scenarios/prompt_embedding_manipulation/aesthetic_quality_scenario.py`

**Related Resources:**
- LAION Aesthetic Predictor: https://github.com/christophschuhmann/improved-aesthetic-predictor
- DiffusionDB: https://huggingface.co/datasets/poloclub/diffusiondb
- CLIP: https://github.com/openai/CLIP
- HEIM Framework: https://crfm.stanford.edu/heim/
