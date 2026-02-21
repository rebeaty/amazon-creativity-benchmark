# Metric Requirements: FSCG-8

Source: Paper "ProCreate, Don't Reproduce! Propulsive Energy Diffusion for Creative Generation" (ECCV 2024)
Dataset: Jacklu0831/few-shot-creative-generation-8

## ⚠️ CRITICAL: Text-to-Image Generation Task

**This benchmark evaluates image generation models (e.g., Stable Diffusion), NOT language models.**

FSCG-8 requires:
1. **Model capability**: Text-to-image generation (e.g., Stable Diffusion, DALL-E, Midjourney)
2. **Custom evaluation pipeline**: Image-based metrics (FID, CLIP) not available in standard HELM
3. **Infrastructure**: GPU support for diffusion models, image storage, batch processing

**Current HELM limitation**: HELM is primarily designed for LLM evaluation. Integrating FSCG-8 requires:
- Image generation model adapter (similar to how HELM adapts different LLM APIs)
- Custom metric implementations for image quality and diversity
- Storage and comparison infrastructure for generated images

## Evaluation Metrics

FSCG-8 evaluates text-to-image generation on two key dimensions:
1. **Sample Diversity** (Creativity): How diverse and novel are the generated images?
2. **Sample Fidelity** (Quality): How well do images match prompts and maintain category consistency?

### Automatic Metrics Used

The paper evaluates using the following metrics:

#### Diversity/Creativity Metrics:
- **Vendi Score**: Measures diversity in the generated image set
- **Mean Similarity Score (MSS)**: Average pairwise similarity (lower = more diverse)
- **SSCD Score**: Self-Supervised Copy Detection score (detects near-duplicates/memorization)

#### Fidelity/Quality Metrics:
- **FID (Fréchet Inception Distance)**: Measures distributional similarity to reference images
- **KID (Kernel Inception Distance)**: Alternative to FID, more robust to small sample sizes
- **Precision**: Measures what fraction of generated images are realistic
- **Recall**: Measures what fraction of the reference distribution is covered
- **Prompt Fidelity (CLIP)**: CLIP cosine similarity between generated images and text prompts

### Implementation Notes

1. **Image Generation Context**:
   - Models generate images from text prompts
   - For few-shot evaluation, models may be fine-tuned on a small subset (e.g., 10 images) before generation
   - The 50 examples per category can serve as both adaptation data and test prompts

2. **Metric Computation**:
   - Diversity metrics require comparing multiple generated images
   - Fidelity metrics compare generated images to reference images or compute prompt alignment
   - Some metrics (FID, KID) require pre-trained feature extractors (e.g., Inception-v3)

3. **HELM Integration Requirements**:
   - Need to implement custom metric class for FSCG-8 evaluation
   - Metrics should handle image-to-image comparison (generated vs reference)
   - Metrics should support text-to-image alignment (CLIP-based)
   - Consider batching all generations per category before computing diversity metrics

4. **Libraries**:
   - `torch-fidelity` for FID, KID, Precision, Recall
   - `vendi-score` library for Vendi score
   - `transformers` (CLIP model) for prompt fidelity
   - `scipy` for pairwise similarity computations

### Required Implementation Components

To fully integrate FSCG-8 into HELM, the following components must be implemented:

#### 1. Model Adapter for Image Generation
```python
class DiffusionModelAdapter:
    """
    Adapter for text-to-image diffusion models.
    Similar to how HELM has adapters for different LLM APIs.
    """
    def generate_images(self, prompt: str, num_samples: int = 1) -> List[Image]:
        """Generate images from text prompt"""
        pass
```

Supported models should include:
- Stable Diffusion (v1.5, SDXL, etc.)
- DALL-E 2/3 (via OpenAI API)
- Midjourney (via API if available)
- Other diffusion models

#### 2. Custom Metrics Implementation

Each metric requires a separate implementation:

**a) FID/KID Metrics** (using `torch-fidelity` or `pytorch-fid`)
```python
class FIDMetric:
    def compute(self, generated_images: List[Image], reference_images: List[Image]) -> float:
        # Extract features using Inception-v3
        # Compute Fréchet distance between distributions
        pass
```

**b) Diversity Metrics** (Vendi Score, MSS)
```python
class DiversityMetrics:
    def compute_vendi_score(self, images: List[Image]) -> float:
        # Measure diversity in generated set
        pass

    def compute_mss(self, images: List[Image]) -> float:
        # Mean pairwise similarity (lower = more diverse)
        pass
```

**c) CLIP-based Metrics**
```python
class CLIPMetrics:
    def compute_prompt_fidelity(self, images: List[Image], prompts: List[str]) -> float:
        # CLIP cosine similarity between images and prompts
        pass
```

**d) SSCD Metric**
```python
class SSCDMetric:
    def compute_copy_detection(self, generated_images: List[Image], training_images: List[Image]) -> float:
        # Detect near-duplicates/memorization
        pass
```

#### 3. Evaluation Pipeline

The evaluation flow differs from standard LLM evaluation:

```
1. Load FSCG-8 prompts and reference images
2. For each category:
   a) Generate N images per prompt (typically N=4-8)
   b) Save generated images
3. Batch compute metrics:
   a) FID/KID: Compare generated vs reference distribution
   b) Diversity: Analyze within-category variation
   c) CLIP: Measure text-image alignment
   d) SSCD: Check for memorization
4. Aggregate results per category and overall
```

### Metric Configuration for RunSpec

```python
# Example RunSpec configuration (to be implemented separately)
def get_fscg8_metric_specs():
    return [
        MetricSpec(
            class_name="ImageGenerationMetrics",
            args={
                "metrics": ["fid", "kid", "precision", "recall"],
                "num_samples_per_prompt": 4,
                "inception_model": "inception_v3",
            }
        ),
        MetricSpec(
            class_name="DiversityMetrics",
            args={
                "metrics": ["vendi", "mss"],
                "embedding_model": "clip-vit-large",
            }
        ),
        MetricSpec(
            class_name="CLIPMetrics",
            args={
                "model": "openai/clip-vit-large-patch14",
                "compute_prompt_fidelity": True,
            }
        ),
        MetricSpec(
            class_name="SSCDMetric",
            args={
                "threshold": 0.85,  # For copy detection
            }
        ),
    ]
```

### Reference Implementation

The original evaluation code can be found in:
- GitHub: https://github.com/agentic-learning-ai-lab/procreate-diffusion
- Evaluation scripts likely in `src/eval/` or similar directory

### Human Evaluation (Optional)

The paper may also include human evaluation for:
- Creativity/Novelty assessment
- Style consistency
- Prompt alignment

However, the automatic metrics above are sufficient for HELM benchmarking.

## Implementation Status and Next Steps

### Current Status

✅ **Completed:**
- FSCG8Scenario class loads dataset and formats prompts
- Reference images are available for comparison
- Dataset is accessible via HuggingFace

❌ **NOT Implemented (Required for Full Integration):**

1. **Model Adapter**: HELM needs a text-to-image model adapter
   - Currently HELM only supports LLMs generating text
   - Need: DiffusionModelAdapter or similar for Stable Diffusion, DALL-E, etc.

2. **Custom Metrics**: All image-based metrics need implementation
   - FID, KID (require Inception-v3 feature extraction)
   - Vendi Score, MSS (require similarity computation over image sets)
   - CLIP metrics (require CLIP model for text-image alignment)
   - SSCD (requires copy detection model)

3. **Evaluation Pipeline**: Image generation and comparison workflow
   - Generate multiple images per prompt
   - Store and manage generated images
   - Batch compute metrics across image sets
   - Handle GPU requirements for diffusion models

### Why This Is Different from Standard HELM Tasks

| Aspect | Standard HELM (LLMs) | FSCG-8 (Image Gen) |
|--------|---------------------|-------------------|
| Model Type | Language models | Diffusion models |
| Input | Text prompt | Text prompt |
| Output | Text completion | Generated image |
| Evaluation | Text-based (BLEU, ROUGE, exact match) | Image-based (FID, CLIP, diversity) |
| Infrastructure | CPU/GPU for inference | GPU for diffusion, storage for images |
| Comparison | String matching, n-gram overlap | Feature extraction, distribution distance |

### Recommended Implementation Path

1. **Prototype with Standalone Script** (Recommended first step)
   - Use the FSCG8Scenario to load prompts
   - Run Stable Diffusion externally to generate images
   - Compute metrics using `torch-fidelity` and `clip` libraries
   - Validate results against paper's reported metrics

2. **Integrate into HELM** (After validation)
   - Implement DiffusionModelAdapter
   - Create ImageGenerationMetrics classes
   - Add RunSpec support for image generation tasks
   - Test end-to-end pipeline

3. **Test with Example Category**
   - Start with one category (e.g., "pokemon")
   - Generate 4-8 images per prompt
   - Compute FID and CLIP scores
   - Compare to paper's reported results

### Alternative: Multimodal LLM Evaluation

If evaluating **multimodal LLMs** (e.g., GPT-4 Vision, Claude 3) on FSCG-8 images:
- Task becomes: "Describe this creative image" or "Rate the creativity of this image"
- Evaluation: Standard text metrics or LLM-as-judge
- This IS supported in current HELM with vision-language models
- But this is NOT the intended use of FSCG-8 (which is for image generation)
