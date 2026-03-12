# Metric Requirements: Calligrapher Typography Benchmark

Source: arXiv:2506.24123, Section 4 (Experiments)

## Metrics

### 1. FID (Frechet Inception Distance)
- **What**: Overall image quality and distribution similarity between generated and real images
- **Scope**: Whole image
- **Lower is better**
- **Implementation**: Standard FID using Inception-v3 features

### 2. CLIP Style Similarity
- **What**: Style similarity within masked text regions
- **Model**: CLIP ViT-base
- **Scope**: Masked regions only (text areas defined by mask image)
- **Higher is better**
- **Computation**: Extract CLIP embeddings for masked regions of generated vs reference images, compute cosine similarity

### 3. DINO Style Similarity
- **What**: Additional style consistency measurement in masked regions
- **Model**: DINOv2
- **Scope**: Masked regions only
- **Higher is better**
- **Computation**: Extract DINO features for masked regions, compute cosine similarity

### 4. OCR Accuracy
- **What**: Text recognition correctness of generated text
- **Tool**: Google Cloud Vision text detection API (paper's choice); any OCR engine is acceptable
- **Higher is better**
- **Computation**: Run OCR on generated image, compare recognized text against target prompt text

## Paper Baselines (Self-Reference Task)

| Model         | FID (↓) | CLIP (↑) | DINO (↑) | OCR (↑) |
|---------------|---------|----------|----------|---------|
| TextDiffuser-2| 66.68   | —        | —        | 0.81    |
| AnyText       | 69.72   | —        | —        | 0.45    |
| FLUX-Fill     | 67.79   | —        | —        | 0.61    |
| Calligrapher  | 38.09   | 0.7401   | 0.9474   | 0.84    |

## User Study
- 30 participants, 1,000+ votes
- Three sub-domain scores on 1-4 scale
- Overall preference percentage
- Sub-domains not named in available documentation

## Notes
- All images are 512x512 resolution
- CLIP and DINO scores computed on masked regions only, not full image
- FID computed on full generated images
- Paper generates 2 outputs per test case with different random seeds
