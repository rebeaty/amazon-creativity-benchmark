"""
Example: FSCG-8 Evaluation with Diffusion Models

This script demonstrates how to evaluate a text-to-image model on FSCG-8
using the scenario for data loading and external tools for metrics.

NOTE: This is a STANDALONE example, not integrated into HELM's evaluation pipeline.
      It shows the components needed for full HELM integration.

Requirements:
    pip install diffusers transformers torch torchvision
    pip install torch-fidelity clip-score vendi-score
"""

from fscg8_scenario import FSCG8Scenario
import tempfile
import os
from typing import List
from PIL import Image


def example_generation_pipeline():
    """
    Example showing how to generate images using Stable Diffusion
    for FSCG-8 prompts.
    """

    # 1. Load FSCG-8 prompts using the scenario
    print("Loading FSCG-8 dataset...")
    scenario = FSCG8Scenario(category="pokemon")  # Start with one category

    with tempfile.TemporaryDirectory() as output_path:
        instances = scenario.get_instances(output_path)
        print(f"Loaded {len(instances)} prompts from Pokemon category")

        # 2. Initialize diffusion model (example - not executed here)
        print("\nExample: Initialize Stable Diffusion")
        print("""
        from diffusers import StableDiffusionPipeline
        import torch

        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        """)

        # 3. Generate images for each prompt
        print("\nExample: Generate images")
        generated_images_by_prompt = {}

        for instance in instances[:3]:  # Show first 3
            prompt = instance.input.text
            print(f"\nPrompt: {prompt}")
            print(f"  Example generation code:")
            print(f"""
            # Generate 4 images per prompt for diversity measurement
            images = pipe(
                prompt="{prompt}",
                num_images_per_prompt=4,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images
            """)

            # In practice:
            # generated_images_by_prompt[prompt] = images

        # 4. Extract reference images from instances
        print("\n" + "="*60)
        print("Reference images available at:")
        for instance in instances[:3]:
            ref = instance.references[0]
            if ref.output.multimedia_content:
                img_location = ref.output.multimedia_content.media_objects[0].location
                print(f"  {img_location}")


def example_metric_computation():
    """
    Example showing how to compute FSCG-8 metrics on generated images.
    """
    print("\n" + "="*60)
    print("METRIC COMPUTATION EXAMPLES")
    print("="*60)

    # FID (Fréchet Inception Distance)
    print("\n1. FID Metric:")
    print("""
    from pytorch_fid import fid_score

    # Compare generated images to reference images
    fid_value = fid_score.calculate_fid_given_paths(
        paths=[generated_images_dir, reference_images_dir],
        batch_size=50,
        device='cuda',
        dims=2048
    )
    print(f"FID: {fid_value:.2f}")  # Lower is better
    """)

    # Vendi Score (Diversity)
    print("\n2. Vendi Score (Diversity):")
    print("""
    from vendi_score import vendi
    from torchvision.models import inception_v3
    import torch

    # Extract features from generated images
    model = inception_v3(pretrained=True)
    features = extract_features(generated_images, model)

    # Compute diversity
    vendi_score = vendi.score_K(features)
    print(f"Vendi Score: {vendi_score:.2f}")  # Higher is better
    """)

    # CLIP Similarity (Prompt Fidelity)
    print("\n3. CLIP Similarity (Prompt Fidelity):")
    print("""
    from transformers import CLIPProcessor, CLIPModel
    import torch

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Compute text-image similarity
    inputs = processor(
        text=[prompt],
        images=generated_images,
        return_tensors="pt",
        padding=True
    )

    outputs = model(**inputs)
    similarity = outputs.logits_per_image.mean().item()
    print(f"CLIP Similarity: {similarity:.3f}")  # Higher is better
    """)

    # Mean Similarity Score (MSS)
    print("\n4. Mean Similarity Score (Diversity):")
    print("""
    from scipy.spatial.distance import pdist

    # Compute pairwise similarities between images
    pairwise_sims = pdist(image_embeddings, metric='cosine')
    mss = pairwise_sims.mean()
    print(f"MSS: {mss:.3f}")  # Lower = more diverse
    """)

    # SSCD (Copy Detection)
    print("\n5. SSCD (Copy Detection):")
    print("""
    # Detect if generated images are near-copies of training data
    from some_sscd_library import SSCDModel

    model = SSCDModel()
    copy_score = model.detect_copies(generated_images, training_images)
    print(f"SSCD Score: {copy_score:.3f}")  # Lower = less copying
    """)


def example_full_evaluation():
    """
    Example of complete evaluation flow for one category.
    """
    print("\n" + "="*60)
    print("FULL EVALUATION FLOW")
    print("="*60)

    print("""
    Step 1: Load dataset
        scenario = FSCG8Scenario(category="pokemon")
        instances = scenario.get_instances(output_path)

    Step 2: Generate images (4 per prompt)
        for instance in instances:
            prompt = instance.input.text
            images = diffusion_model.generate(prompt, num_samples=4)
            save_images(images, f"generated/pokemon/{instance.id}/")

    Step 3: Compute metrics
        results = {
            "fid": compute_fid(generated_dir, reference_dir),
            "kid": compute_kid(generated_dir, reference_dir),
            "vendi": compute_vendi_score(generated_images),
            "clip": compute_clip_similarity(generated_images, prompts),
            "mss": compute_mean_similarity(generated_images),
        }

    Step 4: Compare to paper's baselines
        Paper reports (Table 1, Stable Diffusion v1.5):
            - Pokemon: FID ~15-25 (depends on fine-tuning method)
            - Vendi: ~0.6-0.8 (higher = more diverse)
            - CLIP: ~0.25-0.30 (prompt alignment)

    Step 5: Aggregate across all 8 categories
        for category in FSCG8Scenario.CATEGORIES:
            results[category] = evaluate_category(category)
        overall_fid = mean([r['fid'] for r in results.values()])
    """)


if __name__ == "__main__":
    print("FSCG-8 Evaluation Example")
    print("="*60)
    print("This script shows how to evaluate text-to-image models on FSCG-8")
    print("It demonstrates the components needed for full HELM integration\n")

    example_generation_pipeline()
    example_metric_computation()
    example_full_evaluation()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
To integrate FSCG-8 into HELM, implement:

1. DiffusionModelAdapter class
   - generate_images(prompt, num_samples) method
   - Support for Stable Diffusion, DALL-E, etc.

2. ImageGenerationMetrics classes
   - FIDMetric, KIDMetric (using torch-fidelity)
   - DiversityMetrics (Vendi score, MSS)
   - CLIPMetrics (prompt fidelity)
   - SSCDMetric (copy detection)

3. Modified evaluation pipeline
   - Batch image generation
   - Save generated images to disk
   - Compute metrics across image sets
   - Aggregate results per category

See scenarios/fscg8/metric_notes.md for detailed requirements.
    """)
