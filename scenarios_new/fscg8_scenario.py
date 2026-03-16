"""
HELM Scenario: FSCG-8 (Few-Shot Creative Generation 8)

Paper: ProCreate, Don't Reproduce! Propulsive Energy Diffusion for Creative Generation
       https://arxiv.org/abs/2408.02226
       ECCV 2024

Dataset: Jacklu0831/few-shot-creative-generation-8 (HuggingFace)
Code: https://github.com/agentic-learning-ai-lab/procreate-diffusion

⚠️ IMPORTANT: This benchmark evaluates IMAGE GENERATION models (Stable Diffusion, DALL-E), NOT LLMs.
Full integration requires:
  1. Diffusion model adapter (text-to-image generation capability)
  2. Custom metrics: FID, KID, Vendi Score, CLIP similarity, SSCD
  3. Image storage and batch processing infrastructure

See scenarios/fscg8/metric_notes.md for complete implementation requirements.

Task:
  Text-to-image generation benchmark evaluating creative generation across 8 diverse categories.
  Models receive text prompts and generate images that should be creative (diverse, novel) while
  maintaining fidelity to the category style/concept.

Categories (8):
  1. pokemon - Creative Pokemon character designs
  2. one_piece - One Piece anime character styles
  3. amedeo_modigliani - Paintings in Amedeo Modigliani's artistic style
  4. apple - Apple product designs
  5. frank_gehry - Architecture in Frank Gehry's style
  6. burberry - Burberry fashion items
  7. nouns - Nouns DAO character designs
  8. rococo - Rococo style objects and decorations

Prompt format:
  Direct text description (e.g., "a Rococo style chandelier", "an Apple laptop charger")

Fields used: text (prompt), image (reference for evaluation)
Fields skipped: None

Evaluation:
  This is a generative task where models produce images from text prompts.
  Evaluation uses automatic metrics: FID, KID, Precision, Recall, Mean Similarity Score (MSS),
  Vendi score, Prompt Fidelity (CLIP), and SSCD score.
  These metrics assess both sample diversity (creativity) and fidelity (quality/relevance).
  See scenarios/fscg8/metric_notes.md for metric implementation details.

Dataset: 50 text-image pairs per category (400 total instances)
"""

from typing import List
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject


class FSCG8Scenario(Scenario):
    """
    FSCG-8: Few-Shot Creative Generation across 8 diverse categories.

    Text-to-image generation benchmark for evaluating creative generation capabilities.
    Each instance provides a text prompt describing an object/character/scene to generate.
    Reference images are included for metric computation.
    """

    name = "fscg8"
    description = "Jacklu0831/few-shot-creative-generation-8"
    tags = ["creativity", "multimodal", "text-to-image", "few-shot", "generation"]

    # The 8 categories in FSCG-8
    CATEGORIES = [
        "pokemon",
        "one_piece",
        "amedeo_modigliani",
        "apple",
        "frank_gehry",
        "burberry",
        "nouns",
        "rococo"
    ]

    def __init__(self, category: str = "all"):
        """
        Args:
            category: Which category to evaluate on. Options:
                     - "all": All 8 categories (400 instances)
                     - Individual category name: e.g., "pokemon", "rococo" (50 instances each)
        """
        super().__init__()
        if category != "all" and category not in self.CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be 'all' or one of: {', '.join(self.CATEGORIES)}"
            )
        self.category = category

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load FSCG-8 dataset and create instances for text-to-image generation.

        Each instance contains:
        - Input: Text prompt describing what to generate
        - Reference: Ground truth image (for metric computation)
        """
        instances = []

        # Determine which categories to load
        categories_to_load = self.CATEGORIES if self.category == "all" else [self.category]

        for cat in categories_to_load:
            # Load this category's split
            dataset = load_dataset(
                "Jacklu0831/few-shot-creative-generation-8",
                split=cat,
                # Note: Images will be loaded as PIL Image objects
            )

            for idx, item in enumerate(dataset):
                # Text prompt
                prompt_text = item["text"]

                # Create input with just the text prompt
                # (Some models may need additional formatting like "Generate an image of: {prompt}")
                input_obj = Input(text=prompt_text)

                # Create reference with the ground truth image
                # The image is stored as a PIL Image object from HuggingFace datasets
                # We need to save it to a file and reference it via MediaObject
                import os
                from PIL import Image

                # Create output directory for reference images if needed
                category_output_dir = os.path.join(output_path, "reference_images", cat)
                os.makedirs(category_output_dir, exist_ok=True)

                # Save the reference image
                image_filename = f"{cat}_{idx:03d}.png"
                image_path = os.path.join(category_output_dir, image_filename)

                # Save PIL Image to file
                if isinstance(item["image"], Image.Image):
                    item["image"].save(image_path)

                # Create reference with the image
                reference_image = MediaObject(
                    content_type="image/png",
                    location=image_path
                )

                # For generative tasks, the reference contains the ground truth image
                # Metrics will compare generated images against this reference
                references = [
                    Reference(
                        output=Output(multimedia_content=MultimediaObject([reference_image])),
                        tags=["reference_image"]
                    )
                ]

                # Create instance
                instance = Instance(
                    input=input_obj,
                    references=references,
                    split=TEST_SPLIT,
                    id=f"{cat}_{idx}"
                )

                instances.append(instance)

        return instances


# Alternative implementation pattern for models that need explicit "generate image" instructions:
"""
If models need more explicit instructions, modify the Input creation:

input_obj = Input(text=f"Generate an image of: {prompt_text}")

Or for multimodal models that need structured prompts:

multimedia_input = MultimediaObject([
    MediaObject(content_type="text/plain", text=f"Create an image based on this description: {prompt_text}")
])
input_obj = Input(multimedia_content=multimedia_input)
"""
