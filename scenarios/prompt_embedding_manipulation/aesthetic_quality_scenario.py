"""
HEIM Scenario: Aesthetic Quality Evaluation for Text-to-Image Models

Paper: Manipulating Embeddings of Stable Diffusion Prompts
       Niklas Deckers, Julia Peters, Martin Potthast
       IJCAI 2024
       https://arxiv.org/abs/2308.12059

GitHub: https://github.com/webis-de/ijcai24-manipulating-embeddings-stable-diffusion
Zenodo: https://doi.org/10.5281/zenodo.8274625

Task: Text-to-image generation with aesthetic quality evaluation

This scenario evaluates text-to-image models on their ability to generate
aesthetically pleasing and high-quality images from diverse prompts. While the
original paper focused on embedding manipulation techniques, the curated prompts
and aesthetic evaluation metrics provide a solid foundation for benchmarking
general text-to-image generation quality.

Dataset details:
  - Primary test set: 149 prompts from DiffusionDB
  - Additional set: 13 LAION-Aesthetics-V2 prompts
  - Total: 162 prompts across diverse creative domains
  - Prompts cover: fantasy art, realistic scenes, abstract concepts, specific
    styles, character design, landscapes, and creative compositions

Prompt sources:
  - DiffusionDB: Large-scale dataset of prompts from Stable Diffusion users
    - Subsets: large_random_100k, large_random_1k
    - Filtered for creative diversity
  - LAION-Aesthetics-V2: High aesthetic quality prompts

Evaluation aspects (HEIM framework):
  1. Aesthetics: Visual appeal, composition quality, artistic merit
  2. Image Quality: Technical quality, coherence, detail
  3. Image-Text Alignment: How well image matches prompt
  4. Creativity: Originality, artistic interpretation
  5. Reasoning: Ability to handle complex multi-object prompts

Metrics:
  - Aesthetic Score: LAION Aesthetic Predictor (primary metric)
  - Image Quality: Sharpness, blurriness (inverse), coherence
  - Alignment: CLIP score (image-text similarity)
  - Originality: Diversity across generated images
  - Technical: FID, Inception Score (optional)

Original paper evaluation:
  - User study with 8 participants
  - Qualitative preference evaluation
  - Metric-based optimization demonstrations
  - Compared embedding manipulation vs prompt engineering

This HEIM scenario adaptation:
  - Uses prompts as standard text-to-image test set
  - Applies aesthetic and quality metrics
  - Evaluates ANY text-to-image model (not just Stable Diffusion)
  - Focuses on creative and aesthetic output quality

Prompt characteristics:
  - Highly detailed and specific (e.g., "8k, trending on artstation")
  - Mix of artistic styles (oil painting, digital art, concept art)
  - Complex compositions with multiple elements
  - Creative and imaginative scenarios
  - Tests model's ability to handle detailed instructions

Fields used:
  - prompt: Text prompt for image generation
  - source: "diffusiondb" or "laion_aesthetic"
  - prompt_id: Unique identifier (line number)

Evaluation type: Image generation quality assessment
Primary metric: Aesthetic score
Secondary metrics: Image quality, CLIP score, visual appeal

Note: While the original paper demonstrated embedding manipulation techniques,
this scenario uses their curated prompts as a standalone benchmark for evaluating
aesthetic quality in text-to-image generation, applicable to any model.

For detailed evaluation methodology and metrics, see:
scenarios/prompt_embedding_manipulation/evaluation_notes.md
"""

import os
from typing import List, Optional
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
)


class AestheticQualityScenario(Scenario):
    """
    Aesthetic Quality Evaluation for text-to-image models.

    Evaluates text-to-image models on their ability to generate aesthetically
    pleasing, high-quality images from diverse creative prompts.
    """

    name = "aesthetic_quality"
    description = "Text-to-image aesthetic quality evaluation using DiffusionDB and LAION prompts"
    tags = ["creativity", "text_to_image", "aesthetic", "image_quality", "art_generation"]

    def __init__(
        self,
        use_diffusiondb: bool = True,
        use_laion: bool = True,
        subset: Optional[str] = None
    ):
        """
        Args:
            use_diffusiondb: Include 149 DiffusionDB prompts (default: True)
            use_laion: Include 13 LAION-Aesthetics-V2 prompts (default: True)
            subset: Optional subset selection:
                   - "diffusiondb_only": Only DiffusionDB prompts (149)
                   - "laion_only": Only LAION prompts (13)
                   - None: All prompts (162)
        """
        super().__init__()
        self.use_diffusiondb = use_diffusiondb
        self.use_laion = use_laion

        # Handle subset parameter
        if subset == "diffusiondb_only":
            self.use_diffusiondb = True
            self.use_laion = False
        elif subset == "laion_only":
            self.use_diffusiondb = False
            self.use_laion = True

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load prompt datasets and create instances for text-to-image generation.

        Args:
            output_path: Directory containing the prompt dataset files

        Returns:
            List of instances with text prompts for image generation
        """
        instances = []

        # Get the scenario directory (where prompt files are located)
        scenario_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__))
        )

        # Load DiffusionDB prompts
        if self.use_diffusiondb:
            diffusiondb_path = os.path.join(scenario_dir, "diffusiondb_prompts.txt")
            if os.path.exists(diffusiondb_path):
                with open(diffusiondb_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        prompt = line.strip()
                        if prompt:  # Skip empty lines
                            instances.append(
                                Instance(
                                    input=Input(text=prompt),
                                    references=[],  # No reference images for generation
                                    split=TEST_SPLIT,
                                    id=f"diffusiondb_{idx:03d}",
                                    sub_split="diffusiondb"
                                )
                            )

        # Load LAION-Aesthetics-V2 prompts
        if self.use_laion:
            laion_path = os.path.join(scenario_dir, "laion_aesthetic_prompts.txt")
            if os.path.exists(laion_path):
                with open(laion_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        prompt = line.strip()
                        if prompt:  # Skip empty lines
                            instances.append(
                                Instance(
                                    input=Input(text=prompt),
                                    references=[],  # No reference images for generation
                                    split=TEST_SPLIT,
                                    id=f"laion_aesthetic_{idx:02d}",
                                    sub_split="laion_aesthetic"
                                )
                            )

        return instances


# Additional scenario variants for specific use cases

class DiffusionDBOnlyScenario(AestheticQualityScenario):
    """Variant using only DiffusionDB prompts (149 prompts)."""

    name = "aesthetic_quality_diffusiondb"
    description = "Aesthetic quality evaluation using DiffusionDB prompts only"

    def __init__(self):
        super().__init__(subset="diffusiondb_only")


class LAIONAestheticOnlyScenario(AestheticQualityScenario):
    """Variant using only LAION-Aesthetics-V2 prompts (13 prompts)."""

    name = "aesthetic_quality_laion"
    description = "Aesthetic quality evaluation using LAION-Aesthetics-V2 prompts only"

    def __init__(self):
        super().__init__(subset="laion_only")
