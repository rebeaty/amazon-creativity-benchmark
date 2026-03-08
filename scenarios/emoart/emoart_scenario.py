"""
HELM Scenario: EmoArt - Emotion Classification in Artistic Images

Paper: EmoArt: A Multidimensional Dataset for Emotion-Aware Artistic Generation
       https://arxiv.org/abs/2506.03652
       Published: ACM Multimedia 2025

Dataset: HuggingFace - printblue/EmoArt-130k (full) or printblue/EmoArt-5k (preview)
Project: https://zhiliangzhang.github.io/EmoArt-130k/
GitHub: https://github.com/ZHILIANGZHANG/EmoArt-130k

Task: Given an artistic image, classify the dominant emotion it evokes.

Emotion categories (12 total):
  - High Arousal + Positive: Excited, Happy, Glad
  - High Arousal + Negative: Alarmed, Frustrated, Annoyed, Aroused
  - Low Arousal + Positive: Calm, Contentment
  - Low Arousal + Negative: Bored, Sad, Tired

Dataset details:
  - 132,664 artworks across 56 artistic styles
  - Annotations via GPT-4o with human verification (>85% agreement)
  - Each artwork includes: scene description, 5 visual attributes (brushwork,
    composition, color, line, light), arousal/valence labels, emotion category,
    and art therapy effects

Prompt format (standard MC format, not specified in paper):
  Question: What emotion does this artwork primarily evoke?

  [Image]

  A) Alarmed  B) Annoyed  C) Aroused  D) Bored
  E) Calm  F) Contentment  G) Excited  H) Frustrated
  I) Glad  J) Happy  K) Sad  L) Tired

  Answer:

Fields used:
  - image_path: Path to artwork image
  - description.third_section.dominant_emotion: Correct emotion label

Fields available but not used in this scenario:
  - description.first_section.description: Scene description
  - description.second_section.emotional_impact: Detailed emotional analysis
  - description.second_section.visual_attributes: Brushwork, color, composition, etc.
  - description.third_section.emotional_arousal_level: High/Low
  - description.third_section.emotional_valence: Positive/Negative
  - description.third_section.healing_effects: Art therapy applications

Image download instructions:
  Images are distributed in category-specific .tar.gz archives on HuggingFace.
  Download from: https://huggingface.co/datasets/printblue/EmoArt-130k/tree/main

  Categories: Abstract Art, Classics, Contemporary Fusion, Cubism, Impressionism,
              Modern Edge, Surrealism (and others, 56 total styles)

  Example:
    wget https://huggingface.co/datasets/printblue/EmoArt-130k/resolve/main/Cubism.tar.gz
    tar -xzvf Cubism.tar.gz

Evaluation: exact_match (classify emotion correctly)

Note: This scenario focuses on emotion recognition from art, which evaluates
models' understanding of visual creativity and affective content. While the
original paper evaluates text-to-image generation models, this scenario adapts
EmoArt for vision-language model evaluation.

For detailed evaluation methodology (both original paper and HELM adaptation),
see: scenarios/emoart/evaluation_notes.md
"""

import os
from typing import List
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject


class EmoArtScenario(Scenario):
    """
    EmoArt: Emotion classification from artistic images.

    Evaluates vision-language models on their ability to recognize and classify
    the dominant emotion evoked by artistic images across 12 emotion categories.
    """

    name = "emoart"
    description = "printblue/EmoArt-130k"
    tags = ["creativity", "multimodal", "vision", "emotion", "art"]

    # 12 emotion categories in alphabetical order
    EMOTION_CATEGORIES = [
        "Alarmed",
        "Annoyed",
        "Aroused",
        "Bored",
        "Calm",
        "Contentment",
        "Excited",
        "Frustrated",
        "Glad",
        "Happy",
        "Sad",
        "Tired"
    ]

    def __init__(self, use_preview: bool = False):
        """
        Args:
            use_preview: If True, use EmoArt-5k (5,532 examples) instead of
                        full EmoArt-130k (132,664 examples)
        """
        super().__init__()
        self.use_preview = use_preview

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load EmoArt dataset and create multimodal instances for emotion classification.

        Args:
            output_path: Directory where downloaded images should be located

        Returns:
            List of instances with image + text inputs
        """
        # Load dataset annotations
        dataset_name = "printblue/EmoArt-5k" if self.use_preview else "printblue/EmoArt-130k"
        dataset = load_dataset(dataset_name, split="train")

        instances = []
        for idx, item in enumerate(dataset):
            # Extract emotion label
            dominant_emotion = item['description']['third_section']['dominant_emotion']

            # Construct image path
            # Images are organized as: Images\[Style]\[filename].jpg
            # We assume images have been extracted to output_path/images/
            image_path = item['image_path'].replace('\\', '/')  # Normalize path separators
            full_image_path = os.path.join(output_path, "images", image_path)

            # Create multimodal content: question text + image + answer choices
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="text/plain",
                    text="Question: What emotion does this artwork primarily evoke?\n"
                ),
                MediaObject(
                    content_type="image/jpeg",
                    location=full_image_path
                ),
                MediaObject(
                    content_type="text/plain",
                    text="\nChoices:\n" + "\n".join(
                        f"{chr(65+i)}) {emotion}"
                        for i, emotion in enumerate(self.EMOTION_CATEGORIES)
                    ) + "\n\nAnswer:"
                )
            ])

            # Build references: all emotion categories, correct one tagged
            references = []
            for i, emotion in enumerate(self.EMOTION_CATEGORIES):
                letter = chr(65 + i)  # A, B, C, D, ...
                is_correct = (emotion == dominant_emotion)
                tags = [CORRECT_TAG] if is_correct else []
                references.append(
                    Reference(Output(text=letter), tags=tags)
                )

            # Create instance
            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=item['request_id']
                )
            )

        return instances
