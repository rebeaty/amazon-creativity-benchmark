"""
HELM Scenario: HumorDB

Paper: Is AI fun? HumorDB: a curated dataset and benchmark to investigate graphical humor
       https://arxiv.org/abs/2406.13564
       ICCV 2025
Code: https://github.com/kreimanlab/HumorDB
Dataset: kreimanlab/HumorDB (Hugging Face)

Task: Binary classification of visual humor understanding
      Models view an image and classify it as "Funny" or "Not Funny"

Prompt format:
  [IMAGE]
  Is this image funny?

  A) Yes
  B) No

Note: Paper does not specify exact prompt wording. Using standard binary classification format.
      Images are embedded PIL objects in the dataset (JPEG, 564x564 average size).

Dataset composition:
  - 3,545 images (photos 36%, cartoons, sketches, AI-generated content)
  - Test split: 706 images (352 funny, 354 not funny)
  - Binary ratings from human annotations (threshold 0.5 on range ratings)

Fields used: image, binary_rating
Fields skipped: range_ratings_mean (continuous ratings), comparison_ratings (pairwise), words (metadata)
"""

from typing import List
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
from datasets import load_dataset
import os


class HumorDBScenario(Scenario):
    """
    HumorDB: Binary classification of visual humor understanding.

    Models view graphical images and classify them as funny or not funny.
    Tests AI's ability to understand visual humor through photos, cartoons,
    sketches, and AI-generated content.
    """

    name = "humordb"
    description = "kreimanlab/HumorDB"
    tags = ["creativity", "multimodal", "vision", "humor"]

    def __init__(self, split: str = "test"):
        """
        Args:
            split: Dataset split to use (train/validation/test). Default: test
        """
        super().__init__()
        self.split = split

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load HumorDB dataset and create multimodal instances.

        Each instance contains:
        - Image (embedded PIL object)
        - Binary classification question
        - Two references: "Yes" (funny) and "No" (not funny)
        """
        # Load dataset from HuggingFace
        dataset = load_dataset("kreimanlab/HumorDB", split=self.split)

        instances = []
        for idx, item in enumerate(dataset):
            # Save image to temporary file for MediaObject
            # MediaObject requires a file path or URL
            temp_dir = os.path.join(output_path, "temp_images")
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, f"image_{idx}.jpg")

            # Save PIL image to file
            item['image'].save(temp_image_path, format='JPEG')

            # Create multimedia content: image + question
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="image/jpeg",
                    location=temp_image_path
                ),
                MediaObject(
                    content_type="text/plain",
                    text="\nIs this image funny?\n\nA) Yes\nB) No\n\nAnswer:"
                )
            ])

            # Build references: both choices, correct one tagged
            # binary_rating: 1 = Funny, 0 = Not Funny
            is_funny = (item['binary_rating'] == 1)

            references = [
                Reference(
                    Output(text="A"),  # Yes (Funny)
                    tags=[CORRECT_TAG] if is_funny else []
                ),
                Reference(
                    Output(text="B"),  # No (Not Funny)
                    tags=[CORRECT_TAG] if not is_funny else []
                )
            ]

            # Create instance
            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"humordb_{self.split}_{idx}"
                )
            )

        return instances
