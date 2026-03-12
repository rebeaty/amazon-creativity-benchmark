"""
HELM Scenario: MEMECAP

Paper: MemeCap: A Dataset for Captioning and Interpreting Memes (EMNLP 2023)
Paper URL: https://arxiv.org/abs/2305.13703
Code: https://github.com/eujhwang/meme-cap
Dataset: Kaggle (https://www.kaggle.com/datasets/harshittiwari007/meme-convx)
         or Google Drive (see GitHub repo README)

Prompt format:
  [Image of meme]
  Task: Generate a caption that explains what this meme means.
  Title: {title}
  Caption:

Task: Generate interpretive captions for memes (understanding visual metaphors and humor)
Evaluation: Open-ended generation (BLEU, ROUGE, F1 against ground truth meme_captions)

Fields used: img_fname, title, meme_captions (ground truth)
Fields available but not used: img_captions (literal image descriptions),
                                 metaphors (visual metaphor annotations),
                                 url (original Reddit URL)

Dataset: 559 test examples, 5,823 train+val examples (6,382 total)
Images: Must be downloaded separately from Kaggle or Google Drive
"""

import json
import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALID_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject


class MemecapScenario(Scenario):
    """
    MEMECAP: Dataset for captioning and interpreting memes.

    Task: Generate interpretive captions explaining what a meme means.
    Multimodal vision-language task requiring understanding of visual metaphors,
    humor, and cultural context.
    """

    name = "memecap"
    description = "eujhwang/meme-cap"
    tags = ["creativity", "multimodal", "vision", "humor"]

    # GitHub repo URLs for JSON data
    TRAINVAL_URL = "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-trainval.json"
    TEST_URL = "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-test.json"

    def __init__(self, images_dir: str = None):
        """
        Args:
            images_dir: Directory containing downloaded meme images.
                       If None, will look for images in {output_path}/memecap_images/
                       Images must be downloaded from:
                       - Kaggle: https://www.kaggle.com/datasets/harshittiwari007/meme-convx
                       - Google Drive: https://drive.google.com/file/d/1o1IB6am0HdYS58CEOmmxra3WjJkrn-M1/view?usp=sharing
        """
        super().__init__()
        self.images_dir = images_dir

    def _download_json(self, url: str, output_path: str) -> List[dict]:
        """Download and parse JSON data from GitHub."""
        json_filename = os.path.basename(url)
        json_path = os.path.join(output_path, json_filename)

        if not os.path.exists(json_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(url, json_path)

        with open(json_path, 'r') as f:
            return json.load(f)

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load MEMECAP dataset and create multimodal instances.

        Each instance contains:
        - Image of the meme
        - Text prompt with Reddit post title
        - Multiple ground truth interpretive captions (references)
        """
        # Determine images directory
        if self.images_dir is None:
            self.images_dir = os.path.join(output_path, "memecap_images")

        # Check if images directory exists
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}\n"
                f"Please download meme images from:\n"
                f"  Kaggle: https://www.kaggle.com/datasets/harshittiwari007/meme-convx\n"
                f"  or Google Drive: https://drive.google.com/file/d/1o1IB6am0HdYS58CEOmmxra3WjJkrn-M1/view?usp=sharing\n"
                f"and extract them to: {self.images_dir}"
            )

        # Download JSON data
        trainval_data = self._download_json(self.TRAINVAL_URL, output_path)
        test_data = self._download_json(self.TEST_URL, output_path)

        # Use 10% of trainval as validation (following paper methodology)
        val_size = len(trainval_data) // 10
        train_data = trainval_data[val_size:]
        val_data = trainval_data[:val_size]

        instances = []

        # Process each split
        for data, split in [(train_data, TRAIN_SPLIT), (val_data, VALID_SPLIT), (test_data, TEST_SPLIT)]:
            for item in data:
                # Construct image path
                img_fname = item["img_fname"]
                image_path = os.path.join(self.images_dir, img_fname)

                # Check if image exists
                if not os.path.exists(image_path):
                    # Skip if image not found (rather than failing entire load)
                    continue

                # Build prompt with image and title
                title = item["title"]
                prompt_text = f"Task: Generate a caption that explains what this meme means.\nTitle: {title}\nCaption:"

                # Create multimodal content (image + text)
                multimedia_content = MultimediaObject([
                    MediaObject(content_type="image/png", location=image_path),
                    MediaObject(content_type="text/plain", text=prompt_text)
                ])

                # Create references from ground truth meme captions
                # Each meme has multiple interpretive captions (typically 3)
                references = []
                for caption in item["meme_captions"]:
                    references.append(Reference(Output(text=caption), tags=[]))

                instances.append(
                    Instance(
                        input=Input(multimedia_content=multimedia_content),
                        references=references,
                        split=split,
                    )
                )

        return instances
