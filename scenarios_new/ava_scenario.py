"""
HELM Scenario: AVA (Aesthetic Visual Analysis) - Score Prediction

Paper: https://refbase.cvc.uab.es/files/MMP2012a.pdf
Code: https://github.com/imfing/ava_downloader
Dataset: https://huggingface.co/datasets/Iceclear/AVA
Published: CVPR 2012

ADAPTATION NOTE: This is a regression task adaptation of AVA, which was
originally designed for aesthetic assessment research. We frame it as a
score prediction task to evaluate visual aesthetic understanding.

Task: Predict the mean aesthetic score (1-10 scale) for an image based on
human aesthetic ratings. Tests the model's ability to understand and
predict visual aesthetic qualities.

Prompt format:
  Rate the aesthetic quality of this image on a scale of 1 to 10, where
  1 is the lowest aesthetic quality and 10 is the highest.

  Provide only a single number between 1 and 10 as your rating:

Evaluation: Regression metrics (MAE, MSE, correlation) comparing predicted
scores to ground truth mean aesthetic scores computed from ~200 human votes
per image. See metric_notes.md for detailed evaluation setup.

Dataset structure:
- Images: HuggingFace dataset (Iceclear/AVA) - 'train' split only
- Ratings: Downloaded from GitHub (imfing/ava_downloader/AVA.txt)
- Columns 3-12 in AVA.txt: Vote counts for ratings 1-10
- Column 2: Image ID for matching images with ratings
- Computed ground truth: Weighted mean of vote distribution

CRITICAL LIMITATION: HuggingFace dataset (Iceclear/AVA) only has 'image' field,
no IDs or ratings. Cannot match images with AVA.txt ratings without IDs. This
scenario will likely produce ZERO instances with current HuggingFace dataset.

ALTERNATIVES FOR IMPLEMENTATION:
1. Use Kaggle dataset (nicolacarrassi/ava-aesthetic-visual-assessment) which
   includes images + ratings CSV together (requires kaggle API)
2. Match by index (risky - assumes same ordering between HuggingFace and AVA.txt)
3. Download images directly from dpchallenge.com using Image IDs from AVA.txt
4. Use different HuggingFace dataset that includes metadata

Current implementation uses option 2 (index matching) as fallback if no ID field.

Dataset: 255,500 images with aesthetic ratings
  - Each image rated by ~200 human annotators on 1-10 scale
  - Vote distribution allows computing mean and variance
  - Original dataset has train/test splits; HuggingFace only has 'train'
"""

import os
import numpy as np
from typing import List
from PIL import Image
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


class AVAScenario(Scenario):
    """
    AVA: Aesthetic Visual Analysis - Score Prediction

    Regression task evaluating ability to predict aesthetic quality scores
    for images based on human aesthetic ratings.
    """

    name = "ava"
    description = "Iceclear/AVA"
    tags = ["creativity", "aesthetics", "vision", "multimodal", "regression"]

    def __init__(self, max_instances: int = None):
        """
        Args:
            max_instances: Optional limit on number of instances for testing.
                If None, uses full test set (~25,000 images).
        """
        super().__init__()
        self.max_instances = max_instances

    def _compute_mean_score(self, vote_counts: List[int]) -> float:
        """
        Compute weighted mean aesthetic score from vote distribution.

        Args:
            vote_counts: List of 10 integers, counts for ratings 1-10

        Returns:
            Mean aesthetic score on 1-10 scale
        """
        ratings = np.arange(1, 11)  # Ratings 1 through 10
        counts = np.array(vote_counts)

        # Weighted average
        total_votes = counts.sum()
        if total_votes == 0:
            return 5.0  # Default to middle if no votes (shouldn't happen)

        mean_score = np.sum(ratings * counts) / total_votes
        return float(mean_score)

    def _format_prompt(self, image_path: str) -> MultimediaObject:
        """
        Format the aesthetic rating prompt with image.

        Args:
            image_path: Path to saved image file on disk

        Returns:
            MultimediaObject containing prompt text and image
        """
        prompt_text = (
            "Rate the aesthetic quality of this image on a scale of 1 to 10, "
            "where 1 is the lowest aesthetic quality and 10 is the highest.\n\n"
            "Provide only a single number between 1 and 10 as your rating:"
        )

        image_media = MediaObject(content_type="image/jpeg", location=image_path)

        # Combine text and image
        return MultimediaObject([prompt_text, image_media])

    def _download_ava_txt(self, output_path: str) -> str:
        """
        Download AVA.txt ratings file from GitHub.

        Returns:
            Path to AVA.txt file
        """
        import urllib.request
        import os

        ava_txt_path = os.path.join(output_path, "AVA.txt")

        if os.path.exists(ava_txt_path):
            print(f"AVA.txt already exists at {ava_txt_path}")
            return ava_txt_path

        print("Downloading AVA.txt from GitHub...")
        ava_txt_url = "https://raw.githubusercontent.com/imfing/ava_downloader/master/AVA_dataset/AVA.txt"

        os.makedirs(output_path, exist_ok=True)
        urllib.request.urlretrieve(ava_txt_url, ava_txt_path)
        print(f"Downloaded AVA.txt to {ava_txt_path}")

        return ava_txt_path

    def _load_ratings(self, ava_txt_path: str) -> tuple[dict, list]:
        """
        Load aesthetic ratings from AVA.txt file.

        Returns:
            Tuple of:
            - Dictionary mapping image_id -> vote_counts (list of 10 integers)
            - Ordered list of vote_counts (for index-based matching)
        """
        ratings_by_id = {}
        ratings_by_index = []

        with open(ava_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 14:  # Need at least index, id, and 10 vote columns
                    continue

                # Column 1: Index (1-based, parts[0])
                # Column 2: Image ID (parts[1])
                # Columns 3-12: Vote counts for ratings 1-10 (parts[2:12])
                index = int(parts[0])
                image_id = int(parts[1])
                vote_counts = [int(parts[i]) for i in range(2, 12)]

                ratings_by_id[image_id] = vote_counts
                ratings_by_index.append(vote_counts)

        print(f"Loaded ratings for {len(ratings_by_id)} images from AVA.txt")
        return ratings_by_id, ratings_by_index

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate AVA aesthetic score prediction instances.

        Each instance contains:
        - Input: Image with aesthetic rating prompt
        - Reference: Ground truth mean aesthetic score (1-10)
        """
        # Download AVA.txt ratings file
        ava_txt_path = self._download_ava_txt(output_path)
        ratings_by_id, ratings_by_index = self._load_ratings(ava_txt_path)

        # Load images from HuggingFace
        # Note: HuggingFace dataset only has 'train' split
        print("Loading AVA dataset images from HuggingFace...")
        dataset = load_dataset("Iceclear/AVA", split="train")

        instances = []

        # Check if dataset size matches ratings
        if len(dataset) != len(ratings_by_index):
            print(f"Warning: Dataset size ({len(dataset)}) != ratings size ({len(ratings_by_index)})")
            print("Index-based matching may be inaccurate")

        # Limit instances if specified
        num_instances = len(dataset) if self.max_instances is None else min(self.max_instances, len(dataset))
        num_instances = min(num_instances, len(ratings_by_index))  # Don't exceed ratings

        # Note: HuggingFace dataset doesn't include image IDs in metadata
        # Fallback to index-based matching (assumes same ordering as AVA.txt)
        print(f"Using index-based matching (HuggingFace index -> AVA.txt index)")

        for idx in range(num_instances):
            item = dataset[idx]

            # Get image — save PIL object to disk since MediaObject needs a file path
            image = item['image']
            images_dir = os.path.join(output_path, "images")
            os.makedirs(images_dir, exist_ok=True)
            image_path = os.path.join(images_dir, f"{idx}.jpg")
            if not os.path.exists(image_path):
                image.convert("RGB").save(image_path, "JPEG")

            # Try to get image ID from metadata first
            image_id = None
            if 'id' in item:
                image_id = item['id']
            elif 'image_id' in item:
                image_id = item['image_id']

            # Get vote counts
            if image_id is not None and image_id in ratings_by_id:
                # ID-based matching (preferred)
                vote_counts = ratings_by_id[image_id]
            elif idx < len(ratings_by_index):
                # Index-based matching (fallback)
                vote_counts = ratings_by_index[idx]
                image_id = idx + 1  # Use 1-based index as pseudo-ID
            else:
                continue

            # Compute mean aesthetic score
            mean_score = self._compute_mean_score(vote_counts)

            # Format prompt with image
            prompt = self._format_prompt(image_path)

            # Create reference with mean score
            # For regression, the reference is the ground truth numeric value
            references = [
                Reference(
                    Output(text=f"{mean_score:.2f}"),
                    tags=[CORRECT_TAG]
                )
            ]

            instances.append(
                Instance(
                    input=Input(multimedia_content=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"ava_{image_id}"
                )
            )

        print(f"Created {len(instances)} AVA aesthetic score prediction instances")
        return instances
