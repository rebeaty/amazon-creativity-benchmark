"""
HELM Scenario: KiVA (Kid-inspired Visual Analogies)

Paper: KiVA: Kid-inspired Visual Analogies for Testing Large Multimodal Models
       https://arxiv.org/abs/2407.17773
       ICLR 2025

Dataset: https://github.com/ey242/KiVA/releases/tag/0.1
Code: https://github.com/ey242/KiVA

Task:
  Visual analogical reasoning benchmark testing fundamental visual pattern recognition
  and abstraction skills solvable by children ages 3-5. Models must identify which of
  three object transformations is the same as a given training transformation.

Prompt format:
  "Which one of three left-to-right object transformations shown in the bottom row
   is the same as the transformation shown in the top row? Answer with the correct
   letter surrounded by parentheses: (A), (B), or (C)."

Transformation domains (5):
  1. 2DRotation - Rotational transformations (90°, 180°, 270°, 360°)
  2. Colour - Color changes (Blue, Green, Red)
  3. Counting - Quantity changes (adding/removing objects)
  4. Reflect - Reflection transformations
  5. Resize - Size changes

Fields used: transform, correct, incorrect, nochange, train_input_value, train_output_value,
             test_input_value, incorrect_test_output_value
Fields skipped: None

Evaluation: Accuracy (3-way multiple choice: A/B/C)
  - Uses standard exact_match metric
  - See scenarios/kiva/evaluation_notes.md for detailed evaluation protocol
  - No custom annotators required (see scenarios/kiva/annotator_notes.md)

Dataset: 2,100 total instances (700 unique trials × 3 repetitions with randomized answer positions)
  - 2DRotation: 450 instances
  - Colour: 450 instances
  - Counting: 600 instances
  - Reflect: 300 instances
  - Resize: 300 instances

Note: Each trial is repeated 3 times with randomized option positions to control for
      position bias. The scenario can optionally deduplicate to 700 unique trials.
"""

from typing import List
import json
import os
import tempfile
import urllib.request
import zipfile
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


class KiVAScenario(Scenario):
    """
    KiVA: Visual analogical reasoning for testing multimodal models.

    Models must identify which transformation (A/B/C) matches a training example.
    Designed to test skills solvable by 3-5 year old children.
    """

    name = "kiva"
    description = "github.com/ey242/KiVA"
    tags = ["creativity", "multimodal", "visual_reasoning", "analogies"]

    # The 5 transformation domains in KiVA
    DOMAINS = ["2DRotation", "Colour", "Counting", "Reflect", "Resize"]

    # Release URL for KiVA test data
    RELEASE_URL = "https://github.com/ey242/KiVA/releases/download/0.1/single_image.zip"

    def __init__(self, domain: str = "all", deduplicate: bool = False):
        """
        Args:
            domain: Which transformation domain to evaluate. Options:
                   - "all": All 5 domains (2,100 instances)
                   - Individual domain: "2DRotation", "Colour", "Counting", "Reflect", "Resize"
            deduplicate: If True, use only unique trials (700 instances) instead of
                        all repetitions (2,100 instances). Recommended to avoid
                        position bias.
        """
        super().__init__()
        if domain != "all" and domain not in self.DOMAINS:
            raise ValueError(
                f"Invalid domain '{domain}'. Must be 'all' or one of: {', '.join(self.DOMAINS)}"
            )
        self.domain = domain
        self.deduplicate = deduplicate

    def _download_and_extract_data(self, output_path: str) -> str:
        """Download and extract KiVA test data if not already present."""
        data_dir = os.path.join(output_path, "kiva_data")

        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            # Data already downloaded
            return data_dir

        # Download ZIP file
        zip_path = os.path.join(output_path, "single_image.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading KiVA data from {self.RELEASE_URL}...")
            urllib.request.urlretrieve(self.RELEASE_URL, zip_path)

        # Extract ZIP file
        print("Extracting KiVA data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        return os.path.join(data_dir, "single_image")

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load KiVA visual analogy instances.

        Each instance contains:
        - Input: Image showing visual analogy puzzle + text question
        - References: Three answer choices (A, B, C) with correct one tagged
        """
        # Download data
        data_dir = self._download_and_extract_data(output_path)

        instances = []

        # Determine which domains to load
        domains_to_load = self.DOMAINS if self.domain == "all" else [self.domain]

        for domain in domains_to_load:
            # Load JSON metadata for this domain
            json_path = os.path.join(data_dir, f"{domain}.json")
            with open(json_path, 'r') as f:
                domain_data = json.load(f)

            # Track unique trials if deduplicating
            seen_trials = set()

            for trial_key, trial_info in domain_data.items():
                # trial_key format: "ColourBlue_0_0" (subdomain_trial_repetition)
                parts = trial_key.split('_')
                subdomain = parts[0]
                trial_num = parts[1]
                repetition = parts[2]

                # Skip repetitions if deduplicating (keep only repetition 0)
                if self.deduplicate:
                    unique_key = f"{subdomain}_{trial_num}"
                    if unique_key in seen_trials:
                        continue
                    seen_trials.add(unique_key)

                # Get image path
                image_filename = f"{trial_key}_single.jpg"
                image_path = os.path.join(data_dir, image_filename)

                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue

                # Build prompt
                prompt_text = (
                    "Which one of three left-to-right object transformations shown in the "
                    "bottom row is the same as the transformation shown in the top row? "
                    "Answer with the correct letter surrounded by parentheses.\n\n"
                    "Choices:\n(A)\n(B)\n(C)\n\nAnswer:"
                )

                # Create multimodal input with image + text
                multimedia_content = MultimediaObject([
                    MediaObject(
                        content_type="image/jpeg",
                        location=image_path
                    ),
                    MediaObject(
                        content_type="text/plain",
                        text=prompt_text
                    )
                ])

                # Build references: all three choices, correct one tagged
                correct_answer = trial_info["correct"]  # e.g., "(A)"
                answer_letter = correct_answer.strip("()")  # "A"

                references = []
                for choice in ["A", "B", "C"]:
                    is_correct = (choice == answer_letter)
                    tags = [CORRECT_TAG] if is_correct else []
                    references.append(
                        Reference(Output(text=f"({choice})"), tags=tags)
                    )

                # Create instance
                instance = Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"{domain}_{trial_key}"
                )

                instances.append(instance)

        return instances


# Metadata for documentation
"""
KiVA Instance Counts by Domain:
  2DRotation: 450 (or 150 if deduplicated)
  Colour: 450 (or 150 if deduplicated)
  Counting: 600 (or 200 if deduplicated)
  Reflect: 300 (or 100 if deduplicated)
  Resize: 300 (or 100 if deduplicated)

  Total: 2,100 (or 700 if deduplicated)

Trial Metadata Fields:
  - transform: Transformation type (e.g., "ColourBlue", "2DRotation+90")
  - correct: Correct answer position (A/B/C)
  - incorrect: Incorrect transformation position
  - nochange: No-change option position
  - train_input_value: Input value for training transformation
  - train_output_value: Output value for training transformation
  - test_input_value: Input value for test transformation
  - incorrect_test_output_value: Value used for incorrect option

Evaluation Notes:
  - Use exact_match metric with 3-way MC accuracy
  - Recommended to use deduplicate=True to avoid position bias
  - Can evaluate on individual domains to analyze per-domain performance
  - Human baseline (3-5 year olds): ~80-90% accuracy
  - Best LMM performance (GPT-o1): ~60-70% accuracy
"""
