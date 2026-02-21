"""
HELM Scenario: D-HUMOR (Dark Humor Understanding via Multimodal Open-ended Reasoning)

Paper: D-HUMOR: Dark Humor Understanding via Multimodal Open-ended Reasoning
        (IEEE ICDM 2025, CORE A*)
        https://arxiv.org/abs/2509.06771
Code: https://github.com/Sai-Kartheek-Reddy/D-Humor-Dark-Humor-Understanding-via-Multimodal-Open-ended-Reasoning
Dataset: https://huggingface.co/datasets/UVSKKR/D-Humor (gated - requires access request)

Dataset: 4,379 Reddit memes annotated for dark humor characteristics

Three evaluation tasks:
1. Dark Humor Detection - Binary classification (Yes/No)
2. Target Identification - 6-class classification of target category
3. Intensity Rating - 3-level classification of humor intensity

Prompt format:
  Standard classification format (no explicit instructions provided in paper)
  Multimodal input: meme text + image

Fields used: Text (meme OCR text), post id (for image), Dark/Target/Intensity (labels)
Fields skipped: Explanation1 (model-generated reasoning from paper's proposed method)

Note: This is a GATED dataset. Users must request access via the form at:
      https://forms.gle/t9ynkpq4XGd8Kp93A
      The dataset requires authentication to access from HuggingFace.
"""

from typing import List, Optional
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject


class DHumorDetectionScenario(Scenario):
    """
    Dark Humor Detection (Binary Classification)

    Evaluates whether a meme contains dark humor (Yes/No).
    """

    name = "d_humor_detection"
    description = "UVSKKR/D-Humor"
    tags = ["creativity", "multimodal", "vision", "humor", "classification"]

    def __init__(self, use_huggingface: bool = True):
        """
        Args:
            use_huggingface: If True, load from HuggingFace (requires authentication).
                           If False, expects local dataset files.
        """
        super().__init__()
        self.use_huggingface = use_huggingface

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load D-HUMOR dataset and create instances for dark humor detection.

        Note: This dataset is gated and requires authentication.
        Users must request access at https://forms.gle/t9ynkpq4XGd8Kp93A
        """
        instances = []

        if self.use_huggingface:
            # Load from HuggingFace (requires authentication)
            # Note: This will fail if user hasn't been granted access
            try:
                dataset = load_dataset("UVSKKR/D-Humor", split="test")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load D-Humor dataset. This is a gated dataset. "
                    f"Please request access at https://forms.gle/t9ynkpq4XGd8Kp93A "
                    f"and ensure you are authenticated with HuggingFace. Error: {e}"
                )
        else:
            # Load from local files (if available)
            # Users would need to download the dataset after approval
            raise NotImplementedError(
                "Local dataset loading not yet implemented. "
                "Please use HuggingFace with authentication."
            )

        for item in dataset:
            # Create multimodal content: meme text + image
            # The Text field contains OCR-extracted text from the meme
            # The image is referenced by post_id
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="text/plain",
                    text=f"Meme text: {item['Text']}\n\nDoes this meme contain dark humor?"
                ),
                MediaObject(
                    content_type="image/jpeg",
                    # Assuming images are accessible via HuggingFace dataset
                    # May need adjustment based on actual dataset structure
                    location=item.get('image_path', f"images/{item['post id']}")
                ),
                MediaObject(
                    content_type="text/plain",
                    text="\n\nAnswer: Yes or No?"
                )
            ])

            # Binary classification: Yes (1) or No (0)
            label = item['Dark']
            references = [
                Reference(Output(text="Yes"), tags=[CORRECT_TAG] if label == 1 else []),
                Reference(Output(text="No"), tags=[CORRECT_TAG] if label == 0 else [])
            ]

            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"d_humor_detection_{item['post id']}"
                )
            )

        return instances


class DHumorTargetScenario(Scenario):
    """
    Target Identification (6-class Classification)

    Identifies the target category of dark humor in a meme.

    Categories:
    0: Gender/Sex-Related Topics
    1: Mental Health
    2: Disability
    3: Race/Ethnicity
    4: Violence/Death
    5: Other
    """

    name = "d_humor_target"
    description = "UVSKKR/D-Humor"
    tags = ["creativity", "multimodal", "vision", "humor", "classification"]

    # Target category labels based on paper
    TARGET_LABELS = [
        "Gender/Sex-Related",
        "Mental Health",
        "Disability",
        "Race/Ethnicity",
        "Violence/Death",
        "Other"
    ]

    def __init__(self, use_huggingface: bool = True):
        """
        Args:
            use_huggingface: If True, load from HuggingFace (requires authentication).
        """
        super().__init__()
        self.use_huggingface = use_huggingface

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load D-HUMOR dataset and create instances for target identification.
        """
        instances = []

        if self.use_huggingface:
            try:
                dataset = load_dataset("UVSKKR/D-Humor", split="test")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load D-Humor dataset. This is a gated dataset. "
                    f"Please request access at https://forms.gle/t9ynkpq4XGd8Kp93A "
                    f"and ensure you are authenticated with HuggingFace. Error: {e}"
                )
        else:
            raise NotImplementedError(
                "Local dataset loading not yet implemented. "
                "Please use HuggingFace with authentication."
            )

        for item in dataset:
            # Create multimodal content with target options
            target_options = "\n".join(
                f"{chr(65+i)}) {label}"
                for i, label in enumerate(self.TARGET_LABELS)
            )

            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="text/plain",
                    text=f"Meme text: {item['Text']}\n\nWhat is the target of the dark humor in this meme?\n\n{target_options}"
                ),
                MediaObject(
                    content_type="image/jpeg",
                    location=item.get('image_path', f"images/{item['post id']}")
                ),
                MediaObject(
                    content_type="text/plain",
                    text="\n\nAnswer:"
                )
            ])

            # 6-class classification
            label = item['Target']
            references = []
            for i in range(len(self.TARGET_LABELS)):
                letter = chr(65 + i)  # A, B, C, D, E, F
                is_correct = (i == label)
                tags = [CORRECT_TAG] if is_correct else []
                references.append(
                    Reference(Output(text=letter), tags=tags)
                )

            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"d_humor_target_{item['post id']}"
                )
            )

        return instances


class DHumorIntensityScenario(Scenario):
    """
    Intensity Classification (3-class Classification)

    Classifies the intensity level of dark humor in a meme.

    Levels:
    1: Mild
    2: Moderate
    3: Severe
    """

    name = "d_humor_intensity"
    description = "UVSKKR/D-Humor"
    tags = ["creativity", "multimodal", "vision", "humor", "classification"]

    # Intensity level labels
    INTENSITY_LABELS = {
        1: "Mild",
        2: "Moderate",
        3: "Severe"
    }

    def __init__(self, use_huggingface: bool = True):
        """
        Args:
            use_huggingface: If True, load from HuggingFace (requires authentication).
        """
        super().__init__()
        self.use_huggingface = use_huggingface

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load D-HUMOR dataset and create instances for intensity classification.
        """
        instances = []

        if self.use_huggingface:
            try:
                dataset = load_dataset("UVSKKR/D-Humor", split="test")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load D-Humor dataset. This is a gated dataset. "
                    f"Please request access at https://forms.gle/t9ynkpq4XGd8Kp93A "
                    f"and ensure you are authenticated with HuggingFace. Error: {e}"
                )
        else:
            raise NotImplementedError(
                "Local dataset loading not yet implemented. "
                "Please use HuggingFace with authentication."
            )

        for item in dataset:
            # Create multimodal content with intensity options
            intensity_options = "\n".join(
                f"{chr(65+i)}) {self.INTENSITY_LABELS[i+1]}"
                for i in range(3)
            )

            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="text/plain",
                    text=f"Meme text: {item['Text']}\n\nWhat is the intensity level of the dark humor in this meme?\n\n{intensity_options}"
                ),
                MediaObject(
                    content_type="image/jpeg",
                    location=item.get('image_path', f"images/{item['post id']}")
                ),
                MediaObject(
                    content_type="text/plain",
                    text="\n\nAnswer:"
                )
            ])

            # 3-class classification (labels are 1, 2, 3)
            label = item['Intensity']
            references = []
            for i in range(3):
                letter = chr(65 + i)  # A, B, C
                is_correct = (label == i + 1)  # Labels are 1-indexed
                tags = [CORRECT_TAG] if is_correct else []
                references.append(
                    Reference(Output(text=letter), tags=tags)
                )

            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"d_humor_intensity_{item['post id']}"
                )
            )

        return instances
