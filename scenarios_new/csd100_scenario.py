"""
HELM Scenario: CSD-100 — Content-Style Recognition in Synthesized Images

Paper: "CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models"
  Nguyen et al. (Qualcomm AI Research), arXiv:2507.13984 (July 2025)
  https://arxiv.org/abs/2507.13984

Dataset: https://huggingface.co/datasets/qualcomm/csd100

Task (original paper): Evaluate visual autoregressive models on content-style
  decomposition — given one image, separate its content (depicted object) and
  style (artistic treatment), then recombine them to generate new styled images.
  This is an image generation task for image generation models.

Task (HELM adaptation — VLM content+style recognition):
  Given a synthesized image, a vision-language model must identify:
    1. The content: the specific object depicted (e.g., "violin", "penguin")
    2. The style: the artistic style applied (e.g., "cubism", "watercolor")
  This tests visual creativity understanding — recognising both WHAT is depicted
  and HOW it is artistically rendered.

Dataset:
  - 100 synthesized images (1024×1024 JPG), one per content+style combination
  - 63 unique content objects, 53 unique artistic styles
  - HuggingFace fields: image (PIL), label (ClassLabel, names = "{content}+{style}")
  - Only "train" split; used as TEST_SPLIT (no held-out split exists)
  - Label examples: "balloon+mosaic", "robot+woodcut", "violin+painting"

Prompt format (standard VLM identification, no paper-specified prompt):
  This image was synthesized by rendering a specific object in a particular
  artistic style. Identify both components.

  Content (the depicted object):
  Style (the artistic style):

Fields used:   image (PIL Image), label (ClassLabel name split on '+')
Fields skipped: none

Evaluation: exact_match on content and style independently (two instances per image)
  - Instance type "content": exact match of object name
  - Instance type "style": exact match of style name (underscores → spaces)
  Total: 200 instances (100 images × 2 question types)

Split: TEST_SPLIT (all 100 images; no test/train split in dataset)
"""

import os
from typing import List

from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject


def _clean_style(style: str) -> str:
    """Replace underscores with spaces for readable style names."""
    return style.replace("_", " ")


class CSD100Scenario(Scenario):
    """
    CSD-100: Content-Style Recognition from Synthesized Images.

    Each image is presented twice — once asking for its content (object) and
    once asking for its style (artistic treatment). Ground truth comes from
    the HuggingFace ClassLabel names in the format '{content}+{style}'.
    """

    name = "csd100"
    description = "qualcomm/csd100"
    tags = ["creativity", "multimodal", "vision", "style", "art"]

    CONTENT_PROMPT = (
        "This image was synthesized by rendering a specific object in a particular "
        "artistic style. What is the object depicted in this image?\n\n"
        "Answer with just the object name (e.g., \"violin\", \"penguin\", \"teapot\"):"
    )

    STYLE_PROMPT = (
        "This image was synthesized by rendering a specific object in a particular "
        "artistic style. What artistic style was used to render this image?\n\n"
        "Answer with just the style name (e.g., \"cubism\", \"watercolor\", \"origami\"):"
    )

    def get_instances(self, output_path: str) -> List[Instance]:
        images_dir = os.path.join(output_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        cache_dir = os.path.join(output_path, "hf_cache")
        dataset = load_dataset("qualcomm/csd100", split="train", cache_dir=cache_dir)
        label_names = dataset.features["label"].names  # e.g. ["balloon+mosaic", ...]

        instances = []
        for idx, item in enumerate(dataset):
            pil_image = item["image"]  # PIL Image
            label_name = label_names[item["label"]]  # e.g. "balloon+mosaic"

            # Save image to disk (Bug Pattern B fix)
            image_path = os.path.join(images_dir, f"{idx}.jpg")
            if not os.path.exists(image_path):
                pil_image.convert("RGB").save(image_path, "JPEG")

            # Parse content and style from label name
            content, style_raw = label_name.split("+", 1)
            style = _clean_style(style_raw)

            # --- Content identification instance ---
            instances.append(
                Instance(
                    input=Input(
                        multimedia_content=MultimediaObject([
                            MediaObject(content_type="text/plain", text=self.CONTENT_PROMPT),
                            MediaObject(content_type="image/jpeg", location=image_path),
                        ])
                    ),
                    references=[Reference(Output(text=content), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

            # --- Style identification instance ---
            instances.append(
                Instance(
                    input=Input(
                        multimedia_content=MultimediaObject([
                            MediaObject(content_type="text/plain", text=self.STYLE_PROMPT),
                            MediaObject(content_type="image/jpeg", location=image_path),
                        ])
                    ),
                    references=[Reference(Output(text=style), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
