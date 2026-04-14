"""
HELM Scenario: CreativePair — Advertising Creative Image Selection

Paper: "Creative4U: MLLMs-based Advertising Creative Image Selector with
        Comparative Reasoning" (arXiv:2508.12628)
Authors: Alibaba Group / University of Science and Technology Beijing

Task: Given two advertising images (A and B) for the same product and user
query, determine which creative is more likely to attract clicks (higher CTR).
The model uses a structured 10-question Creative Evaluation Protocol covering
image-text consistency, visual quality dimensions, and a final selection.

This is a multimodal (vision-language) task requiring actual ad images.
Evaluation: exact_match (binary accuracy: is the selected image A or B correct?)

⚠️  DATA AVAILABILITY NOTE:
The CreativePair dataset has not been publicly released as of the scenario
creation date. The authors state: "Our code and dataset will be made public
to advance research and industrial applications."

To run this scenario:
1. Obtain the dataset from the authors (paper contact: Alibaba Group / USTB)
2. Place the data in the HELM output_path directory with this structure:
   {output_path}/
     creative_pair_test.json     ← test split (1,729 pairs)
     images/
       {image_id_a}.jpg          ← image A for each pair
       {image_id_b}.jpg          ← image B for each pair

Expected JSON format for creative_pair_test.json:
  [
    {
      "image_a": "images/img_001_a.jpg",   ← path relative to output_path
      "image_b": "images/img_001_b.jpg",
      "product_title": "...",              ← product title string
      "query": "...",                      ← high-frequency user query
      "label": "A"                         ← CTR winner: "A" or "B"
    },
    ...
  ]
  (Labels derived from real CTR data; pairs kept only when CTR difference > 60%
   and each image has > 1,000 impressions)

Prompt (verbatim from paper, Creative Evaluation Protocol):
  "Please answer each question in the Creative Evaluation Protocol based on
   the high-frequency queries and product information, providing detailed
   explanations for your answers.
   High-frequency queries: {query}; Product title: {title};
   Creative A: <image>, Creative B: <image>;
   {Creative Evaluation Protocol};
   Output format: <think>comparative reasoning</think><answer>A or B</answer>"

Creative Evaluation Protocol (10 questions):
  Q1. Are the two images the same? (YES/NO)
  Q2. The two images are very similar, making judgment impossible? (YES/NO)
  Q3. Hit rate of image content on query (text in image, elements, visual
      style): A>B / A=B / A<B
  Q4. Hit rate of image content on title (text in image, elements, visual
      style): A>B / A=B / A<B
  Q5. Text in Image quality (product info, pain points, selling points,
      CTAs, applicable scenarios): A>B / A=B / A<B
  Q6. Models and Props (presence and functional relevance): A>B / A=B / A<B
  Q7. Layout (borders, picture-in-picture): A>B / A=B / A<B
  Q8. Product Subject (position, size, angle, completeness): A>B / A=B / A<B
  Q9. Background Design (color harmony, scene consistency, aesthetics):
      A>B / A=B / A<B
  Q10. Which image is the user more likely to click on? (A / B)  ← label

Prompt source: Verbatim from paper Section 3.2 and Figure 2.
Fields used: image_a, image_b, product_title, query, label
Evaluation: exact_match (A or B)
Dataset: 1,729 test pairs (full dataset: 8,817; also 10K CoT-augmented version)
"""

import json
import os
from typing import List

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

_DATA_FILENAME = "creative_pair_test.json"

# Verbatim Creative Evaluation Protocol from paper (Section 3.2)
_PROTOCOL = """\
Creative Evaluation Protocol:
Q1. Are the two images the same? (YES / NO)
Q2. The two images are very similar, making it impossible to make a judgment? (YES / NO)
Q3. The hit rate of image content on query (based on text in the image; elements in the image; conveyed visual style): A > B / A = B / A < B
Q4. The hit rate of image content on title (based on text in the image; elements in the image; conveyed visual style): A > B / A = B / A < B
Q5. Text in Image (product information, pain-point addressing, selling points, calls-to-action, applicable scenarios): A > B / A = B / A < B
Q6. Models and Props (presence and functional relevance): A > B / A = B / A < B
Q7. Layout (decorative borders, picture-in-picture elements): A > B / A = B / A < B
Q8. Product Subject (positioning, size, angle, completeness, quantity, usage state): A > B / A = B / A < B
Q9. Background Design (color harmony, scene consistency, aesthetic quality): A > B / A = B / A < B
Q10. Which image is the user more likely to click on? (A / B)"""

# Output format instruction from paper
_OUTPUT_FORMAT = (
    "Output format: "
    "<think>[comparative reasoning process]</think>"
    "<answer>[A or B]</answer>"
)


def _build_prompt_text(query: str, title: str) -> str:
    """Build the textual portion of the prompt (verbatim structure from paper)."""
    return (
        "Please answer each question in the Creative Evaluation Protocol "
        "based on the high-frequency queries and product information, "
        "providing detailed explanations for your answers.\n\n"
        f"High-frequency queries: {query}\n"
        f"Product title: {title}\n\n"
        f"{_PROTOCOL}\n\n"
        f"{_OUTPUT_FORMAT}"
    )


class CreativePairScenario(Scenario):
    """
    CreativePair: multimodal advertising creative selection.

    Given two ad images (A and B) with product title and user query, the model
    applies the 10-question Creative Evaluation Protocol and selects the image
    more likely to attract clicks. Ground truth: real CTR winner (pairs with
    > 60% CTR gap, > 1,000 impressions each).

    ⚠️  Requires dataset from authors — not yet publicly released.
    See scenario header for data placement instructions.
    """

    name = "creative_pair"
    description = "arXiv:2508.12628 (Creative4U / CreativePair)"
    tags = ["creativity", "advertising", "multimodal", "visual_reasoning", "classification"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, _DATA_FILENAME)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"CreativePair dataset not found at {data_path}.\n"
                "The dataset has not been publicly released yet. "
                "Contact the authors (arXiv:2508.12628, Alibaba Group / USTB) "
                "to request access. Once obtained, place the data at:\n"
                f"  {output_path}/\n"
                f"    {_DATA_FILENAME}   (1,729 test pairs)\n"
                "    images/             (ad image files)\n"
                "See the scenario docstring for the expected JSON format."
            )

        with open(data_path, encoding="utf-8") as f:
            records = json.load(f)

        instances = []
        for rec in records:
            image_a_path = os.path.join(output_path, rec["image_a"])
            image_b_path = os.path.join(output_path, rec["image_b"])
            query = rec["query"].strip()
            title = rec["product_title"].strip()
            label = rec["label"].strip().upper()  # "A" or "B"

            prompt_text = _build_prompt_text(query, title)

            # Determine image content type from extension
            def _content_type(path: str) -> str:
                ext = os.path.splitext(path)[1].lower()
                return {
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp",
                    ".gif": "image/gif",
                }.get(ext, "image/jpeg")

            # Build multimodal input: text prompt + Image A + Image B
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="text/plain",
                    text=prompt_text,
                ),
                MediaObject(
                    content_type=_content_type(image_a_path),
                    location=image_a_path,
                ),
                MediaObject(
                    content_type="text/plain",
                    text="[Creative B]",
                ),
                MediaObject(
                    content_type=_content_type(image_b_path),
                    location=image_b_path,
                ),
            ])

            references = [
                Reference(
                    Output(text="A"),
                    tags=[CORRECT_TAG] if label == "A" else [],
                ),
                Reference(
                    Output(text="B"),
                    tags=[CORRECT_TAG] if label == "B" else [],
                ),
            ]

            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
