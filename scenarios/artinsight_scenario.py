"""
HELM Scenario: ArtInsight Artwork Description Evaluation

Paper: ArtInsight: Supporting Blind and Low-Vision Families in Experiencing
       Children's Artwork
       ACM IUI 2025
       https://arxiv.org/abs/2502.19263
Code:  https://github.com/makeabilitylab/ArtInsight

Task: Given an image of a child's artwork, generate a description suitable for
a blind or low-vision (BLV) parent to understand what their child created.
The description must be factual, non-presumptive, non-reductive, and cover all
visible elements (colors, shapes, textures, text, composition). This tests
the model's ability to produce creative, empathetic, and detailed visual
descriptions of informal art — a distinct challenge from standard image captioning.

Dataset: 30 children's artwork JPEG images (makeabilitylab/ArtInsight,
  Artwork-Description-Scoring/images/). Images were collected with parental
  consent; all personally identifiable information has been removed.

Prompt format — based on optimal prompt described in
  Artwork-Description-Scoring/README.md:

  You are Art Insight, an assistant helping blind and low-vision parents
  experience their child's artwork. Describe the artwork in the image using
  only what you can directly observe. Use precise, neutral, and non-diminishing
  language. Cover all major and minor elements including colors, shapes,
  textures, any text, and overall composition. Do not interpret, speculate about
  meaning, or ask questions. Treat all artistic choices with equal respect.

  Describe the artwork:

Prompt source: README.md optimal system prompt description (verbatim prompt
  text not published; above is a faithful reconstruction from README description)
Fields used: image files (input)
Fields skipped: Artwork_Descriptions_Text.csv, descriptions/*.txt (model-generated
  outputs, not ground truth), Artwork_Description_Scores.csv (pre-computed scores)

Evaluation: LLM-as-judge (GPT-4o with vision); see annotator_notes.md
  Rubric: 5 criteria, 0-4 points each, 16 points total
  (not-presumptive, not-reductive, sufficient detail, all elements captured,
   miscellaneous deductions)
"""

import os
import urllib.parse
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject

_REPO_RAW = (
    "https://raw.githubusercontent.com/makeabilitylab/ArtInsight/main"
    "/Artwork-Description-Scoring/images"
)

# All 30 artwork image filenames from the repository
_IMAGE_FILENAMES = [
    "Abstract_Work_Blue.jpeg",
    "Airplane.jpeg",
    "CountryBalls.jpeg",
    "IMG_2271.jpeg",
    "IMG_2272.jpeg",
    "IMG_2273.jpeg",
    "P6 IMG_0511.jpeg",
    "P6 IMG_0512.jpeg",
    "Vampire.jpeg",
    "Xmas_Artwork.jpeg",
    "abstract_blue_white_red.jpeg",
    "bulldog.jpeg",
    "centaur.jpeg",
    "craft_flower.jpeg",
    "creature_hello.jpeg",
    "imprints.jpeg",
    "love_monster.jpeg",
    "mandala.jpeg",
    "mandala_cat.jpeg",
    "mom-daughter.jpeg",
    "pandas.jpeg",
    "peeps.jpeg",
    "pink_flower.jpeg",
    "poms.jpeg",
    "rainbow.jpeg",
    "skull_face_jones.jpeg",
    "sunflower.jpeg",
    "turkey.jpeg",
    "two_people.jpeg",
    "xmas_tree.jpeg",
]

_PROMPT = (
    "You are Art Insight, an assistant helping blind and low-vision parents "
    "experience their child's artwork. Describe the artwork in the image using "
    "only what you can directly observe. Use precise, neutral, and non-diminishing "
    "language. Cover all major and minor elements including colors, shapes, "
    "textures, any text, and overall composition. Do not interpret, speculate "
    "about meaning, or ask questions. Treat all artistic choices with equal "
    "respect.\n\n"
    "Describe the artwork:"
)


class ArtInsightScenario(Scenario):
    """
    ArtInsight: describe children's artwork for blind/low-vision parents.

    30 instances, one per artwork image. Open-ended generation evaluated by
    LLM-as-judge on a 0-16 rubric (see annotator_notes.md).
    """

    name = "artinsight"
    description = "makeabilitylab/ArtInsight"
    tags = ["creativity", "multimodal", "vision", "description_generation"]

    def get_instances(self, output_path: str) -> List[Instance]:
        images_dir = os.path.join(output_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        instances = []
        for filename in _IMAGE_FILENAMES:
            local_path = os.path.join(images_dir, filename)

            if not os.path.exists(local_path):
                # URL-encode spaces in filenames
                encoded = urllib.parse.quote(filename)
                url = f"{_REPO_RAW}/{encoded}"
                urllib.request.urlretrieve(url, local_path)

            multimedia_content = MultimediaObject([
                MediaObject(content_type="image/jpeg", location=local_path),
                MediaObject(content_type="text/plain", text=_PROMPT),
            ])

            # Open-ended generation — no gold text reference; judge evaluates output
            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=[],
                split=TEST_SPLIT,
            ))

        return instances
