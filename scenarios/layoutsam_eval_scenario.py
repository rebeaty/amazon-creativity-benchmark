"""
HELM Scenario: LayoutSAM-Eval (Layout-to-Scene Description)

Paper: LayoutSAM: Enhancing Layout-to-Image Generation via Large Language Models
       https://arxiv.org/abs/2412.03859
Site:  https://creatilayout.github.io/
Data:  https://huggingface.co/datasets/HuiZhang0812/LayoutSAM-eval

LayoutSAM-Eval is a benchmark for evaluating layout-to-image generation quality
across 5,000 images derived from the SAM (Segment Anything) dataset. Each image
is annotated with:
  - Bounding boxes for individual regions
  - Short region labels (e.g. "orange - red sky", "yellow cargo ship")
  - Detailed region descriptions
  - A global scene caption for the full image

The original benchmark evaluates image generation systems via VLM-based Yes/No
spatial and attribute QA (using MiniCPM-V-2.6) plus image quality metrics
(FID, CLIP, PickScore, IS).

HELM task (text-only reframing):
  Given a layout specification — image dimensions, bounding-box positions, and
  short region labels — generate a one-paragraph description of the complete
  scene. This evaluates spatial compositional reasoning: the model must infer
  how individual spatial regions combine into a coherent whole.

  Input example:
    You are given a spatial layout specification for an image (2667 × 1500 pixels).
    ...
    Layout:
      Region 1: [6, 7, 2660, 912] — orange - red sky
      Region 2: [9, 921, 2657, 1490] — river
      Region 3: [444, 892, 1089, 1096] — yellow cargo ship

    Based on this spatial layout, write a one-paragraph description of the
    complete scene.

  Reference: global_caption (human-annotated scene description)

Prompt source: No text-only prompt specified in the paper (which evaluates image
  generation systems). Standard layout-to-description format used here.

Fields used:   global_caption (reference), region_captions (region labels),
               bbox_list (bounding boxes), width, height, image_id
Fields skipped: image (not needed for text-only task),
                detail_region_captions (too detailed; would make task trivial),
                file_name

Dataset quirks:
  - region_captions, detail_region_captions, bbox_list stored as JSON-encoded
    strings — must be parsed with json.loads() before use.
  - Single split named "test" (5,000 rows); no train split.

Evaluation: open_ended (ROUGE-L, BLEU)
"""

import json
from typing import List

from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

_PROMPT_TEMPLATE = (
    "You are given a spatial layout specification for an image "
    "({width} \u00d7 {height} pixels). "
    "Each region is defined by its bounding box [x1, y1, x2, y2] "
    "(absolute pixel coordinates) and a short label.\n\n"
    "Layout:\n"
    "{regions}\n\n"
    "Based on this spatial layout, write a one-paragraph description "
    "of the complete scene."
)


class LayoutSAMEvalScenario(Scenario):
    """
    LayoutSAM-Eval: given a spatial layout (bounding boxes + region labels),
    generate a coherent one-paragraph description of the complete scene.

    5,000 test instances from the SAM dataset covering diverse real-world scenes
    (nature, architecture, urban, indoor, etc.).
    """

    name = "layoutsam_eval"
    description = "HuiZhang0812/LayoutSAM-eval"
    tags = ["creativity", "layout_understanding", "scene_description", "spatial_reasoning"]

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("HuiZhang0812/LayoutSAM-eval", split="test")

        instances = []
        for item in dataset:
            region_captions = json.loads(item["region_captions"])
            bbox_list = json.loads(item["bbox_list"])

            region_lines = []
            for idx, (caption, bbox) in enumerate(zip(region_captions, bbox_list), 1):
                x1, y1, x2, y2 = bbox
                region_lines.append(
                    f"  Region {idx}: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] \u2014 {caption}"
                )

            prompt = _PROMPT_TEMPLATE.format(
                width=item["width"],
                height=item["height"],
                regions="\n".join(region_lines),
            )

            instances.append(Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=item["global_caption"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
                extra_data={"image_id": str(item["image_id"])},
            ))

        return instances
