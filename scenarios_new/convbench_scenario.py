"""
HELM Scenario: ConvBench

Paper: ConvBench: A Multi-Turn Conversation Evaluation Benchmark with
       Hierarchical Capability for Large Vision-Language Models
       NeurIPS 2024
       https://arxiv.org/abs/2403.20194
Code:  https://github.com/shirlyliu64/ConvBench
Data:  https://huggingface.co/datasets/liushuo12345/ConvBench

Task: Given an image and the context of a 2-turn perceptual/reasoning conversation,
generate a creative response for the third turn (e.g., poems, slogans, stories,
recipes, travel plans, essays). The benchmark tests whether models can leverage
visual understanding and reasoning to produce contextually grounded creative content.

Dataset: 578 instances (ConvBench.xlsx), each a 3-turn conversation over an image.
  - Turn 1: Perception (basic recognition/OCR/description)
  - Turn 2: Reasoning (commonsense, meme interpretation, domain knowledge)
  - Turn 3: Creation (the creativity task — poem, story, slogan, recipe, etc.)

HELM adaptation: The benchmark is natively multi-turn, but we flatten it into a
single-turn prompt by prepending turns 1+2 (with their gold reference answers) as
conversation context. This gives the model the perceptual/reasoning groundwork
without requiring true multi-turn state maintenance.

Images: 574 unique images from VisIT-Bench, stored in visit_bench_images/ in the
HuggingFace repo. Downloaded via snapshot_download.

Prompt format (no explicit prompt template in paper; standard instruction format used):
  [image]

  The following conversation is about the image above.

  Turn 1 - Question: {first_turn_instruction}
  Turn 1 - Answer: {first_turn_answer}

  Turn 2 - Question: {second_turn_instruction}
  Turn 2 - Answer: {second_turn_answer}

  Now complete Turn 3 - {third_turn_instruction}

Fields used: image_id (image), The_first_turn_instruction, first_turn_answer,
  The_second_turn_instruction, second_turn_answer, The_third_turn_instruction
  (prompt), third_turn_answer (gold reference)
Fields skipped: instruction-conditioned-caption (not used in model evaluation),
  instruction_category / *_category fields (metadata only)

Evaluation: LLM-as-judge using third_turn_demands rubric; see annotator_notes.md
"""

import os
from typing import List

import pandas as pd
from huggingface_hub import snapshot_download

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject

_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ConvBenchScenario(Scenario):
    """
    ConvBench: vision-grounded creative generation via flattened 3-turn context.

    Each instance presents an image plus the Q&A from turns 1 and 2 as context,
    then asks the model to complete turn 3 (the creative generation task).
    """

    name = "convbench"
    description = "liushuo12345/ConvBench"
    tags = ["creativity", "multimodal", "vision", "open_ended"]

    def get_instances(self, output_path: str) -> List[Instance]:
        repo_dir = snapshot_download(
            repo_id="liushuo12345/ConvBench",
            repo_type="dataset",
            cache_dir=output_path,
        )

        xlsx_path = os.path.join(repo_dir, "ConvBench.xlsx")
        df = pd.read_excel(xlsx_path, sheet_name="multi_turn_benchmark")

        instances = []
        for _, row in df.iterrows():
            image_id = str(row["image_id"]).strip()
            image_path = os.path.join(repo_dir, "visit_bench_images", image_id)
            if not os.path.exists(image_path):
                continue

            ext = os.path.splitext(image_id)[1].lower()
            content_type = _CONTENT_TYPES.get(ext, "image/png")

            # Flatten turns 1+2 (with gold answers) as conversation context,
            # then pose turn 3 as the task to complete.
            t1_q = str(row["The_first_turn_instruction"]).strip()
            t1_a = str(row["first_turn_answer"]).strip()
            t2_q = str(row["The_second_turn_instruction"]).strip()
            t2_a = str(row["second_turn_answer"]).strip()
            t3_q = str(row["The_third_turn_instruction"]).strip()

            conversation = (
                "The following conversation is about the image above.\n\n"
                f"Turn 1 - Question: {t1_q}\n"
                f"Turn 1 - Answer: {t1_a}\n\n"
                f"Turn 2 - Question: {t2_q}\n"
                f"Turn 2 - Answer: {t2_a}\n\n"
                f"Now complete Turn 3 - {t3_q}"
            )

            multimedia_content = MultimediaObject([
                MediaObject(content_type=content_type, location=image_path),
                MediaObject(content_type="text/plain", text=conversation),
            ])

            gold_text = str(row["third_turn_answer"]).strip()
            references = (
                [Reference(Output(text=gold_text), tags=[CORRECT_TAG])]
                if gold_text and gold_text.lower() != "nan" else []
            )

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT,
            ))

        return instances
