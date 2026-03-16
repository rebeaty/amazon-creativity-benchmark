"""
HELM Scenario: CII-Bench (Chinese Image Implication Understanding Benchmark)

Paper: https://arxiv.org/abs/2401.11944 (ACL 2025)
Code: https://github.com/MING-ZCH/CII-Bench
Dataset: https://huggingface.co/datasets/m-a-p/CII-Bench

Related: "Let Androids Dream of Electric Sheep"
  https://github.com/MING-ZCH/Let-Androids-Dream-of-Electric-Sheep

Task: Chinese image implication understanding — evaluating MLLMs' ability to comprehend
deeper cultural and metaphorical meanings embedded in Chinese images, including abstract
artworks, comics, internet memes, and posters. Questions assess metaphors, symbolism,
and cultural context specific to Chinese visual content.

Dataset composition:
  - 698 manually collected Chinese images
  - 800 multiple-choice questions (6 options each, single correct answer)
  - Split: 765 test questions (answers available), 35 dev questions
  - Image types: posters, comics, traditional artworks, internet memes
  - Domains: life, society, art, environment, psychology, others

Performance gap: MLLMs peak at ~64.4% accuracy vs. human average of 78.2%
(human peak 81.0%), indicating significant challenges in Chinese cultural reasoning.

Prompt format (zero-shot baseline):
  <image>
  {question}
  A. {option1}
  B. {option2}
  C. {option3}
  D. {option4}
  E. {option5}
  F. {option6}
  Answer:

Prompt source: Standard multiple-choice format (same as II-Bench; paper evaluates
zero-shot and few-shot modes with domain/emotion/rhetoric hints)

Fields used: image, question, option1-6, answer, correct_option
Fields skipped: id, image_type, difficulty, domain, emotion, rhetoric,
               explanation, metaphorical_meaning, local_path (metadata)
"""

from typing import List
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class CIIBenchScenario(Scenario):
    name = "cii_bench"
    description = "m-a-p/CII-Bench"
    tags = ["creativity", "visual_reasoning", "implication", "metaphor", "multimodal",
            "vision", "chinese", "cultural_understanding"]

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("m-a-p/CII-Bench", split="test")

        instances = []
        for item in dataset:
            instances.append(self._create_instance(item))
        return instances

    def _create_instance(self, item: dict) -> Instance:
        image = item["image"]  # PIL Image object
        question = item["question"]
        options = [
            item["option1"],
            item["option2"],
            item["option3"],
            item["option4"],
            item["option5"],
            item["option6"],
        ]
        answer_letter = item["answer"]  # 'A'–'F'

        # Build prompt text
        prompt_text = f"{question}\n\n"
        for i, option in enumerate(options):
            letter = chr(65 + i)
            prompt_text += f"{letter}. {option}\n"
        prompt_text += "\nAnswer:"

        multimedia_content = MultimediaObject([
            MediaObject(
                content_type="image/jpeg",
                location=image
            ),
            MediaObject(
                content_type="text/plain",
                text=prompt_text
            )
        ])

        # All 6 options as references; correct one tagged
        answer_index = ord(answer_letter) - ord("A")
        references = [
            Reference(
                Output(text=chr(65 + i)),
                tags=[CORRECT_TAG] if i == answer_index else []
            )
            for i in range(6)
        ]

        return Instance(
            input=Input(multimedia_content=multimedia_content),
            references=references,
            split=TEST_SPLIT
        )
