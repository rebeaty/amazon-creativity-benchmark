"""
HELM Scenario: PuzzleWorld

Paper: https://arxiv.org/abs/2506.06211
       "PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning
        in Puzzlehunts"
Code: https://github.com/MIT-MI/PuzzleWorld
Dataset: https://huggingface.co/datasets/hzli1202/PuzzleWorld

PuzzleWorld evaluates open-ended, multimodal reasoning on 667 real-world
puzzlehunt problems from Puzzled Pint (2010-2025). Each puzzle combines
text, visual, and structured inputs with no explicit instructions. Models
must infer hidden problem structure and execute multi-step creative
reasoning to arrive at a short canonical answer.

667 puzzles total (easy: 140, medium: 355, hard: 172)
Modalities: text, visual, structured
Skills: logic, spatial, cryptic, wordplay, commonsense, knowledge

Evaluation: exact_match on canonical solution string.
  Most SOTA models achieve only 1-2% accuracy; best model solves 14%.

Prompt format: Standard open-ended puzzle prompt. No specific prompt
  template in paper — models receive puzzle content and flavor text.

Fields used: title, flavor_text, solution, content_file_names, difficulty,
  modality, skills
Fields skipped: reasoning (human-annotated traces, for analysis only),
  source (URL to original PDF)

Note: Content images hosted on HuggingFace dataset repo. Downloaded via
      hf_hub_download. Single "train" split contains all 667 puzzles.
"""

import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class PuzzleWorldScenario(Scenario):
    name = "puzzleworld"
    description = "hzli1202/PuzzleWorld"
    tags = ["creativity", "multimodal", "vision", "lateral_thinking", "puzzles"]

    HF_REPO = "hzli1202/PuzzleWorld"

    def __init__(self, difficulty: str = "all"):
        """
        Args:
            difficulty: Filter by difficulty - "easy", "medium", "hard", or "all"
        """
        super().__init__()
        if difficulty not in ("easy", "medium", "hard", "all"):
            raise ValueError(f"difficulty must be easy/medium/hard/all, got '{difficulty}'")
        self.difficulty = difficulty

    def _download_content(self, file_name: str) -> str:
        """Download a puzzle content file from HuggingFace."""
        return hf_hub_download(
            self.HF_REPO,
            file_name,
            repo_type="dataset",
        )

    def get_instances(self, output_path: str):
        dataset = load_dataset(self.HF_REPO, split="train")

        instances = []
        for item in dataset:
            if self.difficulty != "all" and item["difficulty"] != self.difficulty:
                continue

            # Build multimedia content: puzzle image(s) + flavor text
            media_objects = []

            # Add puzzle content image(s)
            for file_name in item["content_file_names"]:
                try:
                    image_path = self._download_content(file_name)
                    media_objects.append(
                        MediaObject(
                            content_type="image/png",
                            location=image_path,
                        )
                    )
                except Exception:
                    continue

            # Add flavor text as prompt
            prompt_text = (
                f"You are solving a puzzlehunt puzzle.\n\n"
                f"Title: {item['title']}\n\n"
                f"{item['flavor_text']}\n\n"
                f"Study the puzzle image(s) above carefully. "
                f"Figure out the hidden structure and solve the puzzle. "
                f"Your answer should be a short word or phrase in ALL CAPS."
            )

            media_objects.append(
                MediaObject(
                    content_type="text/plain",
                    text=prompt_text,
                )
            )

            multimedia_content = MultimediaObject(media_objects)

            references = [
                Reference(
                    Output(text=item["solution"]),
                    tags=[CORRECT_TAG],
                )
            ]

            title_slug = item["title"].replace(" ", "_")

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT,
                id=f"puzzleworld_{title_slug}",
            ))

        return instances
