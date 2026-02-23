"""
HELM Scenario: YESBUT V2

Paper: "When 'YES' Meets 'BUT': Can Large Models Comprehend Contradictory
       Humor Through Comparative Reasoning?"
       https://arxiv.org/abs/2503.23137
Code: https://github.com/Tuo-Liang/YESBUT
Dataset: https://huggingface.co/datasets/zhehuderek/YESBUT_Benchmark_V2
Website: https://vulab-ai.github.io/YESBUT-v2/

YESBUT V2 evaluates VLMs on understanding contradictory humor in two-panel
comics. Extends V1 (348 images) to 1,262 comics from diverse multilingual
and multicultural contexts.

Four tasks (implemented as subsets):
  - description: Generate literal description of comic narrative (open_ended)
  - contradiction: Explain the contradiction between panels (open_ended)
  - moral_mcq: Select underlying symbolism from 4 options (exact_match)
  - title_mcq: Select best title from 4 options (exact_match)

Each task supports two variants:
  - w_caption: Image + oracle caption provided as context
  - wo_caption: Image only (default, harder)

Three prompt sets from Table 7 (paper averages across all three to reduce bias).
Prompt source: Table 7 evaluation prompts from the paper.
Fields used: description, caption, contradiction, moral_mcq, moral_mcq_answer,
  title_mcq, title_mcq_answer, url (Google Drive image links)
Fields skipped: social_info, Linguistic_context, Panel_Bounding_Boxes,
  Context_Bounding_Boxes, contain_text, category, original_url
"""

import os
import re
import urllib.request
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class YesButV2Scenario(Scenario):
    name = "yesbut_v2"
    description = "zhehuderek/YESBUT_Benchmark_V2"
    tags = ["creativity", "multimodal", "vision", "humor", "satire"]

    TASKS = ["description", "contradiction", "moral_mcq", "title_mcq"]

    # Evaluation prompts from Table 7 — three prompt sets per task.
    # Paper averages results across all three to reduce prompt bias.
    PROMPTS = {
        "description": {
            1: {
                "wo_caption": (
                    "The given comic shows the same situation from two "
                    "opposite sides with contradictions. Write a "
                    "one-paragraph literal description to describe the "
                    "narrative of the comic."
                ),
            },
            2: {
                "wo_caption": (
                    "Please literally describe the context of the image "
                    "in detail."
                ),
            },
            3: {
                "wo_caption": (
                    "Give me a detailed literal description of the image."
                ),
            },
        },
        "contradiction": {
            1: {
                "w_caption": (
                    "The given comic shows the same situation from two "
                    "opposite sides with contradictions.\n"
                    "The literal caption of the comic is: {image_caption}\n"
                    "Write a short explanation to illustrate the "
                    "contradiction of the two sides."
                ),
                "wo_caption": (
                    "The given comic shows the same situation from two "
                    "opposite sides with contradictions. Write a short "
                    "explanation to illustrate the contradiction of the "
                    "two sides."
                ),
            },
            2: {
                "w_caption": (
                    "The literal caption of the comic is: {image_caption}\n"
                    "Analyze the provided image, which is divided into "
                    "two or more panels, each illustrating contrasting "
                    "views of the same scenario. Describe the elements "
                    "visible in each panel. Then concisely interpret how "
                    "these elements convey contrasting perspectives in "
                    "one or two sentences. Focus and only output the "
                    "contradiction."
                ),
                "wo_caption": (
                    "Analyze the provided image, which is divided into "
                    "two or more panels, each illustrating contrasting "
                    "views of the same scenario. Describe the elements "
                    "visible in each panel. Then concisely interpret how "
                    "these elements convey contrasting perspectives in "
                    "one or two sentences. Focus and only output the "
                    "contradiction."
                ),
            },
            3: {
                "w_caption": (
                    "The literal caption of the comic is: {image_caption}\n"
                    "Given an image with two or more panels showing a "
                    "contrast relationship, describe the elements visible "
                    "in each panel and concisely interpret the "
                    "contradiction in one or two sentences."
                ),
                "wo_caption": (
                    "Given an image with two or more panels showing a "
                    "contrast relationship, describe the elements visible "
                    "in each panel and concisely interpret the "
                    "contradiction in one or two sentences."
                ),
            },
        },
        "moral_mcq": {
            1: {
                "w_caption": (
                    "The given comic shows the same situation from two "
                    "opposite sides with contradictions.\n"
                    "The literal caption of the comic is: {image_caption}\n"
                    "Which of the following options best represents the "
                    "underlying Symbolism of the comic?\n"
                    "{options}\n\nJust output the choice."
                ),
                "wo_caption": (
                    "The given comic shows the same situation from two "
                    "opposite sides with contradictions. Which of the "
                    "following options best represents the underlying "
                    "Symbolism of the comic?\n"
                    "{options}\n\nJust output the choice."
                ),
            },
            2: {
                "w_caption": (
                    "You are presented with an image divided into panels, "
                    "each illustrating contrasting views of the same "
                    "scenario.\n"
                    "The literal caption of the comic is: {image_caption}\n"
                    "Which of the following options best represents the "
                    "Symbolism of the image provided?\n"
                    "{options}\n\nSelect the correct option by typing the "
                    "corresponding letter (A, B, C, or D)."
                ),
                "wo_caption": (
                    "You are presented with an image divided into panels, "
                    "each illustrating contrasting views of the same "
                    "scenario. Which of the following options best "
                    "represents the Symbolism of the image provided?\n"
                    "{options}\n\nSelect the correct option by typing the "
                    "corresponding letter (A, B, C, or D)."
                ),
            },
            3: {
                "w_caption": (
                    "The literal caption of the comic is: {image_caption}\n"
                    "Given an image with two or more panels showing "
                    "contrast, select the best option representing the "
                    "deep semantic of the image.\n"
                    "{options}\n\nJust output the correct option as "
                    "(A, B, C, or D), no more explanation."
                ),
                "wo_caption": (
                    "Given an image with two or more panels showing "
                    "contrast, select the best option representing the "
                    "deep semantic of the image.\n"
                    "{options}\n\nJust output the correct option as "
                    "(A, B, C, or D), no more explanation."
                ),
            },
        },
        "title_mcq": {
            1: {
                "w_caption": (
                    "The given comic presents the same situation from "
                    "two opposing perspectives, highlighting "
                    "contradictions.\n"
                    "The literal caption of the comic is: {image_caption}\n"
                    "Which of the following titles is most suitable for "
                    "the comic?\n"
                    "{options}\n\nOutput only the selected choice."
                ),
                "wo_caption": (
                    "The given comic presents the same situation from "
                    "two opposing perspectives, highlighting "
                    "contradictions. Which of the following titles is "
                    "most suitable for the comic?\n"
                    "{options}\n\nOutput only the selected choice."
                ),
            },
            2: {
                "w_caption": (
                    "You are presented with an image divided into two or "
                    "more panels, each depicting contrasting perspectives "
                    "of the same scenario.\n"
                    "The literal caption of the comic is: {image_caption}\n"
                    "Which of the following title options best represents "
                    "the given image?\n"
                    "{options}\n\nSelect the correct option by typing the "
                    "corresponding letter (A, B, C, or D)."
                ),
                "wo_caption": (
                    "You are presented with an image divided into two or "
                    "more panels, each depicting contrasting perspectives "
                    "of the same scenario. Which of the following title "
                    "options best represents the given image?\n"
                    "{options}\n\nSelect the correct option by typing the "
                    "corresponding letter (A, B, C, or D)."
                ),
            },
            3: {
                "w_caption": (
                    "The literal caption of the comic is: {image_caption}\n"
                    "Given an image divided into two or more panels, a "
                    "contrast relationship exists between the panels. "
                    "Identify the best title from the following options "
                    "that represents the image.\n"
                    "{options}\n\nOutput only the corresponding letter "
                    "(A, B, C, or D) without any additional explanation."
                ),
                "wo_caption": (
                    "Given an image divided into two or more panels, a "
                    "contrast relationship exists between the panels. "
                    "Identify the best title from the following options "
                    "that represents the image.\n"
                    "{options}\n\nOutput only the corresponding letter "
                    "(A, B, C, or D) without any additional explanation."
                ),
            },
        },
    }

    def __init__(self, task: str = "moral_mcq", use_caption: bool = False,
                 prompt_set: int = 1):
        """
        Args:
            task: One of "description", "contradiction", "moral_mcq", "title_mcq"
            use_caption: If True, include oracle caption in prompt (w_caption variant)
            prompt_set: Prompt variant 1-3 from Table 7 (paper averages all three)
        """
        super().__init__()
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}, got '{task}'")
        if prompt_set not in (1, 2, 3):
            raise ValueError(f"prompt_set must be 1, 2, or 3, got {prompt_set}")
        self.task = task
        self.use_caption = use_caption
        self.prompt_set = prompt_set

    @staticmethod
    def _gdrive_direct_url(view_url: str) -> str:
        """Convert Google Drive view URL to direct download URL."""
        match = re.search(r"/d/([^/]+)/", view_url)
        if match:
            return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
        return view_url

    def _download_image(self, view_url: str, image_file: str,
                        images_dir: str) -> str:
        """Download image from Google Drive if not cached."""
        filepath = os.path.join(images_dir, image_file)
        if not os.path.exists(filepath):
            direct_url = self._gdrive_direct_url(view_url)
            req = urllib.request.Request(
                direct_url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req) as resp:
                with open(filepath, "wb") as f:
                    f.write(resp.read())
        return filepath

    def _build_prompt(self, item: dict) -> str:
        """Build text prompt for the given task, prompt set, and caption setting."""
        variant = "w_caption" if self.use_caption else "wo_caption"
        # description task only has wo_caption variant
        if self.task == "description":
            variant = "wo_caption"
        template = self.PROMPTS[self.task][self.prompt_set][variant]

        kwargs = {}
        if "{image_caption}" in template:
            kwargs["image_caption"] = item["caption"]
        if "{options}" in template:
            field = "moral_mcq" if self.task == "moral_mcq" else "title_mcq"
            kwargs["options"] = item[field]

        return template.format(**kwargs) if kwargs else template

    def get_instances(self, output_path: str):
        dataset = load_dataset(
            "zhehuderek/YESBUT_Benchmark_V2", split="train"
        )

        images_dir = os.path.join(output_path, "yesbut_v2_images")
        os.makedirs(images_dir, exist_ok=True)

        instances = []
        for idx, item in enumerate(dataset):
            # Download image
            image_path = self._download_image(
                item["url"], item["image_file"], images_dir
            )

            # Build text prompt
            prompt_text = self._build_prompt(item)

            # Create multimodal content
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="image/jpeg",
                    location=image_path,
                ),
                MediaObject(
                    content_type="text/plain",
                    text=f"\n{prompt_text}",
                ),
            ])

            # Build references based on task type
            if self.task in ("moral_mcq", "title_mcq"):
                answer_field = f"{self.task}_answer"
                correct_letter = item[answer_field]
                references = [
                    Reference(
                        Output(text=letter),
                        tags=[CORRECT_TAG] if letter == correct_letter else [],
                    )
                    for letter in ["A", "B", "C", "D"]
                ]
            elif self.task == "description":
                references = [
                    Reference(
                        Output(text=item["description"]),
                        tags=[CORRECT_TAG],
                    )
                ]
            elif self.task == "contradiction":
                references = [
                    Reference(
                        Output(text=item["contradiction"]),
                        tags=[CORRECT_TAG],
                    )
                ]

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT,
                id=f"yesbut_v2_{item['image_file']}_{self.task}",
            ))

        return instances
