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
  - moral_mcq: Select underlying philosophy from 4 options (exact_match)
  - title_mcq: Select best title from 4 options (exact_match)

Each task supports two variants:
  - w_caption: Image + oracle caption provided as context
  - wo_caption: Image only (default, harder)

Prompt source: Prompts.sh in GitHub repo. Uses Set 1 prompts (paper's primary).
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

    # Prompts Set 1 from Prompts.sh (paper's primary prompt set)
    PROMPTS = {
        "description": {
            "wo_caption": (
                "The given comic shows the same situation from two opposite "
                "sides with contradictions. Write a one-paragraph literal "
                "description to describe the narrative of the comic."
            ),
        },
        "contradiction": {
            "w_caption": (
                "The given comic shows the same situation from two opposite "
                "sides with contradictions.\n"
                "The literal caption of the comic is: {image_caption}\n"
                "Write a short explanation to illustrate the contradiction "
                "of the two sides."
            ),
            "wo_caption": (
                "The given comic shows the same situation from two opposite "
                "sides with contradictions. Write a short explanation to "
                "illustrate the contradiction of the two sides."
            ),
        },
        "moral_mcq": {
            "w_caption": (
                "The given comic shows the same situation from two opposite "
                "sides with contradictions.\n"
                "The literal caption of the comic is: {image_caption}\n"
                "Which of the following options best represents the "
                "underlying philosophy of the comic?\n"
                "{options}\n\nJust output the choice:"
            ),
            "wo_caption": (
                "The given comic shows the same situation from two opposite "
                "sides with contradictions.\n"
                "Which of the following options best represents the "
                "underlying philosophy of the comic?\n"
                "{options}\n\nJust output the choice:"
            ),
        },
        "title_mcq": {
            "w_caption": (
                "The given comic shows the same situation from two opposite "
                "sides with contradictions.\n"
                "The literal caption of the comic is: {image_caption}\n"
                "Which of the following titles are the most suitable for "
                "the comic?\n"
                "{options}\n\nJust output the choice:"
            ),
            "wo_caption": (
                "The given comic shows the same situation from two opposite "
                "sides with contradictions.\n"
                "Which of the following titles are the most suitable for "
                "the comic?\n"
                "{options}\n\nJust output the choice:"
            ),
        },
    }

    def __init__(self, task: str = "moral_mcq", use_caption: bool = False):
        """
        Args:
            task: One of "description", "contradiction", "moral_mcq", "title_mcq"
            use_caption: If True, include oracle caption in prompt (w_caption variant)
        """
        super().__init__()
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}, got '{task}'")
        self.task = task
        self.use_caption = use_caption

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
        """Build text prompt for the given task and caption setting."""
        variant = "w_caption" if self.use_caption else "wo_caption"
        # description task only has wo_caption variant
        if self.task == "description":
            variant = "wo_caption"
        template = self.PROMPTS[self.task][variant]

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
