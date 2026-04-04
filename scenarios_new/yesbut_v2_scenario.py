"""
HELM Scenario: YESBUT V2

Paper: "When 'YES' Meets 'BUT': Can Large Models Comprehend Contradictory
       Humor Through Comparative Reasoning?"
       https://arxiv.org/abs/2503.23137
Code: https://github.com/Tuo-Liang/YESBUT_V2
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

Three prompt sets from Table 7 (paper averages across all three to reduce bias).
Prompt source: Prompts.sh from the V2 GitHub repo (Table 7 in paper).
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

    # Evaluation prompts from Prompts.sh in V2 repo — three prompt sets per task.
    # Paper averages results across all three to reduce prompt bias.
    # Prompts are reproduced verbatim (including original grammar).
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
                    "Give me a detailed literally description of the image."
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
                    "Analyze the provided image with the following "
                    "description: {image_caption}. Identify and concisely "
                    "describe the contradiction depicted in the image in "
                    "one or two sentences."
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
                    "Based on the following image's description: "
                    "{image_caption}. Give me the concise contradiction "
                    "depicted in the image in one or two sentences."
                ),
                "wo_caption": (
                    "Given an image, the image is divided into two or "
                    "more panels. There is the contrast relationship in "
                    "the image through panels. Describe the elements "
                    "visible in each panel. Give me the concise "
                    "interpretation how these panels convey contrasting "
                    "perspectives, which you only need to output the "
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
                    "underlying philosophy of the comic?\n"
                    "{philosophy_options}\n\nJust output the choice:"
                ),
                "wo_caption": (
                    "The given comic shows the same situation from two "
                    "opposite sides with contradictions.\n"
                    "Which of the following options best represents the "
                    "underlying philosophy of the comic?\n"
                    "{philosophy_options}\n\nJust output the choice:"
                ),
            },
            2: {
                "w_caption": (
                    "You are presented with an image with the following "
                    "description: {image_caption}. \n"
                    "Which of the following options best represents the "
                    "philosophy of the image provided? \n"
                    "{philosophy_options} \n"
                    "Select the correct option by typing the "
                    "corresponding letter (A, B, C, or D)."
                ),
                "wo_caption": (
                    "You are presented with an image, which is divided "
                    "into two or more panels, each illustrating "
                    "contrasting views of the same scenario. \n"
                    "Which of the following options best represents the "
                    "philosophy of the image provided? \n"
                    "{philosophy_options} \n"
                    "Select the correct option by typing the "
                    "corresponding letter (A, B, C, or D)."
                ),
            },
            3: {
                "w_caption": (
                    "Given an image with the following description: "
                    "{image_caption}. \n"
                    "Tell me the best option in the following options "
                    "who represents the deep semantic of the image? \n"
                    "{philosophy_options} \n"
                    "Just tell me the correct option by outputing "
                    "corresponding letter (A, B, C, or D), no more "
                    "explanation."
                ),
                "wo_caption": (
                    "Given an image, which has two or more panels. "
                    "There is contrast in these panels. \n"
                    "Tell me the best option in the following options "
                    "who represents the deep semantic of the image? \n"
                    "{philosophy_options} \n"
                    "Just tell me the correct option by outputing "
                    "corresponding letter (A, B, C, or D), no more "
                    "explanation."
                ),
            },
        },
        "title_mcq": {
            1: {
                "w_caption": (
                    "The given comic shows the same situation from two "
                    "opposite sides with contradictions.\n"
                    "The literal caption of the comic is: {image_caption}\n"
                    "Which of the following titles are the most suitable "
                    "for the comic?\n"
                    "{title_options}\n\nJust output the choice:"
                ),
                "wo_caption": (
                    "The given comic shows the same situation from two "
                    "opposite sides with contradictions.\n"
                    "Which of the following titles are the most suitable "
                    "for the comic?\n"
                    "{title_options}\n\nJust output the choice:"
                ),
            },
            2: {
                "w_caption": (
                    "You are presented with an image with the following "
                    "description: {image_caption}. \n"
                    "Which of the following title options best represents "
                    "the image provided? \n"
                    "{title_options} \n"
                    "Select the correct option by typing the "
                    "corresponding letter (A, B, C, or D)."
                ),
                "wo_caption": (
                    "You are presented with an image, which is divided "
                    "into two or more panels, each illustrating "
                    "contrasting views of the same scenario. \n"
                    "Which of the following title options best represents "
                    "the image provided? \n"
                    "{title_options} \n"
                    "Select the correct option by typing the "
                    "corresponding letter (A, B, C, or D)."
                ),
            },
            3: {
                "w_caption": (
                    "Given an image with the following description: "
                    "{image_caption}. \n"
                    "Tell me the best title in the following title "
                    "options who represents the image? \n"
                    "{title_options} \n"
                    " Just tell me the correct option by outputing "
                    "corresponding letter (A, B, C, or D), no more "
                    "explanation."
                ),
                "wo_caption": (
                    "Given an image, the image is divided into two or "
                    "more panels. There is the contrast relationship in "
                    "the image through panels. \n"
                    "Tell me the best title in the following title "
                    "options who represents the image? \n"
                    "{title_options} \n"
                    "Just tell me the correct option by outputing "
                    "corresponding letter (A, B, C, or D), no more "
                    "explanation."
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
            with urllib.request.urlopen(req, timeout=30) as resp:
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
        if "{philosophy_options}" in template:
            kwargs["philosophy_options"] = item["moral_mcq"]
        if "{title_options}" in template:
            kwargs["title_options"] = item["title_mcq"]

        return template.format(**kwargs) if kwargs else template

    def get_instances(self, output_path: str):
        dataset = load_dataset(
            "zhehuderek/YESBUT_Benchmark_V2", split="train"
        )

        images_dir = os.path.join(output_path, "yesbut_v2_images")
        os.makedirs(images_dir, exist_ok=True)

        instances = []
        for idx, item in enumerate(dataset):
            # Download image — skip on failure
            try:
                image_path = self._download_image(
                    item["url"], item["image_file"], images_dir
                )
            except Exception:
                continue

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
