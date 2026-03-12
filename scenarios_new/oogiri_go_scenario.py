"""
HELM Scenario: Oogiri-GO Standard Evaluation

Paper: "Let's Think Outside the Box: Exploring Leap-of-Thought in Large
        Language Models with Creative Humor Generation"
       https://arxiv.org/abs/2312.02439 (CVPR 2024)
Code: https://github.com/sail-sg/CLoT

Oogiri-GO evaluates creative humor discrimination through the Japanese
Oogiri game format. Given a visual/textual prompt, models must identify
the funniest response from multiple candidates ranked by human preference
(star/like counts from Oogiri communities).

The paper's "Standard Evaluation" uses choice questions (mTn format).
The original uses BLIP2 captions, cross-image responses, and Qwen-14B
rewrites as distractors. This scenario constructs MC questions directly
from the dataset's multi-response groups: for each image/question with
multiple star-rated responses, the highest-starred response is the
correct answer and lower-starred responses are distractors. This yields
contextually harder distractors since all candidates are topical.

Task types:
  - I2T: Image-to-text humor (17,336 EN). Prompt is an image.
  - T2T: Text-to-text humor (6,433 EN). Prompt is text question + image.

Data: JSONL files + images.zip (1.3GB) from HuggingFace dataset repo.
130K+ total samples across EN/CN/JP with human star ratings.

Prompt source: Standard MC format. Paper's instruction templates
(Figure 6) are for training, not public evaluation prompts.

Fields used: text (response), question (T2T prompt), image (image ID),
             star (human preference score), type (I2T/T2T)
Fields skipped: None

Evaluation: exact_match on MC choice selection.
"""

import json
import os
import random
import zipfile
from collections import defaultdict

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class OogiriGOScenario(Scenario):
    name = "oogiri_go"
    description = "zhongshsh/CLoT-Oogiri-GO"
    tags = ["creativity", "humor", "multimodal"]

    HF_REPO = "zhongshsh/CLoT-Oogiri-GO"
    LANG_FILES = {"en": "en.jsonl", "cn": "cn.jsonl", "jp": "jp.jsonl"}

    TASK_PROMPT_I2T = (
        "This is an Oogiri game — a Japanese comedy format where players "
        "create witty, surprising responses to a given image prompt.\n\n"
        "Look at the image and select the funniest, most creative response.\n\n"
        "{choices}\n\nAnswer:"
    )

    TASK_PROMPT_T2T = (
        "This is an Oogiri game — a Japanese comedy format where players "
        "create witty, surprising responses to a given prompt.\n\n"
        "Prompt: {question}\n\n"
        "Select the funniest, most creative response.\n\n"
        "{choices}\n\nAnswer:"
    )

    def __init__(
        self,
        language: str = "en",
        task: str = "all",
        num_choices: int = 4,
        seed: int = 42,
    ):
        """
        Args:
            language: "en", "cn", or "jp"
            task: "i2t", "t2t", or "all"
            num_choices: Number of MC options (default 4, matching paper's 4T1)
            seed: Random seed for distractor sampling
        """
        super().__init__()
        if language not in self.LANG_FILES:
            raise ValueError(f"language must be one of {list(self.LANG_FILES)}, got '{language}'")
        if task not in ("i2t", "t2t", "all"):
            raise ValueError(f"task must be 'i2t', 't2t', or 'all', got '{task}'")
        if num_choices < 2:
            raise ValueError(f"num_choices must be >= 2, got {num_choices}")
        self.language = language
        self.task = task
        self.num_choices = num_choices
        self.seed = seed

    def _download_data(self, output_path: str):
        """Download JSONL and images.zip from HuggingFace."""
        from huggingface_hub import hf_hub_download

        jsonl_path = hf_hub_download(
            repo_id=self.HF_REPO,
            filename=self.LANG_FILES[self.language],
            repo_type="dataset",
            local_dir=output_path,
        )

        images_zip = hf_hub_download(
            repo_id=self.HF_REPO,
            filename="images.zip",
            repo_type="dataset",
            local_dir=output_path,
        )

        # Extract images if not already done
        images_dir = os.path.join(output_path, "images")
        if not os.path.exists(images_dir):
            with zipfile.ZipFile(images_zip, "r") as zf:
                zf.extractall(output_path)

        return jsonl_path, images_dir

    def _load_entries(self, jsonl_path: str):
        """Load and filter JSONL entries."""
        entries = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Must have star rating for MC construction
                if obj.get("star") is None:
                    continue
                # Filter by task type
                entry_type = obj.get("type", "").upper()
                if self.task == "i2t" and entry_type != "I2T":
                    continue
                if self.task == "t2t" and entry_type != "T2T":
                    continue
                entries.append(obj)
        return entries

    def _build_groups(self, entries):
        """Group entries by their prompt (image + optional question)."""
        groups = defaultdict(list)
        for entry in entries:
            img_id = entry.get("image", "")
            question = entry.get("question") or ""
            entry_type = entry.get("type", "")
            key = (img_id, question, entry_type)
            groups[key].append(entry)
        return groups

    def get_instances(self, output_path: str):
        jsonl_path, images_dir = self._download_data(output_path)
        entries = self._load_entries(jsonl_path)
        groups = self._build_groups(entries)

        rng = random.Random(self.seed)
        instances = []

        # Sort group keys for deterministic ordering
        sorted_keys = sorted(groups.keys())

        for key in sorted_keys:
            group_entries = groups[key]
            if len(group_entries) < self.num_choices:
                continue

            img_id, question, entry_type = key

            # Sort by star rating descending; break ties by text for stability
            group_entries.sort(key=lambda e: (-e["star"], e["text"]))

            # Correct answer: highest-starred response
            correct = group_entries[0]

            # Distractors: sample from the rest
            candidates = group_entries[1:]
            if len(candidates) > self.num_choices - 1:
                distractors = rng.sample(candidates, self.num_choices - 1)
            else:
                distractors = candidates

            # Shuffle all choices
            all_choices = [correct] + distractors
            rng.shuffle(all_choices)

            # Find correct index after shuffle
            correct_idx = next(
                i for i, c in enumerate(all_choices)
                if c["text"] == correct["text"]
            )

            # Format choices text
            letters = [chr(65 + i) for i in range(len(all_choices))]
            choices_text = "\n".join(
                f"{letters[i]}) {all_choices[i]['text']}"
                for i in range(len(all_choices))
            )

            # Build prompt
            if entry_type == "T2T" and question:
                prompt_text = self.TASK_PROMPT_T2T.format(
                    question=question, choices=choices_text
                )
            else:
                prompt_text = self.TASK_PROMPT_I2T.format(choices=choices_text)

            # Build multimodal content with image
            image_path = os.path.join(images_dir, f"{img_id}.jpg")
            media_objects = [
                MediaObject(content_type="image/jpeg", location=image_path),
                MediaObject(content_type="text/plain", text=prompt_text),
            ]
            multimedia_content = MultimediaObject(media_objects)

            # References: all choices, correct one tagged
            references = [
                Reference(
                    Output(text=letters[i]),
                    tags=[CORRECT_TAG] if i == correct_idx else [],
                )
                for i in range(len(all_choices))
            ]

            instance_id = f"{entry_type}_{img_id}"
            if question:
                # Truncate question for ID readability
                q_slug = question[:20].replace(" ", "_")
                instance_id = f"{entry_type}_{img_id}_{q_slug}"

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT,
                id=instance_id,
            ))

        return instances
