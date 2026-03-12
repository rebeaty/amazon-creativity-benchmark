"""
HELM Scenario: VGSG (Visually Grounded Story Generation)

Paper: "Summary of the Visually Grounded Story Generation Challenge"
       https://aclanthology.org/2024.inlg-genchal.3/ (INLG 2024)
Dataset: https://huggingface.co/datasets/tonyhong/vwp (Visual Writing Prompts v2)
Original paper: https://arxiv.org/abs/2301.08571 (TACL 2023)

Task: Given a sequence of 5-10 movie shot images and up to 5 named characters
(with reference portrait images), generate a coherent short story grounded in
the visual content.

Dataset: Visual Writing Prompts (VWP)
  - train: 11,778 stories
  - val: 849 stories (used here; test split has no ground truth)
  - Each example: 5-10 sequential movie shots + up to 5 characters
  - Images hosted at datasets.d2.mpi-inf.mpg.de

Prompt source: No specific prompt in paper; standard visual story generation
  instruction used. Crowdsource instructions from original VWP paper: writers
  were given image sequences and grounded characters to write stories.
Fields used: link0-link9 (scene images), char0-char4 (character names),
  char0_url-char4_url (character reference images), story (ground truth)
Fields skipped: text0-text9 (per-image captions, not used as input),
  sep_story, anonymised_story, img_id_list, bounding_box, imdb_id

Evaluation: open_ended (BLEU, METEOR, ROUGE-L, CIDEr)
  Paper also uses human evaluation: Coherence, Diversity, Grammaticality,
  Visual Grounding, Overall quality.
"""

import os
import urllib.request
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class VGSGScenario(Scenario):
    name = "vgsg"
    description = "tonyhong/vwp"
    tags = ["creativity", "multimodal", "vision", "narrative", "story_generation"]

    PROMPT_WITH_CHARS = (
        "You are given a sequence of images from a story, along with "
        "reference images of the main characters. Write a coherent short "
        "story based on these images. The story should follow the narrative "
        "depicted across the image sequence and feature the named characters."
    )

    PROMPT_NO_CHARS = (
        "You are given a sequence of images from a story. Write a coherent "
        "short story based on these images. The story should follow the "
        "narrative depicted across the image sequence."
    )

    def __init__(self, split: str = "val"):
        """
        Args:
            split: Dataset split to use. "val" (default, 849 examples with
                   ground truth) or "test" (586 examples, no ground truth).
        """
        super().__init__()
        if split not in ("val", "test"):
            raise ValueError(f"split must be 'val' or 'test', got '{split}'")
        self.split = split

    def _download_image(self, url: str, images_dir: str) -> str:
        """Download image from URL if not cached. Returns local path."""
        filename = url.split("/")[-2] + "_" + url.split("/")[-1]
        filepath = os.path.join(images_dir, filename)
        if not os.path.exists(filepath):
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(filepath, "wb") as f:
                    f.write(resp.read())
        return filepath

    def get_instances(self, output_path: str):
        dataset = load_dataset("tonyhong/vwp", split=self.split)

        images_dir = os.path.join(output_path, "vgsg_images")
        os.makedirs(images_dir, exist_ok=True)

        instances = []
        for idx, item in enumerate(dataset):
            media_objects = []

            # Collect character info
            characters = []
            for c in range(5):
                name = item[f"char{c}"]
                char_url = item[f"char{c}_url"]
                if isinstance(name, str) and name and name != "{}" and isinstance(char_url, str) and char_url and char_url != "{}":
                    characters.append((name, char_url))

            # Choose prompt based on whether characters exist
            if characters:
                media_objects.append(
                    MediaObject(
                        content_type="text/plain",
                        text=self.PROMPT_WITH_CHARS,
                    )
                )
                # Add character reference images with names
                media_objects.append(
                    MediaObject(
                        content_type="text/plain",
                        text="\n\nCharacters:",
                    )
                )
                for name, char_url in characters:
                    char_path = self._download_image(char_url, images_dir)
                    media_objects.append(
                        MediaObject(
                            content_type="text/plain",
                            text=f"\n{name}:",
                        )
                    )
                    media_objects.append(
                        MediaObject(
                            content_type="image/jpeg",
                            location=char_path,
                        )
                    )
            else:
                media_objects.append(
                    MediaObject(
                        content_type="text/plain",
                        text=self.PROMPT_NO_CHARS,
                    )
                )

            # Add scene images in sequence
            media_objects.append(
                MediaObject(
                    content_type="text/plain",
                    text="\n\nStory images (in order):",
                )
            )
            for i in range(10):
                link = item[f"link{i}"]
                if isinstance(link, str) and link:
                    img_path = self._download_image(link, images_dir)
                    media_objects.append(
                        MediaObject(
                            content_type="image/jpeg",
                            location=img_path,
                        )
                    )

            # Final instruction
            media_objects.append(
                MediaObject(
                    content_type="text/plain",
                    text="\n\nStory:",
                )
            )

            multimedia_content = MultimediaObject(media_objects)

            # Ground truth story as reference (None for test split)
            references = []
            if item["story"] is not None:
                references.append(
                    Reference(
                        Output(text=item["story"]),
                        tags=[CORRECT_TAG],
                    )
                )

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT,
                id=f"vgsg_{item['scene_full_id']}_{item.get('story_id', idx)}",
            ))

        return instances
