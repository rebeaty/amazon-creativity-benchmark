"""
HELM Scenario: TextLogo3K

Paper: "Aesthetic Text Logo Synthesis via Content-aware Layout Inferring"
       https://arxiv.org/abs/2204.02701 (CVPR 2022)
Code: https://github.com/yizhiwang96/TextLogoLayout

TextLogo3K evaluates spatial creativity in text logo layout design. Given
glyph images of individual characters and their text, models must predict
aesthetically pleasing bounding box coordinates for arranging them into a
logo on a 128x128 canvas.

The dataset contains 3,470 Chinese text logos from movie/TV/comic posters
on Tencent Video, with character-level bounding boxes and segmentation masks.
300 test samples with 2-11 characters each (mean 4.8).

Prompt source: No prompt specified in paper (the original model is a
neural network, not an LLM). Task prompt designed for LLM layout generation,
following the JSON coordinate format established by GLDesigner (arXiv
2411.11435), LayoutGPT (NeurIPS 2023), and PosterLLaVa (2024).

Fields used: elements.png (glyph images), coords_seg.npy (ground truth
             bounding boxes), len.npy (character count), logo_titles.txt
             (character text)
Fields skipped: char_embeds.npy, word_embeds.npy (word2vec embeddings for
                neural model), coords_det.npy (detection-based coords),
                seg.png, poster.jpg (source materials)

Evaluation: Custom metrics computed from predicted vs ground-truth
coordinates. See metric_notes.md for details.
"""

import json
import os
import urllib.request
import zipfile
import numpy as np
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class TextLogo3KScenario(Scenario):
    name = "textlogo3k"
    description = "yizhiwang96/TextLogoLayout"
    tags = ["creativity", "design", "layout", "multimodal"]

    DATA_URL = "https://drive.google.com/uc?export=download&id=1KKnUET6ec9eSLpr5TkGMCgKHaY7Z9_36&confirm=t"
    OUTPUT_SIZE = 128  # Canvas size in pixels

    TASK_PROMPT = (
        "You are an expert text logo designer. Given the glyph images of "
        "individual characters, arrange them into an aesthetically pleasing "
        "text logo layout on a square canvas.\n\n"
        "Characters to arrange: {title} ({num_chars} characters)\n\n"
        "For each character (indexed 0 to {max_idx}), predict a bounding box "
        "as [left, top, right, bottom] with values normalized to [0, 1] "
        "where (0,0) is the top-left corner and (1,1) is the bottom-right "
        "corner of the canvas.\n\n"
        "Design guidelines:\n"
        "- Create visual balance and harmony across the composition\n"
        "- Avoid overlapping characters\n"
        "- Use the full canvas space effectively\n"
        "- Consider the visual weight and shape of each glyph\n"
        "- Characters can vary in size for emphasis and hierarchy\n\n"
        "Output a JSON array of bounding boxes, one per character, in order:\n"
        '[{{"char": "X", "box": [left, top, right, bottom]}}, ...]'
    )

    def _download_data(self, output_path: str) -> str:
        """Download and extract TextLogo3K dataset from Google Drive."""
        data_dir = os.path.join(output_path, "TextLogo3K")
        if os.path.exists(os.path.join(data_dir, "test")):
            return data_dir

        zip_path = os.path.join(output_path, "TextLogo3K_v1.zip")
        if not os.path.exists(zip_path):
            try:
                import gdown
                gdown.download(
                    "https://drive.google.com/uc?id=1KKnUET6ec9eSLpr5TkGMCgKHaY7Z9_36",
                    zip_path, quiet=False
                )
            except ImportError:
                urllib.request.urlretrieve(self.DATA_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)

        return data_dir

    def _load_titles(self, data_dir: str) -> dict:
        """Load character text from logo_titles.txt."""
        titles = {}
        titles_path = os.path.join(data_dir, "logo_titles.txt")
        with open(titles_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    sample_id = parts[0]
                    chars = parts[3]
                    titles[sample_id] = chars
        return titles

    def get_instances(self, output_path: str):
        data_dir = self._download_data(output_path)
        titles = self._load_titles(data_dir)
        test_dir = os.path.join(data_dir, "test")
        samples = sorted(os.listdir(test_dir))

        instances = []
        for sample_id in samples:
            sample_path = os.path.join(test_dir, sample_id)

            # Load metadata
            text_len = int(np.load(
                os.path.join(sample_path, "len.npy")
            )[0])
            coords = np.load(
                os.path.join(sample_path, "coords_seg.npy")
            )

            # Get character text
            title = titles.get(sample_id, "?" * text_len)
            char_list = list(title)[:text_len]

            # Ground truth: normalized coordinates
            gt_boxes = []
            for i in range(text_len):
                box = (coords[i] / self.OUTPUT_SIZE).tolist()
                gt_boxes.append({
                    "char": char_list[i] if i < len(char_list) else "?",
                    "box": [round(v, 3) for v in box]
                })
            gt_json = json.dumps(gt_boxes, ensure_ascii=False)

            # Build prompt
            prompt_text = self.TASK_PROMPT.format(
                title=title,
                num_chars=text_len,
                max_idx=text_len - 1,
            )

            # Glyph image path
            elements_path = os.path.join(sample_path, "elements.png")

            # Multimodal: text prompt + glyph images
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="text/plain",
                    text=prompt_text,
                ),
                MediaObject(
                    content_type="image/png",
                    location=elements_path,
                ),
            ])

            # Reference: ground truth layout as JSON
            references = [
                Reference(
                    Output(text=gt_json),
                    tags=[CORRECT_TAG],
                )
            ]

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT,
                id=sample_id,
            ))

        return instances
