"""
HELM Scenario: Calligrapher Typography Benchmark

Paper: "Calligrapher: Freestyle Text Image Customization"
       https://arxiv.org/abs/2506.24123
Code: https://github.com/Calligrapher2025/Calligrapher

Calligrapher evaluates text-to-image diffusion models on style-centric
typography customization. Given a source image, a binary mask indicating
text regions, a style reference image, and a text prompt, models must
generate styled text that matches the reference's typographic style.

Two task variants:
  - self_reference: Source image doubles as the style reference; model
    edits text content while preserving the original style.
  - cross_reference: Style comes from a separate reference image,
    testing cross-image style transfer.

Test set: 100 text images at 512x512 with masks, prompts, and references.
Data: Calligrapher_bench_testing.zip from HuggingFace model repo.

Prompt source: Inference scripts in the repository
  (infer_calligrapher_self_custom.py, infer_calligrapher_cross_custom.py).
  Text prompt format: "The text is '{target_text}'."

Fields used: test{id}_source.png, test{id}_mask.png, test{id}_ref.png,
             self_bench.txt, cross_bench.txt
Fields skipped: calligrapher.bin (model weights)

Evaluation: Custom metrics (FID, CLIP, DINO, OCR accuracy).
See metric_notes.md for details.
"""

import os
import zipfile
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class CalligrapherScenario(Scenario):
    name = "calligrapher"
    description = "Calligrapher2025/Calligrapher"
    tags = ["creativity", "design", "typography", "multimodal", "diffusion"]

    HF_REPO = "Calligrapher2025/Calligrapher"
    ZIP_FILENAME = "Calligrapher_bench_testing.zip"

    TASK_PROMPT = "The text is '{text}'."

    def __init__(self, task: str = "self_reference"):
        """
        Args:
            task: "self_reference" or "cross_reference"
        """
        super().__init__()
        if task not in ("self_reference", "cross_reference"):
            raise ValueError(f"task must be 'self_reference' or 'cross_reference', got '{task}'")
        self.task = task

    def _download_data(self, output_path: str) -> str:
        """Download and extract the benchmark test set from HuggingFace."""
        data_dir = os.path.join(output_path, "Calligrapher_bench_testing")
        if os.path.exists(data_dir):
            return data_dir

        zip_path = os.path.join(output_path, self.ZIP_FILENAME)
        if not os.path.exists(zip_path):
            from huggingface_hub import hf_hub_download
            zip_path = hf_hub_download(
                repo_id=self.HF_REPO,
                filename=self.ZIP_FILENAME,
                local_dir=output_path,
            )

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)

        return data_dir

    def _parse_self_bench(self, bench_path: str) -> list:
        """Parse self_bench.txt: each line is 'id-text_prompt'."""
        entries = []
        with open(bench_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: id-prompt_text (split on first hyphen)
                parts = line.split("-", 1)
                if len(parts) == 2:
                    entries.append({
                        "id": parts[0].strip(),
                        "text": parts[1].strip(),
                    })
        return entries

    def _parse_cross_bench(self, bench_path: str) -> list:
        """Parse cross_bench.txt: each line is 'id-ref_ids-text_prompt'."""
        entries = []
        with open(bench_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: id-ref_ids-prompt_text
                parts = line.split("-", 2)
                if len(parts) == 3:
                    entries.append({
                        "id": parts[0].strip(),
                        "ref_ids": [r.strip() for r in parts[1].split(",")],
                        "text": parts[2].strip(),
                    })
        return entries

    def get_instances(self, output_path: str):
        data_dir = self._download_data(output_path)

        instances = []

        if self.task == "self_reference":
            bench_path = os.path.join(data_dir, "self_bench.txt")
            entries = self._parse_self_bench(bench_path)

            for entry in entries:
                img_id = entry["id"]
                source_path = os.path.join(data_dir, f"test{img_id}_source.png")
                mask_path = os.path.join(data_dir, f"test{img_id}_mask.png")

                if not os.path.exists(source_path):
                    continue

                prompt_text = self.TASK_PROMPT.format(text=entry["text"])

                # Self-reference: source image is also the style reference
                multimedia_content = MultimediaObject([
                    MediaObject(content_type="text/plain", text=prompt_text),
                    MediaObject(content_type="image/png", location=source_path),
                    MediaObject(content_type="image/png", location=mask_path),
                ])

                instances.append(Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"self_{img_id}",
                ))

        else:  # cross_reference
            bench_path = os.path.join(data_dir, "cross_bench.txt")
            entries = self._parse_cross_bench(bench_path)

            for entry in entries:
                img_id = entry["id"]
                source_path = os.path.join(data_dir, f"test{img_id}_source.png")
                mask_path = os.path.join(data_dir, f"test{img_id}_mask.png")

                if not os.path.exists(source_path):
                    continue

                prompt_text = self.TASK_PROMPT.format(text=entry["text"])

                for ref_id in entry["ref_ids"]:
                    ref_path = os.path.join(data_dir, f"test{ref_id}_ref.png")
                    if not os.path.exists(ref_path):
                        # Fallback: try source as ref
                        ref_path = os.path.join(data_dir, f"test{ref_id}_source.png")
                        if not os.path.exists(ref_path):
                            continue

                    # Cross-reference: separate style reference image
                    multimedia_content = MultimediaObject([
                        MediaObject(content_type="text/plain", text=prompt_text),
                        MediaObject(content_type="image/png", location=source_path),
                        MediaObject(content_type="image/png", location=mask_path),
                        MediaObject(content_type="image/png", location=ref_path),
                    ])

                    instances.append(Instance(
                        input=Input(multimedia_content=multimedia_content),
                        references=[],
                        split=TEST_SPLIT,
                        id=f"cross_{img_id}_ref{ref_id}",
                    ))

        return instances
