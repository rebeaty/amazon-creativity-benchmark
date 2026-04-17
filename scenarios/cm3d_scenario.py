"""
HELM Scenario: CM3D (Chinese Multimodal Metaphor Mapping Dataset)

Paper: https://arxiv.org/abs/2501.02434
       "Towards Multimodal Metaphor Understanding: A Chinese Dataset
        and Model for Metaphor Mapping Identification"
Code: https://gitfront.io/r/GiveATry/nNCeJacwmNpG/C3MD/
Dataset: https://www.kaggle.com/datasets/ec70ddcbb3913b970f69e12fb2a0cc9254ccec95534542840d11b4fc373819ed

CM3D evaluates multimodal metaphor understanding in Chinese advertisements.
6,108 text-image pairs with annotations for metaphor occurrence, target
domain, and source domain.

Two subsets:
  - detection: Binary metaphor detection (exact_match)
    Is the advertisement metaphorical? (Yes/No)
  - mapping: Open-ended target/source domain identification (open_ended)
    For metaphorical ads, identify Target and Source domains.

Prompt source:
  - detection: Custom prompt for HELM (not from paper). The paper does not
    define a binary metaphor detection task; its Stage 1 asks to "Briefly
    describe the purpose of the text-image pair."
  - mapping: Simplified single-turn adaptation of the paper's two-stage
    pipeline (Section 4). Paper Stage 1 identifies the target domain, then
    Stage 2 uses the target to identify incongruent source elements. Here
    both are requested in one turn for HELM compatibility.

Fields used: Pic_id, MetaphorOccurrence, Target, Source, Type
Fields skipped: None

Note: Images on Kaggle (743 MB, requires Kaggle account). Annotations
      in data.json from gitfront repo. Chinese-language dataset.
      Train: 4,888 / Val: 610 / Test: 610
"""

import json
import os
import subprocess
import urllib.request
import zipfile
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT, VALID_SPLIT, TRAIN_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class CM3DScenario(Scenario):
    name = "cm3d"
    description = "C3MD/Chinese-Multimodal-Metaphor"
    tags = ["creativity", "multimodal", "vision", "figurative_language", "chinese"]

    SUBSETS = ["detection", "mapping"]
    REPO_URL = "https://gitfront.io/r/GiveATry/nNCeJacwmNpG/C3MD.git"
    KAGGLE_DATASET = "ec70ddcbb3913b970f69e12fb2a0cc9254ccec95534542840d11b4fc373819ed"

    # Split sizes from paper: train 4888, val 610, test 610
    TRAIN_SIZE = 4888
    VAL_SIZE = 610

    def __init__(self, subset: str = "detection"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"subset must be one of {self.SUBSETS}, got '{subset}'")
        self.subset = subset

    def _download_annotations(self, output_path: str) -> list:
        """Download data.json from gitfront and parse it."""
        json_path = os.path.join(output_path, "cm3d_data.json")
        if not os.path.exists(json_path):
            # Clone the repo to get raw data.json
            repo_dir = os.path.join(output_path, "cm3d_repo")
            if not os.path.exists(repo_dir):
                subprocess.run(
                    ["git", "clone", self.REPO_URL, repo_dir],
                    check=True, capture_output=True
                )
            src = os.path.join(repo_dir, "data.json")
            with open(src) as f:
                data = json.load(f)
            with open(json_path, "w") as f:
                json.dump(data, f)
        else:
            with open(json_path) as f:
                data = json.load(f)
        return data

    def _download_images(self, output_path: str) -> str:
        """Download images from Kaggle."""
        images_dir = os.path.join(output_path, "cm3d_images")
        if not os.path.exists(images_dir) or len(os.listdir(images_dir)) < 100:
            os.makedirs(images_dir, exist_ok=True)
            zip_path = os.path.join(output_path, "cm3d_images.zip")
            try:
                subprocess.run(
                    ["kaggle", "datasets", "download",
                     "-d", self.KAGGLE_DATASET,
                     "-p", output_path],
                    check=True, capture_output=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"WARNING: Kaggle download failed ({e}). "
                      "Place images manually in {images_dir}.")
                return images_dir
            # Find and extract the downloaded zip
            for f in os.listdir(output_path):
                if f.endswith(".zip") and "cm3d" not in f.lower():
                    zip_path = os.path.join(output_path, f)
                    break
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(images_dir)
        return images_dir

    def _assign_split(self, idx: int):
        """Assign train/val/test split based on paper sizes."""
        if idx < self.TRAIN_SIZE:
            return TRAIN_SPLIT
        elif idx < self.TRAIN_SIZE + self.VAL_SIZE:
            return VALID_SPLIT
        else:
            return TEST_SPLIT

    def get_instances(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)

        data = self._download_annotations(output_path)
        images_dir = self._download_images(output_path)

        instances = []
        for idx, (key, item) in enumerate(sorted(data.items(), key=lambda x: int(x[0]))):
            pic_id = item["Pic_id"]
            is_metaphor = item["MetaphorOccurrence"] == 1
            target = item.get("Target", "")
            source = item.get("Source", "")
            helm_split = self._assign_split(idx)

            # Skip NaN entries for mapping subset
            if self.subset == "mapping":
                if not is_metaphor:
                    continue
                if not target or not source or str(target) == "nan" or str(source) == "nan":
                    continue

            # Find image file
            image_path = os.path.join(images_dir, pic_id)
            if not os.path.exists(image_path):
                # Try with common extensions
                for ext in [".jpg", ".png", ".jpeg"]:
                    candidate = os.path.join(images_dir, pic_id + ext)
                    if os.path.exists(candidate):
                        image_path = candidate
                        break

            if self.subset == "detection":
                prompt_text = (
                    "观察以下广告图片和文字。\n"
                    "这个广告是否使用了隐喻手法？请回答：是 或 否。\n\n"
                    "(Observe the following advertisement image and text. "
                    "Does this advertisement use metaphor? Answer: Yes or No.)"
                )
                correct = "是" if is_metaphor else "否"
                references = [
                    Reference(Output(text="是"), tags=[CORRECT_TAG] if is_metaphor else []),
                    Reference(Output(text="否"), tags=[CORRECT_TAG] if not is_metaphor else []),
                ]
            else:  # mapping
                prompt_text = (
                    "这个广告使用了隐喻手法。\n"
                    "请识别目标域（广告宣传的主题）和源域（用于隐喻的元素）。\n"
                    "格式：目标域：[目标] 源域：[来源]\n\n"
                    "(This advertisement uses metaphor. Identify the target "
                    "domain (what is being advertised) and source domain "
                    "(the metaphorical element). "
                    "Format: Target: [target] Source: [source])"
                )
                references = [
                    Reference(
                        Output(text=f"目标域：{target} 源域：{source}"),
                        tags=[CORRECT_TAG],
                    )
                ]

            if os.path.exists(image_path):
                multimedia_content = MultimediaObject([
                    MediaObject(
                        content_type="image/jpeg",
                        location=image_path,
                    ),
                    MediaObject(
                        content_type="text/plain",
                        text=prompt_text,
                    ),
                ])
            else:
                # Image not available locally — use text-only input
                multimedia_content = MultimediaObject([
                    MediaObject(
                        content_type="text/plain",
                        text=prompt_text,
                    ),
                ])

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=helm_split,
                id=f"cm3d_{key}",
            ))

        return instances
