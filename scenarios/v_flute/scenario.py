"""
HELM Scenario: V-FLUTE (Visual Figurative Language Understanding
                        with Textual Explanations)

Paper: https://arxiv.org/abs/2405.01474 (NAACL 2024 FigLang Workshop)
Code: https://github.com/asaakyan/V-FLUTE
Dataset: https://huggingface.co/datasets/ColumbiaNLP/V-FLUTE (gated, requires HF login)

V-FLUTE frames figurative language understanding as explainable visual
entailment: given an image (premise) and a textual claim (hypothesis),
the model must predict entailment/contradiction and justify its answer.

6,027 instances across 5 figurative phenomena: metaphors, similes,
idioms, sarcasm, and humor.

Two subsets:
  - classification: Binary entailment/contradiction (exact_match)
    723 test instances
  - explanation: Generate textual explanation for the relationship (open_ended)
    723 test instances

Prompt source: Paper Section 4 / Appendix (21 paraphrased prompt variants).
  Representative prompt used here; paper notes models received instruction
  to predict label and provide explanation.

Fields used: image, claim, label, explanation, phenomenon
Fields skipped: source_dataset (metadata only)

Note: Dataset is gated on HuggingFace. Requires HF_TOKEN with access granted.
      Images arrive as PIL objects and are saved to output_path for MediaObject.
"""

import os
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT, VALID_SPLIT, TRAIN_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class VFluteScenario(Scenario):
    name = "v_flute"
    description = "ColumbiaNLP/V-FLUTE"
    tags = ["creativity", "multimodal", "vision", "figurative_language"]

    SUBSETS = ["classification", "explanation"]

    def __init__(self, subset: str = "classification"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"subset must be one of {self.SUBSETS}, got '{subset}'")
        self.subset = subset

    def _save_image(self, pil_image, images_dir: str, idx: int) -> str:
        """Save a PIL image to disk and return the file path."""
        filepath = os.path.join(images_dir, f"{idx}.jpg")
        if not os.path.exists(filepath):
            pil_image.save(filepath, "JPEG")
        return filepath

    def get_instances(self, output_path: str):
        images_dir = os.path.join(output_path, "v_flute_images")
        os.makedirs(images_dir, exist_ok=True)

        split_map = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}
        instances = []

        for split_name, helm_split in split_map.items():
            dataset = load_dataset("ColumbiaNLP/V-FLUTE", split=split_name)

            for idx, item in enumerate(dataset):
                image_path = self._save_image(
                    item["image"], images_dir, hash(f"{split_name}_{idx}")
                )

                if self.subset == "classification":
                    prompt_text = (
                        f"Look at the image and consider the following claim:\n"
                        f"\"{item['claim']}\"\n\n"
                        f"Does the image entail or contradict this claim? "
                        f"Answer with: Entailment or Contradiction."
                    )

                    correct_label = item["label"]
                    references = [
                        Reference(
                            Output(text="Entailment"),
                            tags=[CORRECT_TAG] if correct_label == "Entailment" else [],
                        ),
                        Reference(
                            Output(text="Contradiction"),
                            tags=[CORRECT_TAG] if correct_label == "Contradiction" else [],
                        ),
                    ]

                else:  # explanation
                    prompt_text = (
                        f"Look at the image and consider the following claim:\n"
                        f"\"{item['claim']}\"\n\n"
                        f"The relationship between the image and claim is: "
                        f"{item['label']}.\n"
                        f"Explain why the image {item['label'].lower()}s "
                        f"this claim. Focus on the figurative language used."
                    )

                    references = [
                        Reference(
                            Output(text=item["explanation"]),
                            tags=[CORRECT_TAG],
                        )
                    ]

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

                instances.append(Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=helm_split,
                    id=f"v_flute_{split_name}_{idx}",
                ))

        return instances
