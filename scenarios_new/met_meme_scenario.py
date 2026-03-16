"""
HELM Scenario: MET-Meme

Paper: "MET-Meme: A Multimodal Meme Dataset Rich in Metaphors"
       https://dl.acm.org/doi/10.1145/3477495.3532019 (SIGIR 2022)
Code: https://github.com/liaolianfoka/MET-Meme-A-Multi-modal-Meme-Dataset-Rich-in-Metaphors

MET-Meme is a multimodal meme understanding benchmark with 10,039 text-image
pairs (6,045 Chinese + 3,994 English) annotated for metaphor occurrence,
metaphor category, sentiment, communicative intention, and offensiveness.

Five classification subsets:
  - metaphor_occurrence: Binary — does the meme contain a visual metaphor?
    (2,007 test instances)
  - metaphor_category: 3-way — text dominant / image dominant / complementary
    (688 test instances, metaphor-positive subset only)
  - sentiment: 7-way — happiness / love / anger / sorrow / fear / hate / surprise
  - intention: 5-way — interactive / expressive / entertaining / offensive / other
  - offensiveness: 4-way — non-offensive / slightly / moderately / very

Prompt format: Standard MC classification (no LLM prompts in original paper;
  pre-LLM era classifier benchmark using BERT + VGG16/ResNet50 baselines).

  [Meme image]
  The text on this meme reads: "{text}"
  Question: {task-specific question}
  A) option_a  B) option_b  ...
  Answer:

Fields used: image (PIL), text, metaphor occurrence, metaphor category,
             sentiment category, intention detection, offensiveness detection
Fields skipped: sentiment degree, source domain, target domain,
                target modality, source modality, images_name

Data quirks:
  - 1 instance has Chinese label "表达情感" in intention detection (mapped to
    Expressive, equivalent to "2(expressive)")
  - Labels use format "N(description)" e.g. "1(happiness)" — cleaned to
    display names in prompts
"""

import os
from typing import List

from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject


class METMemeScenario(Scenario):
    name = "met_meme"
    description = "Anthony3456347095/MET-Meme"
    tags = ["creativity", "metaphor", "multimodal", "classification"]

    SUBSETS = [
        "metaphor_occurrence",
        "metaphor_category",
        "sentiment",
        "intention",
        "offensiveness",
    ]

    TASK_CONFIG = {
        "metaphor_occurrence": {
            "question": "Does this meme contain a visual metaphor?",
            "options": ["Yes", "No"],
            "label_field": "metaphor occurrence",
            "label_map": {"1": "Yes", "0": "No"},
        },
        "metaphor_category": {
            "question": "What type of metaphor is used in this meme?",
            "options": ["Text dominant", "Image dominant", "Complementary"],
            "label_field": "metaphor category",
            "label_map": {
                "text dominant": "Text dominant",
                "image dominant": "Image dominant",
                "complementary": "Complementary",
            },
        },
        "sentiment": {
            "question": "What sentiment is expressed in this meme?",
            "options": [
                "Happiness",
                "Love",
                "Anger",
                "Sorrow",
                "Fear",
                "Hate",
                "Surprise",
            ],
            "label_field": "sentiment category",
            "label_map": {
                "1(happiness)": "Happiness",
                "2(love)": "Love",
                "3(anger)": "Anger",
                "4(sorrow)": "Sorrow",
                "5(fear)": "Fear",
                "6(hate)": "Hate",
                "7(surprise)": "Surprise",
            },
        },
        "intention": {
            "question": "What is the communicative intention of this meme?",
            "options": [
                "Interactive",
                "Expressive",
                "Entertaining",
                "Offensive",
                "Other",
            ],
            "label_field": "intention detection",
            "label_map": {
                "1(interactive)": "Interactive",
                "2(expressive)": "Expressive",
                "3(entertaining)": "Entertaining",
                "4(offensive)": "Offensive",
                "5(other)": "Other",
                "\u8868\u8fbe\u60c5\u611f": "Expressive",
            },
        },
        "offensiveness": {
            "question": "How offensive is this meme?",
            "options": [
                "Non-offensive",
                "Slightly offensive",
                "Moderately offensive",
                "Very offensive",
            ],
            "label_field": "offensiveness detection",
            "label_map": {
                "0(non-offensive)": "Non-offensive",
                "1(slightly)": "Slightly offensive",
                "2(moderately)": "Moderately offensive",
                "3(very)": "Very offensive",
            },
        },
    }

    def __init__(self, subset: str = "metaphor_occurrence"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(
                f"Unknown subset '{subset}'. Choose from: {self.SUBSETS}"
            )
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("Anthony3456347095/MET-Meme", split="test")

        config = self.TASK_CONFIG[self.subset]
        question = config["question"]
        options = config["options"]
        label_field = config["label_field"]
        label_map = config["label_map"]

        image_dir = os.path.join(output_path, "met_meme_images")
        os.makedirs(image_dir, exist_ok=True)

        instances = []
        for idx, item in enumerate(dataset):
            # For metaphor_category, skip non-metaphor memes
            if (
                self.subset == "metaphor_category"
                and item["metaphor occurrence"] != "1"
            ):
                continue

            raw_label = item[label_field]
            answer = label_map.get(raw_label)
            if answer is None:
                continue

            # Save PIL image to file
            image_path = os.path.join(image_dir, f"{idx}.png")
            if not os.path.exists(image_path):
                item["image"].save(image_path)

            # Build prompt
            meme_text = item["text"]
            choices_str = "\n".join(
                f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)
            )

            multimedia_content = MultimediaObject(
                [
                    MediaObject(
                        content_type="image/png",
                        location=image_path,
                    ),
                    MediaObject(
                        content_type="text/plain",
                        text=(
                            f'The text on this meme reads: "{meme_text}"\n\n'
                            f"Question: {question}\n\n"
                            f"{choices_str}\n\n"
                            f"Answer:"
                        ),
                    ),
                ]
            )

            # Build references
            correct_idx = options.index(answer)
            references = []
            for i in range(len(options)):
                letter = chr(65 + i)
                tags = [CORRECT_TAG] if i == correct_idx else []
                references.append(Reference(Output(text=letter), tags=tags))

            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"{self.subset}_{idx}",
                )
            )

        return instances
