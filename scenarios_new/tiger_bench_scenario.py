"""
HELM Scenario: TIGeR-Bench (Text-to-Image Generation vs. Retrieval)

Paper: Unified Text-to-Image Generation and Retrieval
       https://arxiv.org/abs/2406.05814
Code:  https://github.com/LgQu/TIGeR
Data:  https://huggingface.co/datasets/leigangqu/TIGeR-Bench

TIGeR-Bench evaluates whether an AI system should generate a new image
or retrieve an existing one for a given text query. It spans 8 domains:

  Creative (label = Generate):
    - pick_a_pic  (2,500): Generative AI prompts (Stable Diffusion/Midjourney style)
    - whoops      (  500): Counterfactual/compositionally unusual scene descriptions

  Knowledge-intensive (label = Retrieve):
    - food2k           (500): Food/dish names
    - google_landmark  (500): Famous landmark names
    - inaturalist      (500): Wildlife and plant species names
    - logo2k           (500): Brand logo descriptions ("the logo of <Brand>")
    - visual_news      (500): News headline fragments
    - wit              (500): Wikipedia article/entity titles

Subsets:
  "all"        — all 6,000 examples (default)
  "creative"   — pick_a_pic + whoops (3,000 examples)
  "knowledge"  — the six knowledge-intensive splits (3,000 examples)

Prompt format:
  No explicit prompt specified in the paper (which evaluates full generation/
  retrieval systems, not LLMs doing text classification). Standard binary
  classification format used here.

  Text: {text}

  Should an AI system generate a new image or retrieve an existing image
  to best satisfy the above request?
  A. Generate (create a new image)
  B. Retrieve (find an existing image)

Fields used: text (query), split name (derives Generate/Retrieve label)
Fields skipped: image (reference image; not needed for text-only task)

Evaluation: exact_match (A or B)
"""

from typing import List
from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

_CREATIVE_SPLITS = {"pick_a_pic", "whoops"}
_KNOWLEDGE_SPLITS = {"food2k", "google_landmark", "inaturalist", "logo2k", "visual_news", "wit"}
_ALL_SPLITS = _CREATIVE_SPLITS | _KNOWLEDGE_SPLITS

_PROMPT_TEMPLATE = (
    "Text: {text}\n\n"
    "Should an AI system generate a new image or retrieve an existing image "
    "to best satisfy the above request?\n"
    "A. Generate (create a new image)\n"
    "B. Retrieve (find an existing image)"
)


class TIGeRBenchScenario(Scenario):
    """
    TIGeR-Bench: binary classification of text queries as requiring image
    generation (creative domain) or image retrieval (knowledge-intensive domain).

    subset="all"        — all 6,000 examples across 8 domains
    subset="creative"   — pick_a_pic + whoops (3,000 examples, label=Generate)
    subset="knowledge"  — 6 knowledge-intensive splits (3,000 examples, label=Retrieve)
    """

    name = "tiger_bench"
    description = "leigangqu/TIGeR-Bench"
    tags = ["creativity", "generate_vs_retrieve", "text_to_image", "classification"]

    SUBSETS = ["all", "creative", "knowledge"]

    def __init__(self, subset: str = "all"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"subset must be one of {self.SUBSETS}, got '{subset}'")
        self.subset = subset

    def _splits_for_subset(self) -> set:
        if self.subset == "creative":
            return _CREATIVE_SPLITS
        if self.subset == "knowledge":
            return _KNOWLEDGE_SPLITS
        return _ALL_SPLITS

    def get_instances(self, output_path: str) -> List[Instance]:
        instances = []

        for split_name in sorted(self._splits_for_subset()):
            dataset = load_dataset("leigangqu/TIGeR-Bench", split=split_name)
            correct_answer = "A" if split_name in _CREATIVE_SPLITS else "B"

            for item in dataset:
                prompt = _PROMPT_TEMPLATE.format(text=item["text"])

                references = [
                    Reference(Output(text="A"), tags=[CORRECT_TAG] if correct_answer == "A" else []),
                    Reference(Output(text="B"), tags=[CORRECT_TAG] if correct_answer == "B" else []),
                ]

                instances.append(Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    extra_data={"source_split": split_name},
                ))

        return instances
