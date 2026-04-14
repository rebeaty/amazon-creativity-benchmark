"""
HELM Scenario: Conditional Story Generation Evaluation (TheNextChapter)

Paper: The Next Chapter: A Study of Large Language Models in Storytelling
       INLG 2023
       https://arxiv.org/abs/2301.09790
Code:  https://github.com/ZhuohanX/TheNextChapter

Task: Given a story condition (opening sentence or creative prompt), generate
a coherent and engaging story continuation. Tests conditional narrative
creativity across three distinct storytelling styles.

Three subsets (via subset= parameter):
  roc  — ROCStories: 800 instances. Short everyday narratives (4-5 sentences).
          Conditions use [MALE]/[FEMALE]/[NEUTRAL] gender placeholders.
          e.g. "[MALE] was at the county fair ."
  wp   — WritingPrompts: 1,000 instances. Open-ended Reddit creative fiction
          prompts; most imaginative and varied conditions.
  cnn  — CNN/DailyMail: 600 instances. News article continuation; conditions
          are article opening paragraphs, stories are article bodies.

Data source: GeneratedStories/{subset}/sample_human.txt — human-written baseline
  stories from the paper, one per line in `condition|||story` format.
  These are human-authored continuations from the original source corpora,
  used as the gold reference for evaluation.

Note on the HumanEvaluation/ JSON files: these 20-item subsets include human
  ratings (fluency, coherence, relatedness, logicality, interestingness on 1-5
  scales) per model output. See annotator_notes.md for the judge prompt setup
  if LLM-as-judge evaluation is desired instead of BLEU/ROUGE.

Prompt format (no prompt specified in paper — paper passes conditions directly
  to models; standard instruction format used here):

  roc: Continue the following story in a few sentences:
       {condition}

  wp:  Write a short story based on the following prompt:
       {condition}

  cnn: Continue the following news article:
       {condition}

Prompt source: Standard instruction format (paper passes raw conditions to models)
Fields used: condition (input), story from sample_human.txt (gold reference)
Fields skipped: GeneratedStories/{model}.txt files (model-generated outputs),
  HumanEvaluation scores (pre-computed for other models, not new evaluations)

Evaluation: open_ended (BLEU, ROUGE against human-written gold story)
  For multi-dimension human-aligned scoring, see annotator_notes.md
"""

import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

_BASE_URL = (
    "https://raw.githubusercontent.com/ZhuohanX/TheNextChapter/master"
    "/GeneratedStories"
)

_SUBSETS = {
    "roc": {
        "url": f"{_BASE_URL}/roc/sample_human.txt",
        "instruction": "Continue the following story in a few sentences:",
    },
    "wp": {
        "url": f"{_BASE_URL}/wp/sample_human.txt",
        "instruction": "Write a short story based on the following prompt:",
    },
    "cnn": {
        "url": f"{_BASE_URL}/cnn/sample_human.txt",
        "instruction": "Continue the following news article:",
    },
}


class TheNextChapterScenario(Scenario):
    """
    Conditional story generation across three subsets.

    subset="roc"  — 800 ROCStories short narrative continuations
    subset="wp"   — 1,000 WritingPrompts open creative fiction
    subset="cnn"  — 600 CNN/DailyMail news article continuations
    """

    name = "thenextchapter"
    description = "ZhuohanX/TheNextChapter"
    tags = ["creativity", "story_generation", "conditional_generation", "open_ended"]

    SUBSETS = list(_SUBSETS.keys())

    def __init__(self, subset: str):
        super().__init__()
        assert subset in self.SUBSETS, (
            f"subset must be one of {self.SUBSETS}, got '{subset}'"
        )
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        os.makedirs(output_path, exist_ok=True)
        config = _SUBSETS[self.subset]

        local_path = os.path.join(output_path, f"{self.subset}_human.txt")
        if not os.path.exists(local_path):
            urllib.request.urlretrieve(config["url"], local_path)

        instruction = config["instruction"]
        instances = []

        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|||")
                if len(parts) != 2:
                    continue

                condition, story = parts[0].strip(), parts[1].strip()
                if not condition or not story:
                    continue

                prompt = f"{instruction}\n\n{condition}"

                instances.append(Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=story), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                ))

        return instances
