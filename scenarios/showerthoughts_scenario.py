"""
HELM Scenario: Showerthoughts Generation and Evaluation

Paper: https://aclanthology.org/2024.starsem-1.23/ (*SEM 2024)
      "Investigating Wit, Creativity, and Detectability of Large Language Models
       in Domain-Specific Writing Style Adaptation of Reddit's Showerthoughts"
Code: https://github.com/aiintelligentsystems/showerthoughts-dataset

Task: Generate creative, witty "Showerthoughts" - miniature epiphanies that highlight
the amusing/interesting within mundane everyday observations. At their best,
Showerthoughts are universally relatable and exhibit wit, creativity, and sometimes humor.

Example Showerthoughts (from Reddit r/Showerthoughts):
  - "When you're a kid, you don't realize you're also watching your mom and dad grow up."
  - "Toilet paper is the only thing we willingly throw away after it touches our butt."
  - "People were probably way more confident before mirrors were invented"

Dataset: 411,189 Showerthoughts from Reddit (April 2020 - November 2022)
This scenario uses genuine human-written Showerthoughts as a reference corpus.

Evaluation: LLM-as-judge (see annotator_notes.md for details)
  The paper evaluates generated Showerthoughts on 5 dimensions via human ratings:
  1. Logical validity (makes a true/valid/logical statement)
  2. Creativity
  3. Humor (funny)
  4. Cleverness
  5. General score (overall quality)
  Each rated on 6-point Likert scale (1=lowest, 6=highest)

Prompt format:
  Please generate a Showerthought, which is inspired by the Reddit community
  r/Showerthoughts. Try to be clever, creative, and funny. The Showerthought should
  be relatable and connected to things that people might encounter during mundane tasks.

  Showerthought:

Prompt source: Paper Section 4.1 (ChatGPT zero-shot prompt), adapted for single-generation
format. Original prompt asked for "100 Showerthoughts" and to "vary the sentence structure
between the different sentences" - modified here for one-at-a-time generation
Fields used: title (Showerthought text), label (filtering for genuine examples)
Fields skipped: label=generated (AI-generated examples not used)
"""

import json
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class ShowerthoughtsScenario(Scenario):
    name = "showerthoughts"
    description = "aiintelligentsystems/showerthoughts-dataset"
    tags = ["creativity", "generation", "wit", "humor", "reddit"]

    # Test data contains both genuine and AI-generated examples (50/50 split)
    DATA_URL = "https://raw.githubusercontent.com/aiintelligentsystems/showerthoughts-dataset/main/generated/roberta_test_data_mixed.ndjson"

    def __init__(self, num_instances: int = 300):
        """
        Initialize Showerthoughts scenario.

        Args:
            num_instances: Number of instances to include (default 300).
                          Uses genuine Showerthoughts as reference examples.
        """
        super().__init__()
        self.num_instances = num_instances

    def get_instances(self, output_path: str):
        # Download the data file
        data_path = os.path.join(output_path, "showerthoughts_test.ndjson")
        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(self.DATA_URL, data_path)

        # Load genuine Showerthoughts only
        genuine_showerthoughts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if item['label'] == 'genuine':
                    genuine_showerthoughts.append(item['title'])

        # Limit to requested number of instances
        genuine_showerthoughts = genuine_showerthoughts[:self.num_instances]

        # Generation prompt (from paper Section 4.1, adapted for single-generation)
        prompt = (
            "Please generate a Showerthought, which is inspired by the Reddit community "
            "r/Showerthoughts. Try to be clever, creative, and funny. The Showerthought should "
            "be relatable and connected to things that people might encounter during mundane tasks.\n\n"
            "Showerthought:"
        )

        instances = []
        for showerthought in genuine_showerthoughts:
            # Each instance prompts for generation
            # The genuine Showerthought serves as a reference for style/quality
            # (Note: automatic metrics like BLEU won't be meaningful for unconditional generation;
            #  this task requires LLM-as-judge evaluation)
            instances.append(Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=showerthought), tags=[CORRECT_TAG])],
                split=TEST_SPLIT
            ))

        return instances
