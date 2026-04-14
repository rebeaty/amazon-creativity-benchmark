"""
HELM Scenario: CLEF JOKER 2025 Task 2 — English to French Wordplay Translation

Overview paper: "Overview of the CLEF 2025 JOKER Task 2: Wordplay"
  CEUR-WS Vol-4038, paper_219 (2025)
  https://ceur-ws.org/Vol-4038/paper_219.pdf

Participant paper: "Pun Intended: Multi-Agent Translation of Wordplay with
  Contrastive Learning and Phonetic-Semantic Embeddings for CLEF JOKER 2025 Task 2"
  arXiv:2507.06506 — https://arxiv.org/abs/2507.06506

Project: http://www.joker-project.com/2025/

Task: Given an English pun sentence, produce a French equivalent that preserves the
wordplay (double meaning / humorous effect). Literal word-for-word translation is
penalised; the ideal output reconstructs the joke's mechanism in French using
homophonic or polysemous French words.

Dataset: CLEF JOKER 2025 Task 2 training corpus
  - 1,405 unique English pun sentences
  - 5,838 professional French translations (1–29 per English pun)
  - JSON format: list of objects with fields id_en, en, fr
    - id_en : unique identifier (e.g. "pun_001")
    - en    : English pun sentence
    - fr    : French translation (string); multiple objects may share the same en

  Dataset access: gated behind CLEF 2025 registration (Codabench platform).
  Register at http://www.joker-project.com/2025/ and place the downloaded training
  file at: <output_path>/joker_2025_task2_train.json

  The test set French translations are NOT released by the organisers.
  This scenario therefore uses the training data evaluated with TEST_SPLIT.

Prompt format (Standard creative-translation prompt — no official wording specified):
  The paper (arXiv:2507.06506) notes that participants deliberately avoided the word
  "translate" and instructed models to "produce a pun where both meanings are obvious
  and funny." The prompt below follows that guidance.

  You are a creative wordplay specialist. Below is an English pun. Write a French pun
  that captures the same humour and double meaning. The French version should work as
  its own wordplay — do not simply translate word-for-word.

  English: {pun}

  French:

Fields used:   en (English pun input), fr (reference French translations)
Fields skipped: id_en (identifier only, not used as model input)

Evaluation: open_ended (BLEU-4, ROUGE-L, F1 via get_open_ended_generation_metric_specs)
  See metric_notes.md for BERTScore and human-evaluation details.

Split used: TEST_SPLIT (train data; official test references not publicly released)
"""

import json
import os
from collections import defaultdict
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)


class CLEFJoker2025Task2Scenario(Scenario):
    """
    CLEF JOKER 2025 Task 2: English-to-French wordplay translation.

    Each instance presents one English pun; the model must generate a French
    pun that reconstructs the double meaning. All professional human translations
    for that pun are provided as CORRECT references for BLEU/ROUGE scoring.
    """

    name = "clef_joker_2025_task2"
    description = "CLEF JOKER 2025 Task 2 — English to French wordplay translation"
    tags = ["creativity", "translation", "wordplay", "multilingual", "puns"]

    PROMPT_TEMPLATE = (
        "You are a creative wordplay specialist. Below is an English pun. "
        "Write a French pun that captures the same humour and double meaning. "
        "The French version should work as its own wordplay — do not simply "
        "translate word-for-word.\n\n"
        "English: {pun}\n\n"
        "French:"
    )

    DATA_FILENAME = "joker_2025_task2_train.json"

    # Directories to search for the data file (in priority order)
    _SEARCH_DIRS = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scenarios", "clef_joker_2025_task2"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "clef_joker_2025_task2"),
    ]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_file = os.path.join(output_path, self.DATA_FILENAME)

        # Search alternative locations if not found at the default path
        if not os.path.exists(data_file):
            for search_dir in self._SEARCH_DIRS:
                candidate = os.path.join(search_dir, self.DATA_FILENAME)
                if os.path.exists(candidate):
                    os.makedirs(output_path, exist_ok=True)
                    import shutil
                    shutil.copy2(candidate, data_file)
                    break

        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"CLEF JOKER 2025 Task 2 dataset not found at:\n  {data_file}\n\n"
                "The dataset is gated behind CLEF registration. Please:\n"
                "  1. Register at http://www.joker-project.com/2025/\n"
                "  2. Download the Task 2 training JSON from Codabench\n"
                f"  3. Place it at the path above as '{self.DATA_FILENAME}'\n"
                f"  Or place it in one of: {self._SEARCH_DIRS}"
            )

        with open(data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Group all French translations by English pun text.
        # The JSON may contain multiple objects for the same English pun
        # (one per French translation) or a single object with fr as a list.
        pun_to_refs: dict = defaultdict(list)
        for item in raw:
            en = item["en"].strip()
            fr = item["fr"]
            if isinstance(fr, list):
                pun_to_refs[en].extend([t.strip() for t in fr if t.strip()])
            elif isinstance(fr, str) and fr.strip():
                pun_to_refs[en].append(fr.strip())

        instances = []
        for en_pun, fr_translations in pun_to_refs.items():
            prompt = self.PROMPT_TEMPLATE.format(pun=en_pun)

            # All professional human translations are valid references
            references = [
                Reference(Output(text=fr), tags=[CORRECT_TAG])
                for fr in fr_translations
            ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
