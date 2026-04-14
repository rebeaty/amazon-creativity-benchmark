"""
HELM Scenario: Diverse-not-Short Creativity Evaluation Suite

Paper: "Diverse, not Short: Improving Length-Robustness of DPO-Based
        Diverse Creative Writing" (EMNLP 2025)
       arXiv: 2505.16245
Code: https://github.com/text-machine-lab/diverse-not-short

Six creativity evaluation tasks from psychometric and NLP literature:

  - RAT: Remote Associates Test (143 items, exact_match)
    Find a 4th word connecting 3 cue words. (Mednick, 1962)
  - AUT: Alternative Uses Test (15 items, open_ended)
    Generate 5 creative uses for everyday objects. (Guilford, 1967)
  - CWT: Creative Writing Task (7 items, open_ended)
    Write a 5-sentence story incorporating 3 given words.
  - DAT: Divergent Association Test (1 item, open_ended)
    Generate 10 maximally different words. (Olson et al., 2021)
  - C-DAT: Conditional Divergent Association Test (15 items, open_ended)
    Generate 5 words related to a given word but different from each other.
  - PGT: Persona Generation Task (1 item, open_ended)
    Generate a random persona with name, city, occupation.

Note: Existing scenarios cover DAT (dat_creative_writing) and C-DAT (cdat)
from other papers. This scenario uses distinct prompts and test sets from
the Diverse-not-Short evaluation framework.

Data: RAT items from rat_test_examples.tsv in the paper's GitHub repo.
      AUT/CWT/C-DAT stimuli hardcoded from system_prompts.py.
      DAT/PGT use single fixed prompts.

Prompt source: Exact prompts from system_prompts.py in the paper's codebase.

Fields used: Cue1, Cue2, Cue3, Answer (RAT); hardcoded stimuli (others)
Fields skipped: None

Evaluation:
  - RAT: exact_match (labeled answers)
  - Others: open_ended (DSI, entropy, TTR, lexical metrics).
    See metric_notes.md.
"""

import csv
import io
import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)

# Exact prompts from the paper's system_prompts.py

RAT_TASK_DESC = (
    "Task Description: For this task, you'll be shown 3 unrelated cue words, "
    "and you must find a 4th word that relates to all 3 words shown.\n"
    "The generation should be restricted only one word (the 4th word) and "
    'the format of the generation of the 4th word should be: "Solution: '
    '[WORD]". Do not generated anything else.'
)

AUT_TASK_DESC = (
    "Task Description: For this task, you'll be asked to come up with as many "
    "original and creative uses for objects as you can.\n"
    "The goal is to come up with creative ideas, which are ideas that strike "
    "people as clever, unusual, interesting, uncommon, humorous, innovative, "
    "or different.\n"
    "You must list each use with a number and a period. For example, "
    '"1. Use-1, 2. Use-2, 3. Use-3, etc.". You must provide exactly '
    "five (5) uses for each object."
)

CWT_TASK_DESC = (
    "Task Description: For this task, you will write a very short story. "
    "You will be given 3 words, and write a story that includes all 3 words. "
    "Your story should be about 5 sentences long.\n"
    "Use your imagination and be creative when writing your story. "
    "But, also be sure your story makes sense."
)

DAT_TASK_DESC = (
    "Task description: Please generate 10 words that are as different from "
    "each other as possible, in all meanings and uses of the words.\n"
    "Rules: Only single words in English. Only nouns (e.g., things, objects, "
    "concepts). No proper nouns (e.g., no specific people or places). No "
    "specialized vocabulary (e.g., no technical terms).\n"
    "Think of the words on your own (e.g., do not just look at objects in "
    "your surroundings). Make a list of these 10 words, without any "
    "repetition.\n"
    'You must list each word with a number and a period. For example, '
    '"1. word-1, 2. word-2, etc."'
)

CDAT_TASK_DESC = (
    "Task description: Please generate 5 words that are closely related to "
    "the given word.\n"
    "However, all 5 words must be as different from each other as possible, "
    "in all meanings and uses of the words.\n"
    "Rules:\n"
    "- Each word in the list must be a single English word\n"
    "- No proper nouns (e.g., no specific people or places). No specialized "
    "vocabulary (e.g., no technical terms)\n"
    "- Make a list of these 5 words, without any repetition.\n"
    '- You must list each word with a number and a period. For example, '
    '"1. word-1, 2. word-2, etc."\n'
    "- Make list of 5 words and do not generate anything else."
)

PGT_TASK_DESC = (
    "Generate a random persona description with three characteristics.\n\n"
    "Characteristics are:\n\n"
    "- First Name\n"
    "- The city of birth\n"
    "- Current occupation\n\n"
    "Format the output strictly using JSON schema. Use 'first_name' for "
    "First Name, 'city' for the city of birth, 'occupation' for current "
    "occupation as corresponding JSON keys. The ordering of characteristics "
    "should be arbitrary in your answer."
)

# Test stimuli from system_prompts.py

AUT_OBJECTS = [
    "belt", "brick", "broom", "bucket", "candle", "clock", "comb",
    "knife", "lamp", "pencil", "pillow", "purse", "rope", "sock", "table",
]

CWT_WORD_SETS = [
    "stamp, letter, send",
    "petrol, diesel, pump",
    "statement, stealth, detect",
    "belief, faith, sing",
    "gloom, payment, exist",
    "organ, empire, comply",
    "year, week, embark",
]

CDAT_WORDS = [
    "belt", "brick", "broom", "bucket", "candle", "clock", "comb",
    "knife", "lamp", "pencil", "pillow", "purse", "rope", "sock", "table",
]

RAT_TSV_URL = (
    "https://raw.githubusercontent.com/text-machine-lab/diverse-not-short/"
    "main/src/diverse_not_short/evaluation_scripts/rat_test_examples.tsv"
)


class DiverseNotShortScenario(Scenario):
    name = "diverse_not_short"
    description = "text-machine-lab/diverse-not-short"
    tags = ["creativity", "divergent_thinking", "writing"]

    VALID_TASKS = ("rat", "aut", "cwt", "dat", "cdat", "pgt")

    def __init__(self, task: str = "rat"):
        """
        Args:
            task: One of "rat", "aut", "cwt", "dat", "cdat", "pgt".
        """
        super().__init__()
        task = task.lower()
        if task not in self.VALID_TASKS:
            raise ValueError(
                f"task must be one of {self.VALID_TASKS}, got '{task}'"
            )
        self.task = task

    def _download_rat_data(self, output_path: str) -> str:
        """Download RAT test examples TSV from GitHub."""
        os.makedirs(output_path, exist_ok=True)
        tsv_path = os.path.join(output_path, "rat_test_examples.tsv")
        if not os.path.exists(tsv_path):
            urllib.request.urlretrieve(RAT_TSV_URL, tsv_path)
        return tsv_path

    def _build_rat_instances(self, output_path: str) -> List[Instance]:
        tsv_path = self._download_rat_data(output_path)
        instances = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                cue_words = f"{row['Cue1']}, {row['Cue2']}, {row['Cue3']}"
                prompt = (
                    f"{RAT_TASK_DESC}\n\n"
                    f"Cue Words: {cue_words},\nSolution:"
                )
                answer = row["Answer"].strip().lower()
                instances.append(Instance(
                    input=Input(text=prompt),
                    references=[
                        Reference(
                            Output(text=answer), tags=[CORRECT_TAG]
                        )
                    ],
                    split=TEST_SPLIT,
                    id=f"rat_{i}",
                ))
        return instances

    def _build_aut_instances(self) -> List[Instance]:
        instances = []
        for obj in AUT_OBJECTS:
            prompt = f"{AUT_TASK_DESC}\n\nObject: {obj}, Uses:"
            instances.append(Instance(
                input=Input(text=prompt),
                references=[],
                split=TEST_SPLIT,
                id=f"aut_{obj}",
            ))
        return instances

    def _build_cwt_instances(self) -> List[Instance]:
        instances = []
        for i, words in enumerate(CWT_WORD_SETS):
            prompt = (
                f"{CWT_TASK_DESC}\n\n"
                f"Write a short story that includes these three words: "
                f"{words}."
            )
            instances.append(Instance(
                input=Input(text=prompt),
                references=[],
                split=TEST_SPLIT,
                id=f"cwt_{i}",
            ))
        return instances

    def _build_dat_instances(self) -> List[Instance]:
        prompt = (
            f"{DAT_TASK_DESC}\n\n"
            "List 10 words that are as different from each other as possible:"
        )
        return [Instance(
            input=Input(text=prompt),
            references=[],
            split=TEST_SPLIT,
            id="dat_0",
        )]

    def _build_cdat_instances(self) -> List[Instance]:
        instances = []
        for word in CDAT_WORDS:
            prompt = f"{CDAT_TASK_DESC}\n\nGiven word: {word}"
            instances.append(Instance(
                input=Input(text=prompt),
                references=[],
                split=TEST_SPLIT,
                id=f"cdat_{word}",
            ))
        return instances

    def _build_pgt_instances(self) -> List[Instance]:
        return [Instance(
            input=Input(text=PGT_TASK_DESC),
            references=[],
            split=TEST_SPLIT,
            id="pgt_0",
        )]

    def get_instances(self, output_path: str) -> List[Instance]:
        builders = {
            "rat": lambda: self._build_rat_instances(output_path),
            "aut": self._build_aut_instances,
            "cwt": self._build_cwt_instances,
            "dat": self._build_dat_instances,
            "cdat": self._build_cdat_instances,
            "pgt": self._build_pgt_instances,
        }
        return builders[self.task]()
