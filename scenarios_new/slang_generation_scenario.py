"""
HELM Scenario: Slang Generation Evaluative Framework

Paper: An Evaluative Framework for Creativity in LLM-Generated Slang
       https://arxiv.org/abs/2502.XXXXX  (paper_id: f75eed717c978f385dd0e5fd6d58294f4e1bbc21)
Code:  https://github.com/siyangwu1/LLM-Slang-Dictionary

Task: Given a concept/definition, generate a creative slang term, a brief
definition, and a usage example. Tests creative lexical invention — the ability
to coin new informal language for a given meaning.

Dataset: conv_slang.txt — 666 real conversational slang entries (term, definition)
stored as Python tuple strings. Sourced from conversational slang corpora;
used in the paper as reference human slang for evaluation.

The three generation modes from the paper (all use the same definition input):
  - Freeform: generate any novel slang usage
  - Coinage:  coin an entirely new word
  - Reuse:    repurpose an existing English word with a new slang meaning

This scenario implements the Freeform mode, which is the most general and
broadly applicable. Coinage and Reuse are stylistic variants on the same input.

Prompt format — adapted from scripts/generation.py (Freeform / general mode):
  Generate a novel slang term in English that expresses the following definition.
  Provide: (1) the slang word, (2) a brief definition, and (3) a usage example.

  Definition: {definition}

Prompt source: scripts/generation.py build_prompt_general(), simplified to n=1
Fields used: definition (input), word (soft reference)
Fields skipped: model-generated CSV files (gpt4o_*.csv, llama_8b-it_*.csv)
  — these contain pre-computed outputs, not gold standards

Evaluation: custom — see metric_notes.md
  Primary: Semantic Novelty (SBERT mean Euclidean distance vs. standard dict)
  Secondary: open_ended BLEU/ROUGE against gold slang term (soft proxy only)

Note on references: The gold slang term from conv_slang.txt is included as a
soft reference for BLEU/ROUGE computation, but it is NOT the unique correct
answer — many valid slang terms can express any definition. The semantic novelty
metric (reference-free) is the primary evaluation signal per the paper.
"""

import ast
import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

_DATA_URL = (
    "https://raw.githubusercontent.com/siyangwu1/LLM-Slang-Dictionary"
    "/main/data/conv_slang.txt"
)


class SlangGenerationScenario(Scenario):
    """
    Given a definition, generate a creative slang term and usage example.

    Uses 666 real conversational slang entries from conv_slang.txt as the
    test set. The gold slang term is a soft reference; evaluation primarily
    uses semantic novelty (see metric_notes.md).
    """

    name = "slang_generation"
    description = "siyangwu1/LLM-Slang-Dictionary"
    tags = ["creativity", "language_generation", "slang", "lexical_creativity"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "conv_slang.txt")
        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(_DATA_URL, data_path)

        instances = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Each line is a Python tuple string: ('term', 'definition')
                try:
                    term, definition = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue

                definition = definition.strip()
                term = term.strip()
                if not definition or not term:
                    continue

                prompt = (
                    "Generate a novel slang term in English that expresses the "
                    "following definition. Provide: (1) the slang word, "
                    "(2) a brief definition, and (3) a usage example.\n\n"
                    f"Definition: {definition}"
                )

                # Gold slang term as a soft reference (not unique correct answer)
                references = [
                    Reference(Output(text=term), tags=[CORRECT_TAG])
                ]

                instances.append(Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                ))

        return instances
