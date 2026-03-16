"""
HELM Scenario: MOOSE-Chem2 — Fine-grained Scientific Hypothesis Discovery

Paper: MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery
       via Hierarchical Search (NeurIPS 2025)
       https://arxiv.org/abs/2505.19209
Code:  https://github.com/ZonglinY/MOOSE-Chem2

Task: Given a research background survey and a coarse-grained hypothesis direction,
      generate a detailed, experimentally actionable fine-grained scientific hypothesis
      in the chemistry domain.

Dataset: Data/chem_research_2024_finegrained.xlsx (GitHub raw download)
  - 51 rows (No. 0–50), all chemistry papers published post-2024
  - Columns used:
      Background Question          — the specific research problem to solve
      Background Little Survey     — prior-work survey providing context
      Main hypothesis              — coarse-grained hypothesis (given as input hint)
      Finegrained Hypothesis       — ground-truth fine-grained hypothesis (reference)
  - Columns skipped:
      Finegrained Experiment       — experiment protocols, not part of hypothesis task
      Inspiration paper 1-3 *      — inspiration papers used internally by MOOSE-Chem2 method
      Reasoning Process, Note      — internal annotations
      Background Little Survey (strict), Background Question (strict) — stricter variants

Prompt format: No explicit single-turn prompt is specified in the paper (MOOSE-Chem2 uses
  a multi-step hierarchical search). Standard generation format used:
    Research Question:
    {background_question}

    Background Survey:
    {background_survey}

    Coarse-grained Hypothesis:
    {main_hypothesis}

    Based on the research context above, generate a detailed fine-grained scientific
    hypothesis. Include specific chemicals, materials, reaction conditions, and
    experimental mechanisms that would make this hypothesis directly testable in a lab.

Evaluation: llm_judge (LLM-as-judge)
  See scenarios/moose_chem2/annotator_notes.md for LLMAsJuryAnnotator configuration.
  Two evaluation approaches:
    1. Component Recall: break hypothesis into technical components, score coverage
       against ground truth (Soft Recall: coverage > 0; Hard Recall: full coverage)
    2. Pairwise comparison on Overall, Effectiveness, Novelty, Detailedness, Feasibility
"""

import os
import urllib.request
from typing import List

import pandas as pd

from helm.benchmark.scenarios.scenario import (
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
    TEST_SPLIT,
)

DATA_URL = (
    "https://github.com/ZonglinY/MOOSE-Chem2/raw/master/Data/chem_research_2024_finegrained.xlsx"
)


class MOOSEChem2Scenario(Scenario):
    """MOOSE-Chem2 Fine-grained Scientific Hypothesis Discovery Scenario

    Tests a model's ability to generate detailed, experimentally actionable
    scientific hypotheses in chemistry from a research background and a
    coarse-grained hypothesis direction.
    """

    name = "moose_chem2"
    description = "ZonglinY/MOOSE-Chem2"
    tags = ["creativity", "scientific_discovery", "hypothesis_generation", "chemistry", "open_ended"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download and cache the Excel file
        cache_path = os.path.join(output_path, "chem_research_2024_finegrained.xlsx")
        if not os.path.exists(cache_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(DATA_URL, cache_path)

        df = pd.read_excel(cache_path, engine="openpyxl")

        instances = []
        for _, row in df.iterrows():
            background_question = str(row["Background Question"]).strip()
            background_survey = str(row["Background Little Survey"]).strip()
            coarse_hypothesis = str(row["Main hypothesis"]).strip()
            finegrained_hypothesis = str(row["Finegrained Hypothesis"]).strip()

            # Skip rows with missing key fields
            if not background_question or background_question.lower() == "nan":
                continue
            if not finegrained_hypothesis or finegrained_hypothesis.lower() == "nan":
                continue

            prompt = (
                "Research Question:\n"
                f"{background_question}\n\n"
                "Background Survey:\n"
                f"{background_survey}\n\n"
                "Coarse-grained Hypothesis:\n"
                f"{coarse_hypothesis}\n\n"
                "Based on the research context above, generate a detailed fine-grained "
                "scientific hypothesis. Include specific chemicals, materials, reaction "
                "conditions, and experimental mechanisms that would make this hypothesis "
                "directly testable in a lab."
            )

            # Ground truth fine-grained hypothesis as reference for annotator comparison
            references = [Reference(Output(text=finegrained_hypothesis), tags=[])]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
