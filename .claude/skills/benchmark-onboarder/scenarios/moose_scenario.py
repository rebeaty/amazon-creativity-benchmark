"""
HELM Scenario: MOOSE — Scientific Hypothesis Generation

Paper: MOOSE: Multi-module Open-ended Social-science hypothesis Exploration
       ACL 2024 Findings
       https://github.com/ZonglinY/MOOSE

Task: Given two research background documents (title + passage) from social science,
      generate a novel, valid, and practically useful scientific hypothesis.

Dataset: Data/business_research.xlsx in the MOOSE GitHub repo
  - 50 rows, each describing one research problem
  - Columns used:
      background_1_title, background_1_passage  — first source document
      background_2_title, background_2_passage  — second source document
      Main hypotheis                            — gold-standard hypothesis (note: typo in column name)
  - Columns skipped:
      inspiration_*   — used only by the MOOSE generation framework, not the raw task
      background_*_golden, inspiration_*_golden — system-internal annotations

Prompt format: No explicit model prompt is given in the paper (MOOSE is a method paper).
  Standard generation format used:
    Research Background:

    [Title 1]
    [Passage 1]

    [Title 2]
    [Passage 2]

    Based on the research above, generate a novel, testable scientific hypothesis
    in the business/social science domain.

Evaluation: llm_judge (GPT-4)
  See scenarios/moose/annotator_notes.md for LLMAsJuryAnnotator configuration.
  Dimensions: Validness, Novelty, Helpfulness (1–5 scale each)
  The original evaluator scores hypotheses in isolation (without background context);
  our annotator_notes.md extends this to include background context for grounded evaluation.

Note: Gold-standard hypotheses (Main hypotheis column) are included as references
      and may optionally be used by the annotator for comparison-based evaluation.
"""

import io
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

DATA_URL = "https://github.com/ZonglinY/MOOSE/raw/main/Data/business_research.xlsx"


class MOOSEScenario(Scenario):
    """MOOSE Scientific Hypothesis Generation Scenario

    Tests a model's ability to synthesize novel, valid scientific hypotheses
    from pairs of social science research background documents.
    """

    name = "moose"
    description = "ZonglinY/MOOSE"
    tags = ["creativity", "scientific_discovery", "hypothesis_generation", "open_ended"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download business_research.xlsx and load with pandas
        cache_path = os.path.join(output_path, "business_research.xlsx")
        if not os.path.exists(cache_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(DATA_URL, cache_path)

        df = pd.read_excel(cache_path, engine="openpyxl")

        instances = []
        for _, row in df.iterrows():
            bg1_title = str(row.get("background_1_title", "") or "").strip()
            bg1_passage = str(row.get("background_1_passage", "") or "").strip()
            bg2_title = str(row.get("background_2_title", "") or "").strip()
            bg2_passage = str(row.get("background_2_passage", "") or "").strip()
            gold_hypothesis = str(row.get("Main hypotheis", "") or "").strip()

            # Skip rows without usable background content
            if not (bg1_passage or bg2_passage):
                continue

            # Build background section — include only non-empty documents
            background_parts = []
            if bg1_title and bg1_passage:
                background_parts.append(f"{bg1_title}\n{bg1_passage}")
            elif bg1_passage:
                background_parts.append(bg1_passage)

            if bg2_title and bg2_passage:
                background_parts.append(f"{bg2_title}\n{bg2_passage}")
            elif bg2_passage:
                background_parts.append(bg2_passage)

            background_text = "\n\n".join(background_parts)

            prompt = (
                "Research Background:\n\n"
                f"{background_text}\n\n"
                "Based on the research above, generate a novel, testable scientific "
                "hypothesis in the business/social science domain."
            )

            # Gold hypothesis as reference (used by annotator for comparison if needed)
            references = []
            if gold_hypothesis and gold_hypothesis.lower() != "nan":
                references.append(Reference(Output(text=gold_hypothesis), tags=[]))

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
