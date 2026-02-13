"""
HELM Scenario: Metaphoric Analogies from Literary Texts

Paper: Automatic Extraction of Metaphoric Analogies from Literary Texts:
       Task Formulation, Dataset Construction, and Evaluation
       https://arxiv.org/abs/2412.15375
       COLING 2025

Dataset: https://github.com/Mionies/metaphoric-analogies-extraction
License: CC BY 4.0

Task: Metaphoric Analogy Extraction
Extract the four concepts (T1, T2, S1, S2) forming a metaphoric analogy from
literary texts.

- T1, T2: Target domain (main topic discussed)
- S1, S2: Source domain (image/metaphor)

The relation between T1 and T2 is analogous to the relation between S1 and S2.

Prompt format (adapted from paper experiments):
  Text: [literary text with metaphor]

  Extract the four concepts forming the metaphoric analogy:
  T1 (target entity):
  T2 (target category):
  S1 (source entity):
  S2 (source category):

Example:
  Text: "A skyscraper is in architecture as a boast is in interpersonal relations."
  T1: skyscraper
  T2: architecture
  S1: boast
  S2: interpersonal-relations

Fields used: tagged_sentence, T1, T2, S1, S2
Fields skipped: id, n. implicit term (requires manual evaluation), Author,
                ref page or link, source, long context (metadata)

Evaluation: Open-ended generation (paper uses lemmatized head noun match)
The paper evaluates using lemmatized head noun matching, which can be approximated
with standard NLG metrics (BLEU, ROUGE, F1) in HELM.

Note: Some terms may be implicit (marked with <angle brackets>), requiring
inference beyond literal text extraction.
"""

import pandas as pd
import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)


class MetaphoricAnalogiesScenario(Scenario):
    """
    Metaphoric Analogies from Literary Texts

    Evaluates models' ability to extract metaphoric analogies (4-term proportional
    analogies) from literary texts, requiring understanding of figurative language
    and conceptual mappings.
    """

    name = "metaphoric_analogies"
    description = "github.com/Mionies/metaphoric-analogies-extraction"
    tags = ["creativity", "metaphor", "analogy", "figurative_language"]

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load Metaphoric Analogies dataset and create instances.

        Each instance contains:
        - Text: Literary text with metaphoric analogy
        - Reference: The four terms (T1, T2, S1, S2) forming the analogy
        """
        # Download data from GitHub if not already present
        data_path = os.path.join(output_path, "signed-met-1.3.csv")

        if not os.path.exists(data_path):
            import urllib.request
            url = "https://raw.githubusercontent.com/Mionies/metaphoric-analogies-extraction/main/data/signed-met-dataset-v1.3/signed-met-1.3.csv"
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(url, data_path)

        # Load dataset
        df = pd.read_csv(data_path)

        instances = []
        for idx, row in df.iterrows():
            # Remove XML tags from text for cleaner input
            text = row['tagged_sentence']
            text = text.replace('<t>', '').replace('</t>', '')
            text = text.replace('<m>', '').replace('</m>', '')

            # Build prompt
            prompt = (
                f"Text: {text}\n\n"
                "Extract the four concepts forming the metaphoric analogy. "
                "T1 and T2 are the target domain (main topic), "
                "S1 and S2 are the source domain (metaphor/image).\n\n"
                "T1 (target entity):\n"
                "T2 (target category):\n"
                "S1 (source entity):\n"
                "S2 (source category):"
            )

            # Build reference answer
            # Format: "T1: [term]\nT2: [term]\nS1: [term]\nS2: [term]"
            reference_text = (
                f"T1: {row['T1']}\n"
                f"T2: {row['T2']}\n"
                f"S1: {row['S1']}\n"
                f"S2: {row['S2']}"
            )

            references = [
                Reference(
                    Output(text=reference_text),
                    tags=[CORRECT_TAG]
                )
            ]

            # Create instance
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"metaphoric_analogy_{idx}"
                )
            )

        return instances
