"""
HELM Scenario: Materials Transformers - ICSD-mix

Paper: "Materials Transformers Language Models for Generative Materials Design:
        a benchmark study" (Fu et al., 2022)
        https://arxiv.org/abs/2206.13578
Code: https://github.com/usccolumbia/mtransformer
Dataset: ICSD-mix (52,317 materials from ICSD database, mixed purity)

Task: Open-ended generation of chemical formulas in expanded element form.
      Models learn to generate novel, chemically valid material compositions.

Prompt format:
  Generate a chemically valid material composition:

Expected output format (expanded element sequence):
  Rb Rb Sn Sn Se Se Se Se Se .
  (represents Rb2Sn2Se5)

Fields used: Expanded element sequences from training data
Evaluation: Custom metrics - validity (charge neutrality: 97.54%,
            electronegativity balance: 91.40%), uniqueness, recovery rate, novelty
            (6x better than random baseline)

Data source: Figshare MT_dataset (https://figshare.com/articles/dataset/MT_dataset/20122796)
             Extracts from icsd_mix/ folder
"""

from typing import List
import os
import zipfile
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
    VALID_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class MaterialsTransformersICSDMixScenario(Scenario):
    """
    Materials Transformers ICSD-mix benchmark

    Evaluates LLMs on generating chemically valid material formulas.
    ICSD-mix contains 52,317 materials from the Inorganic Crystal Structure Database
    with mixed purity (includes charge-imbalanced samples for robustness).

    Models generate expanded element sequences like "Li Li O ." which represent
    chemical formulas (Li2O). Evaluation focuses on validity, uniqueness, and novelty.
    """

    name = "materials_transformers_icsd_mix"
    description = "ICSD-mix dataset from Materials Transformers"
    tags = ["creativity", "materials_science", "chemistry", "generation"]

    DATASET_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/35991917"
    DATASET_NAME = "icsd_mix"

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load ICSD-mix dataset and create instances for formula generation.

        Each instance prompts the model to generate a chemically valid material
        formula in expanded element form. The test set contains held-out materials
        for evaluating generalization.
        """

        # Download and extract dataset
        zip_path = os.path.join(output_path, "MT_dataset.zip")
        extract_path = os.path.join(output_path, "MT_dataset")

        if not os.path.exists(extract_path):
            ensure_file_downloaded(
                source_url=self.DATASET_DOWNLOAD_URL,
                target_path=zip_path,
                unpack=False,
            )

            # Extract the specific dataset folder
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract only the needed dataset
                members = [m for m in zip_ref.namelist()
                          if m.startswith(f"MT_dataset/{self.DATASET_NAME}/")]
                zip_ref.extractall(output_path, members)

        instances = []

        # Load test and validation splits
        for split_name, split_tag in [("test", TEST_SPLIT), ("valid", VALID_SPLIT)]:
            split_path = os.path.join(extract_path, self.DATASET_NAME, f"{split_name}.txt")

            with open(split_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    # The line is an expanded element sequence (e.g., "Li Li O .")
                    # For generation tasks, we use an empty prompt (unconditional generation)
                    # or could use the first few elements as a seed
                    prompt = "Generate a chemically valid material composition:"

                    # Reference is the target formula for evaluation
                    # In practice, evaluation will use custom metrics for validity,
                    # not direct string matching
                    reference_formula = line

                    instances.append(
                        Instance(
                            input=Input(text=prompt),
                            references=[Reference(Output(text=reference_formula), tags=[])],
                            split=split_tag,
                            id=f"{self.DATASET_NAME}_{split_name}_{idx}",
                        )
                    )

        return instances
