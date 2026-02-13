"""
HELM Scenario: Material Generation Benchmark (PEROV-5)

Paper: MGB: The Material Generation Benchmark (OpenReview)
Code: https://github.com/txie-93/cdvae
Dataset: PEROV-5 from perovskite water-splitting database (18,928 perovskite materials)

Task: Text-to-structure generation for perovskite materials (ABX3 formula).

Prompt format:
  Generate a crystal structure in CIF format for the following perovskite material:

  Formula: {formula}
  Heat of formation (all): {heat_all} eV/atom
  Heat of formation (reference): {heat_ref} eV/atom
  Direct band gap: {dir_gap} eV
  Indirect band gap: {ind_gap} eV

  Output the structure in standard CIF format:

Fields used: formula, heat_all, heat_ref, dir_gap, ind_gap
Fields skipped: material_id (identifier), cif (reference output)

Notes: All materials have 5 atoms per unit cell (ABX3 structure).
       A, B are nonradioactive metals; X is from O, N, S, F.
"""

from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    TEST_SPLIT,
    VALID_SPLIT,
)
from helm.common.general import ensure_file_downloaded
import csv
import os


class MaterialGenerationPerov5Scenario(Scenario):
    """
    Material Generation Benchmark - PEROV-5 dataset

    Evaluates LLMs on generating perovskite crystal structures in CIF format
    from material composition and property specifications.

    The PEROV-5 dataset contains 18,928 perovskite materials (ABX3 formula)
    curated from a water-splitting database, all with 5 atoms per unit cell.
    """

    name = "material_generation_perov5"
    description = "PEROV-5 perovskite dataset via CDVAE"
    tags = ["creativity", "materials_science", "structure_generation", "chemistry", "perovskite"]

    DATASET_DOWNLOAD_URL = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/perov_5/{split}.csv"

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load PEROV-5 dataset and create instances for text-to-structure generation.
        """
        instances = []

        for split_name, split_tag in [("test", TEST_SPLIT), ("val", VALID_SPLIT)]:
            dataset_url = self.DATASET_DOWNLOAD_URL.format(split=split_name)
            dataset_path = os.path.join(output_path, f"perov_5_{split_name}.csv")
            ensure_file_downloaded(
                source_url=dataset_url,
                target_path=dataset_path,
                unpack=False,
            )

            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompt = self._build_prompt(row)
                    reference_cif = row['cif']

                    instances.append(
                        Instance(
                            input=Input(text=prompt),
                            references=[Reference(output=reference_cif, tags=[])],
                            split=split_tag,
                        )
                    )

        return instances

    def _build_prompt(self, row: dict) -> str:
        """Build prompt for perovskite structure generation."""
        formula = row['formula']
        heat_all = row['heat_all']
        heat_ref = row['heat_ref']
        dir_gap = row['dir_gap']
        ind_gap = row['ind_gap']

        prompt = f"""Generate a crystal structure in CIF format for the following perovskite material:

Formula: {formula}
Heat of formation (all): {heat_all} eV/atom
Heat of formation (reference): {heat_ref} eV/atom
Direct band gap: {dir_gap} eV
Indirect band gap: {ind_gap} eV

Output the structure in standard CIF format:"""

        return prompt
