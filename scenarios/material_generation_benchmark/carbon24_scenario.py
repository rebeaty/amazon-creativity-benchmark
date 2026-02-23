"""
HELM Scenario: Material Generation Benchmark (Carbon-24)

Paper: MGB: The Material Generation Benchmark (OpenReview)
Code: https://github.com/txie-93/cdvae
Dataset: Carbon-24 from AIRSS carbon structures at 10 GPa (10,153 carbon allotropes)

Task: Text-to-structure generation for carbon allotrope structures.

Prompt format:
  Generate a crystal structure in CIF format for the following carbon allotrope:

  Material ID: {material_id}
  Energy per atom: {energy_per_atom} eV/atom
  Pressure condition: 10 GPa

  Output the structure in standard CIF format:

Fields used: material_id, energy_per_atom
Fields skipped: cif (reference output)

Notes: All materials are pure carbon with 6-24 atoms per unit cell.
       Structures obtained via ab initio random structure searching (AIRSS) at 10 GPa.
       Most structures are thermodynamically unstable but kinetically stable.
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


class MaterialGenerationCarbon24Scenario(Scenario):
    """
    Material Generation Benchmark - Carbon-24 dataset

    Evaluates LLMs on generating carbon allotrope crystal structures in CIF
    format from energy specifications.

    The Carbon-24 dataset contains 10,153 carbon structures with 6-24 atoms
    per unit cell, obtained via AIRSS at 10 GPa pressure.
    """

    name = "material_generation_carbon24"
    description = "Carbon-24 dataset from AIRSS via CDVAE"
    tags = ["creativity", "materials_science", "structure_generation", "chemistry", "carbon"]

    DATASET_DOWNLOAD_URL = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/{split}.csv"

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load Carbon-24 dataset and create instances for text-to-structure generation.
        """
        instances = []

        for split_name, split_tag in [("test", TEST_SPLIT), ("val", VALID_SPLIT)]:
            dataset_url = self.DATASET_DOWNLOAD_URL.format(split=split_name)
            dataset_path = os.path.join(output_path, f"carbon_24_{split_name}.csv")
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
        """Build prompt for carbon allotrope structure generation."""
        material_id = row['material_id']
        energy_per_atom = row['energy_per_atom']

        prompt = f"""Generate a crystal structure in CIF format for the following carbon allotrope:

Material ID: {material_id}
Energy per atom: {energy_per_atom} eV/atom
Pressure condition: 10 GPa

Output the structure in standard CIF format:"""

        return prompt
