"""
HELM Scenario: Material Generation Benchmark (MP-20)

Paper: MGB: The Material Generation Benchmark (OpenReview)
Code: https://github.com/txie-93/cdvae
Dataset: MP-20 subset from Materials Project (45,231 stable materials, 1-20 atoms per unit cell)

Task: Text-to-structure generation - LLMs generate crystal structures in CIF format
      given material composition and properties.

Prompt format:
  Generate a crystal structure in CIF format for the following material:

  Formula: {formula}
  Space group: {spacegroup}
  Formation energy per atom: {formation_energy} eV/atom
  Band gap: {band_gap} eV
  Energy above hull: {e_above_hull} eV/atom

  Output the structure in standard CIF format:

Fields used: pretty_formula, spacegroup.number, formation_energy_per_atom, band_gap, e_above_hull
Fields skipped: material_id (identifier), elements (redundant with formula), cif (reference output)

Evaluation: Custom metrics required - structural matching (match rate, RMSE),
            chemical validity, physical plausibility. See metric_notes.md.
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


class MaterialGenerationMP20Scenario(Scenario):
    """
    Material Generation Benchmark - MP-20 dataset

    Evaluates LLMs on generating crystal structures in CIF format from
    material composition and property specifications.

    The MP-20 dataset contains 45,231 experimentally stable materials from
    Materials Project with up to 20 atoms per unit cell, spanning 89 elements.
    """

    name = "material_generation_mp20"
    description = "MP-20 dataset from Materials Project via CDVAE"
    tags = ["creativity", "materials_science", "structure_generation", "chemistry"]

    DATASET_DOWNLOAD_URL = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/{split}.csv"

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load MP-20 dataset and create instances for text-to-structure generation.

        Each instance prompts the model to generate a CIF format crystal structure
        given material formula and properties.
        """

        # Download test and validation splits
        instances = []

        for split_name, split_tag in [("test", TEST_SPLIT), ("val", VALID_SPLIT)]:
            # Download dataset file
            dataset_url = self.DATASET_DOWNLOAD_URL.format(split=split_name)
            dataset_path = os.path.join(output_path, f"mp_20_{split_name}.csv")
            ensure_file_downloaded(
                source_url=dataset_url,
                target_path=dataset_path,
                unpack=False,
            )

            # Parse CSV and create instances
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Build prompt with material specifications
                    prompt = self._build_prompt(row)

                    # Reference is the ground-truth CIF structure
                    reference_cif = row['cif']

                    # Create instance
                    # Note: For structure generation tasks, the reference contains
                    # the expected CIF output. Custom metrics will compare generated
                    # CIF to reference CIF using structural matching algorithms.
                    instances.append(
                        Instance(
                            input=Input(text=prompt),
                            references=[Reference(output=reference_cif, tags=[])],
                            split=split_tag,
                        )
                    )

        return instances

    def _build_prompt(self, row: dict) -> str:
        """
        Build text prompt for material structure generation.

        Provides material formula and key properties as constraints.
        """
        formula = row['pretty_formula']
        spacegroup = row['spacegroup.number']
        formation_energy = row['formation_energy_per_atom']
        band_gap = row['band_gap']
        e_above_hull = row['e_above_hull']

        prompt = f"""Generate a crystal structure in CIF format for the following material:

Formula: {formula}
Space group: {spacegroup}
Formation energy per atom: {formation_energy} eV/atom
Band gap: {band_gap} eV
Energy above hull: {e_above_hull} eV/atom

Output the structure in standard CIF format:"""

        return prompt
