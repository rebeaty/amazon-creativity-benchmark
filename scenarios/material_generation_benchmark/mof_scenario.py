"""
HELM Scenario: MOF Crystal Structure Generation (MOFFlow)

Paper: "MOFFlow: Flow Matching for Structure Prediction of Metal-Organic Frameworks"
       (ICLR 2025)
       https://arxiv.org/abs/2410.17270
Code: https://github.com/nayoung10/MOFFlow
Dataset: Boyd et al. (2019) - 324,426 MOF structures
         https://zenodo.org/records/15187230

Task: Text-to-structure generation - LLMs generate MOF (Metal-Organic Framework)
      structures in CIF format given building block specifications.

MOFs are highly porous crystalline materials with applications in gas storage, separation,
and catalysis. Unlike simple inorganic crystals, MOFs consist of metal nodes and organic
linkers forming complex 3D structures.

Prompt format:
  Generate a MOF structure in CIF format with the following specifications:

  Metal nodes: {metal_nodes}
  Organic linkers: {organic_linkers}
  Topology: {topology}

  Output the structure in standard CIF format:

Fields used: building blocks (metal nodes, organic linkers), topology
Evaluation: Structural matching (match rate: 31.69% baseline), validity, physical
            plausibility. See ../metric_notes.md.

Data source: Zenodo (15187230), filtered structures with ≥200 blocks
Split: 8:1:1 (train:valid:test) following MOFFlow paper
Test set: ~32,443 examples (10% of 324,426 filtered structures)
"""

from typing import List
import os
import csv
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


class MOFFlowScenario(Scenario):
    """
    MOF Crystal Structure Generation benchmark (MOFFlow)

    Evaluates LLMs on generating Metal-Organic Framework structures in CIF format
    from building block specifications. MOFs are complex porous materials with
    hundreds of atoms, requiring understanding of:
    - Metal-organic coordination chemistry
    - 3D spatial arrangement of rigid building blocks
    - Topological constraints
    - Crystallographic symmetry

    This is significantly harder than simple inorganic crystal generation (MP-20)
    due to the complexity and size of MOF structures.
    """

    name = "mof_structure_generation"
    description = "Boyd et al. (2019) MOF dataset via MOFFlow/Zenodo"
    tags = ["creativity", "materials_science", "structure_generation", "chemistry", "mof"]

    # Note: This URL points to Zenodo where the MOFFlow dataset is hosted
    # The actual CSV with metadata would need to be constructed from the CIF files
    # or obtained from the MOFFlow preprocessing pipeline
    DATASET_BASE_URL = "https://zenodo.org/records/15187230/files"

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load MOF structure generation dataset.

        Each instance contains specifications for a MOF structure (metal nodes,
        organic linkers, topology). Models generate CIF format crystal structures.

        Note: The MOFFlow dataset is distributed as CIF files on Zenodo. For HELM
        integration, we would need to either:
        1. Download and process the CIF files to extract metadata
        2. Use a pre-processed CSV with structure specifications
        3. Work with the MOFFlow authors to release a HELM-compatible format

        This scenario provides the structure; actual data loading would require
        additional preprocessing steps or collaboration with dataset authors.
        """

        instances = []

        # Placeholder implementation - in practice, would load from Zenodo/preprocessed files
        # The MOFFlow paper uses ~32K test examples (10% of 324K structures)

        # Example instance format (actual data would come from Zenodo/MOFFlow processing):
        # for split_name, split_tag in [("test", TEST_SPLIT), ("val", VALID_SPLIT)]:
        #     # Load MOF metadata (building blocks, topology, properties)
        #     for mof_id, mof_data in dataset[split_name].items():
        #         prompt = self._build_prompt(mof_data)
        #         reference_cif = mof_data['cif']
        #
        #         instances.append(
        #             Instance(
        #                 input=Input(text=prompt),
        #                 references=[Reference(Output(text=reference_cif), tags=[])],
        #                 split=split_tag,
        #                 id=f"mof_{mof_id}",
        #             )
        #         )

        # Note: To fully implement this scenario, we need:
        # 1. Access to preprocessed metadata (building blocks, topology per structure)
        # 2. A way to extract this from the 324K CIF files on Zenodo
        # 3. Or collaboration with MOFFlow authors for a release format

        return instances

    def _build_prompt(self, mof_data: dict) -> str:
        """
        Build text prompt for MOF structure generation.

        MOF structures are specified by their building blocks:
        - Metal nodes (e.g., Zn paddle-wheel, Cu dimer)
        - Organic linkers (e.g., terephthalic acid, BPDC)
        - Topology (e.g., pcu, dia, sod)
        """
        # Example prompt structure:
        metal_nodes = mof_data.get('metal_nodes', 'Unknown')
        organic_linkers = mof_data.get('organic_linkers', 'Unknown')
        topology = mof_data.get('topology', 'Unknown')

        prompt = f"""Generate a MOF structure in CIF format with the following specifications:

Metal nodes: {metal_nodes}
Organic linkers: {organic_linkers}
Topology: {topology}

Output the structure in standard CIF format:"""

        return prompt


# Note to implementers:
# To complete this scenario, coordinate with MOFFlow authors or:
# 1. Download CIF files from Zenodo (https://zenodo.org/records/15187230)
# 2. Use MOFid to extract building blocks from each CIF
# 3. Create a CSV mapping: mof_id -> building_blocks -> CIF
# 4. Implement the loading logic above
# The MOFFlow paper uses this dataset with train/valid/test split 8:1:1
