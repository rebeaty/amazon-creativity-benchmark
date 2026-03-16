"""
HELM Scenario: Materials Transformers - Hybrid-pure

Paper: "Materials Transformers Language Models for Generative Materials Design" (Fu et al., 2022)
Code: https://github.com/usccolumbia/mtransformer
Dataset: Hybrid-pure (257,138 materials from ICSD+MP+OQMD, charge-neutral filtered)

Task: Open-ended generation of chemical formulas in expanded element form.
Evaluation: Validity, uniqueness, recovery rate, novelty
"""

from typing import List
import os
import zipfile
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Reference, Output, TEST_SPLIT, VALID_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class MaterialsTransformersHybridPureScenario(Scenario):
    name = "materials_transformers_hybrid_pure"
    description = "Hybrid-pure dataset from Materials Transformers"
    tags = ["creativity", "materials_science", "chemistry", "generation"]
    DATASET_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/35991917"
    DATASET_NAME = "hy_pure"

    def get_instances(self, output_path: str) -> List[Instance]:
        zip_path = os.path.join(output_path, "MT_dataset.zip")
        extract_path = os.path.join(output_path, "MT_dataset")

        if not os.path.exists(extract_path):
            ensure_file_downloaded(source_url=self.DATASET_DOWNLOAD_URL,
                                 target_path=zip_path, unpack=False)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = [m for m in zip_ref.namelist()
                          if m.startswith(f"MT_dataset/{self.DATASET_NAME}/")]
                zip_ref.extractall(output_path, members)

        instances = []
        for split_name, split_tag in [("test", TEST_SPLIT), ("valid", VALID_SPLIT)]:
            split_path = os.path.join(extract_path, self.DATASET_NAME, f"{split_name}.txt")
            with open(split_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    prompt = "Generate a chemically valid material composition:"
                    instances.append(Instance(
                        input=Input(text=prompt),
                        references=[Reference(Output(text=line), tags=[])],
                        split=split_tag,
                        id=f"{self.DATASET_NAME}_{split_name}_{idx}",
                    ))
        return instances
