"""
HELM Scenario: Speak-to-Structure (S2-Bench / TOMG-Bench)
          — Open-Domain Natural Language-Driven Molecule Generation

Paper: "Speak-to-Structure: Evaluating LLMs in Open-domain
        Natural Language-Driven Molecule Generation"
       (arXiv:2412.14642)
Authors: Jiatong Li, Junxian Li, Yunqing Liu, Dongzhan Zhou, Qing Li
Website: https://phenixace.github.io/tomgbench/
Data:    https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench

Task: Given a natural language instruction, generate a SMILES string
representing a molecule that satisfies the specified constraints
(property-based generation) or modifications (molecule editing /
optimization). Tests creative molecular design from open-domain
language descriptions.

Three main tasks, 10 subtasks total:
  MolCustom — Generate a novel molecule satisfying given properties:
    AtomNum:        "Generate a molecule with 8 carbon and 3 oxygen atoms."
    BasicProp:      "Generate a molecule with high melting point and toxicity."
    BondNum:        "Generate a molecule with 15 single bonds."
    FunctionalGroup:"Generate a molecule with 1 hydroxyl and 2 benzene rings."

  MolEdit — Edit an existing molecule by adding/removing/substituting groups:
    AddComponent:   "Add an amide group to molecule {SMILES}."
    DelComponent:   "Remove a benzene ring from molecule {SMILES}."
    SubComponent:   "Substitute nitro with hydroxyl in molecule {SMILES}."

  MolOpt — Optimize a molecule to improve a target property:
    LogP:           "Modify molecule {SMILES} to decrease its LogP value."
    MR:             "Modify molecule {SMILES} to decrease its MR value."
    QED:            "Modify molecule {SMILES} to increase its QED value."

Prompt format: Verbatim `Instruction` field from TOMG-Bench CSV, prepended
with a SMILES-output instruction for general LLMs.
Dataset: Duke-de-Artois/TOMG-Bench on HuggingFace (10 subtask CSVs)
  Note: load_dataset() fails due to mixed file formats; use hf_hub_download
  for per-subtask CSV download (confirmed working approach).
Fields used:   Instruction (NL prompt), molecule (source SMILES for MolEdit/Opt)
Fields skipped: atom count columns, group columns, property baselines
               (used as evaluation ground truth, not model inputs)
Evaluation: custom (requires RDKit; see metric_notes.md)
  Tier 1 (HELM-computable): SMILES validity, correct syntax
  Tier 2 (RDKit required): constraint satisfaction, novelty, similarity

Parameters:
  task:    "MolCustom" | "MolEdit" | "MolOpt" | "all" (default: "all")
  subtask: specific subtask name or "all" (default: "all")
  max_instances_per_subtask: int (default: 500; full subtask has 5,000)
"""

import csv
import os
from typing import List, Optional

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

_HF_REPO_ID = "Duke-de-Artois/TOMG-Bench"

_SUBTASKS = {
    "MolCustom": ["AtomNum", "BasicProp", "BondNum", "FunctionalGroup"],
    "MolEdit":   ["AddComponent", "DelComponent", "SubComponent"],
    "MolOpt":    ["LogP", "MR", "QED"],
}

_VALID_TASKS = list(_SUBTASKS.keys()) + ["all"]

# System context prepended to every instruction to elicit SMILES output
_SYSTEM_CONTEXT = (
    "You are an expert chemist specializing in molecular design. "
    "When asked to generate or modify a molecule, output only the SMILES "
    "string of the resulting molecule. Do not include any explanation, "
    "reasoning, formula name, or additional text. Output exactly one "
    "valid SMILES string."
)

_PROMPT_TEMPLATE = "{system}\n\n{instruction}"


def _hf_csv_url(task: str, subtask: str) -> str:
    """Construct HuggingFace resolve URL for a subtask CSV."""
    return (
        f"https://huggingface.co/datasets/{_HF_REPO_ID}/resolve/main/"
        f"benchmarks/open_generation/{task}/{subtask}/test.csv"
    )


def _load_subtask_csv(task: str, subtask: str, output_path: str) -> List[dict]:
    """Download and parse a TOMG-Bench subtask CSV."""
    import urllib.request

    url = _hf_csv_url(task, subtask)
    local_path = os.path.join(output_path, f"{task}_{subtask}_test.csv")

    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)

    with open(local_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class SpeakToStructureScenario(Scenario):
    """
    Speak-to-Structure (S2-Bench / TOMG-Bench) — molecule generation from
    natural language instructions.

    10 subtasks across 3 task types (MolCustom, MolEdit, MolOpt). Each
    instance asks the model to output a valid SMILES string satisfying the
    described molecular constraints or modifications.

    50,000 test records total (5,000 per subtask); default loads 500 per
    subtask for practical HELM evaluation. Evaluation requires RDKit; see
    metric_notes.md. SMILES validity is HELM-computable without chemistry tools.
    """

    name = "speak_to_structure"
    description = "huggingface.co/datasets/Duke-de-Artois/TOMG-Bench (arXiv:2412.14642)"
    tags = ["creativity", "scientific_creativity", "molecule_generation",
            "chemistry", "open_ended_generation"]

    def __init__(
        self,
        task: str = "all",
        subtask: str = "all",
        max_instances_per_subtask: int = 500,
    ):
        super().__init__()
        if task not in _VALID_TASKS:
            raise ValueError(
                f"Unknown task: {task!r}. Must be one of {_VALID_TASKS}"
            )
        self.task = task
        self.subtask = subtask
        self.max_instances_per_subtask = max_instances_per_subtask

    def get_instances(self, output_path: str) -> List[Instance]:
        active_tasks = (
            list(_SUBTASKS.keys()) if self.task == "all" else [self.task]
        )

        instances = []
        for task in active_tasks:
            subtasks = (
                _SUBTASKS[task]
                if self.subtask == "all"
                else [self.subtask]
            )
            for subtask in subtasks:
                rows = _load_subtask_csv(task, subtask, output_path)
                rows = rows[: self.max_instances_per_subtask]

                for row in rows:
                    instruction = row["Instruction"].strip()
                    prompt = _PROMPT_TEMPLATE.format(
                        system=_SYSTEM_CONTEXT,
                        instruction=instruction,
                    )

                    instances.append(
                        Instance(
                            input=Input(text=prompt),
                            references=[],   # No gold SMILES; evaluated by RDKit
                            split=TEST_SPLIT,
                        )
                    )

        return instances
        # Default (task="all", max_instances_per_subtask=500):
        # 10 subtasks × 500 = 5,000 instances
