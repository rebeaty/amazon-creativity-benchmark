"""
HELM Scenario: LLM-SRBench

Paper: LLM-SRBench: A New Benchmark for Scientific Equation Discovery
       with Large Language Models (ICML 2025 Oral)
       https://arxiv.org/abs/2504.10415
Code: https://github.com/deep-symbolic-mathematics/llm-srbench

LLM-SRBench evaluates LLMs on scientific equation discovery (symbolic regression).
Given variable descriptions and training data points, the model must discover
the underlying mathematical equation.

Prompt format (adapted from LLMSR method, methods/llmsr/searcher.py lines 130-134):
  Find the mathematical equation that represents {output_desc},
  given data on {input_descs}.

  Variables:
    {symbol}: {description} ({property})

  Training Data (first {n} of {total} points):
    {formatted table}

  Write the equation as a symbolic mathematical expression using the
  variable names above.

Prompt source: Adapted from LLMSR direct prompting (searcher.py);
  original methods use iterative code generation + execution
Fields used: name, symbols, symbol_descs, symbol_properties, expression, samples (from HDF5)
Fields skipped: none

Subsets:
  - "lsr_transform": 111 Feynman equations in non-standard mathematical forms
  - "matsci": 25 materials science equations (synthetic)
  - "chem_react": 36 chemical reaction kinetics equations (synthetic)
  - "bio_pop_growth": 24 population growth equations (synthetic)
  - "phys_osc": 43 nonlinear oscillation equations (synthetic)
  - "all": All 239 problems (default)

Notes:
  - Gated HuggingFace dataset; requires HF_TOKEN with access granted at
    https://huggingface.co/datasets/nnheui/llm-srbench
  - Sample data stored in HDF5 (lsr_bench_data.hdf5), downloaded via huggingface_hub
  - Original benchmark uses iterative code generation + execution; adapted here
    for single-turn text-based equation generation
  - Ground truth is symbolic expression; evaluation requires custom metric (NMSE/R²)
    since equivalent equations can differ in string form (e.g. x^2+2x+1 vs (x+1)^2)
"""

import os
from pathlib import Path

import numpy as np
import h5py
import datasets
from huggingface_hub import snapshot_download
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)

REPO_ID = "nnheui/llm-srbench"
MAX_DATA_ROWS = 20

SUBSETS = {
    "lsr_transform": {"hf_split": "lsr_transform", "h5_prefix": "lsr_transform"},
    "matsci": {"hf_split": "lsr_synth_matsci", "h5_prefix": "lsr_synth/matsci"},
    "chem_react": {"hf_split": "lsr_synth_chem_react", "h5_prefix": "lsr_synth/chem_react"},
    "bio_pop_growth": {"hf_split": "lsr_synth_bio_pop_growth", "h5_prefix": "lsr_synth/bio_pop_growth"},
    "phys_osc": {"hf_split": "lsr_synth_phys_osc", "h5_prefix": "lsr_synth/phys_osc"},
}

# Sample keys differ between transform and synth datasets
TRAIN_KEY = {"lsr_transform": "train", "default": "train_data"}


def _format_data_table(samples: np.ndarray, symbols: list, max_rows: int = MAX_DATA_ROWS) -> str:
    """Format numerical samples as a readable text table."""
    n_points = samples.shape[0]
    display = samples[:max_rows]

    header = "\t".join(symbols)
    rows = []
    for row in display:
        rows.append("\t".join(f"{v:.6g}" for v in row))

    table = header + "\n" + "\n".join(rows)
    if n_points > max_rows:
        table += f"\n... ({n_points} points total)"
    return table


class LlmSrbenchScenario(Scenario):
    name = "llm_srbench"
    description = "nnheui/llm-srbench"
    tags = ["creativity", "scientific_reasoning", "equation_discovery"]

    def __init__(self, subset: str = "all"):
        super().__init__()
        if subset != "all" and subset not in SUBSETS:
            raise ValueError(f"subset must be 'all' or one of {list(SUBSETS.keys())}, got '{subset}'")
        self.subset = subset

    def _get_subsets(self):
        if self.subset == "all":
            return list(SUBSETS.keys())
        return [self.subset]

    def get_instances(self, output_path: str):
        dataset_dir = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset", token=True))
        h5_path = dataset_dir / "lsr_bench_data.hdf5"
        all_ds = datasets.load_dataset(REPO_ID, token=True)

        instances = []
        with h5py.File(h5_path, "r") as h5:
            for subset_name in self._get_subsets():
                cfg = SUBSETS[subset_name]
                ds = all_ds[cfg["hf_split"]]
                train_key = TRAIN_KEY.get(subset_name, TRAIN_KEY["default"])

                for entry in ds:
                    name = entry["name"]
                    symbols = entry["symbols"]
                    symbol_descs = entry["symbol_descs"]
                    symbol_properties = entry["symbol_properties"]
                    expression = entry["expression"]

                    # Load training samples from HDF5
                    h5_key = f"/{cfg['h5_prefix']}/{name}/{train_key}"
                    samples = h5[h5_key][...].astype(np.float64)

                    # Build variable descriptions
                    # symbols[0] is output, rest are inputs
                    output_desc = symbol_descs[0]
                    input_descs = []
                    var_lines = []
                    for i, (sym, desc, prop) in enumerate(
                        zip(symbols, symbol_descs, symbol_properties)
                    ):
                        role = "output" if i == 0 else "input"
                        var_lines.append(f"  {sym}: {desc} ({role})")
                        if i > 0 and "V" in prop:
                            input_descs.append(desc)

                    if len(input_descs) > 2:
                        input_str = ", ".join(input_descs[:-1]) + ", and " + input_descs[-1]
                    elif len(input_descs) == 2:
                        input_str = input_descs[0] + " and " + input_descs[1]
                    elif input_descs:
                        input_str = input_descs[0]
                    else:
                        input_str = "the input variables"

                    data_table = _format_data_table(samples, symbols)

                    prompt = (
                        f"Find the mathematical equation that represents {output_desc}, "
                        f"given data on {input_str}.\n\n"
                        f"Variables:\n"
                        f"{chr(10).join(var_lines)}\n\n"
                        f"Training Data:\n{data_table}\n\n"
                        f"Write the equation as a symbolic mathematical expression "
                        f"using the variable names above. Output only the equation."
                    )

                    references = [
                        Reference(Output(text=expression), tags=[CORRECT_TAG])
                    ]

                    instances.append(Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=TEST_SPLIT,
                    ))

        return instances
