"""
HELM Scenario: AlpacaEval 2.0

Paper: "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators"
       (Dubois et al., 2024)
       https://arxiv.org/abs/2404.04475
Code: https://github.com/tatsu-lab/alpaca_eval

AlpacaEval 2.0 is an LLM-based evaluation benchmark with 805 instruction-following
prompts. Models generate responses which are compared against GPT-4-turbo baseline
outputs by an LLM judge.

Prompt format:
  {instruction}
  (plain instruction, no additional formatting)

Example:
  "What are the names of some famous actors that started their careers on Broadway?"

Fields used: instruction (prompt), output (reference baseline), dataset (source)
Fields skipped: generator (baseline model name)

Evaluation: LLM-as-judge (GPT-4-turbo or similar) compares model output against baseline
            Metrics: Win Rate (WR) and Length-Controlled Win Rate (LCWR)
            Higher WR/LCWR indicates better performance

Dataset source: https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json
Used in: DARLING paper (arXiv:2509.02534) for creative writing evaluation
"""

from typing import List
import json
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class AlpacaEval2Scenario(Scenario):
    """
    AlpacaEval 2.0 benchmark

    Evaluates instruction-following capabilities across 805 diverse prompts.
    While primarily an instruction-following benchmark, it includes creative
    writing, open-ended reasoning, and other tasks requiring diverse responses.

    LLM judge compares model outputs against GPT-4-turbo baseline using
    pairwise comparison. Length-controlled metrics address length bias.
    """

    name = "alpaca_eval_2"
    description = "tatsu-lab/alpaca_eval"
    tags = ["creativity", "instruction_following", "open_ended", "diverse"]

    DATASET_DOWNLOAD_URL = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load AlpacaEval 2.0 dataset and create instances.

        Each instance contains an instruction prompt. Models generate responses
        which are then compared against reference baseline outputs by an LLM judge.
        """

        # Download dataset
        data_path = os.path.join(output_path, "alpaca_eval.json")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            unpack=False,
        )

        # Load JSON data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instances = []
        for idx, item in enumerate(data):
            # Extract fields
            instruction = item['instruction']
            reference_output = item['output']  # Baseline (text_davinci_003 or gpt4)
            dataset_source = item.get('dataset', '')

            # Create prompt (plain instruction)
            prompt = instruction

            # Reference is the baseline output for comparison by LLM judge
            # In practice, the judge sees both the model output and this reference
            references = [Reference(Output(text=reference_output), tags=[])]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"alpaca_eval_{idx}_{dataset_source}",
                )
            )

        return instances
