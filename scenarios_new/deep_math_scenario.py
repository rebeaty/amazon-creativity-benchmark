"""
HELM Scenario Example: DeepMath-Creative

Paper: DeepMath-Creative: A Benchmark for Evaluating Mathematical Creativity of Large Language Models (2025)
       https://arxiv.org/abs/2505.08744
Code: https://github.com/DeepMathLLM/DeepMath

Description:
Evaluates mathematical creativity through constructive problems across algebra, geometry, topology,
and analysis. Contains two task types:
1. Proof problems (78 total): Prove or provide counterexample to mathematical statements
2. Counterexample problems (101 total): Determine if mathematical objects exist and provide examples

Language: Mixed Chinese and English
- Proof problems 1-46: Chinese
- Proof problems 47-78: English
- All counterexample problems: Chinese

Prompt format: Problem statements are used directly as prompts with no additional formatting.
The problems already contain clear instructions embedded within them.

Evaluation: Open-ended generation requiring mathematical correctness judgment. The paper mentions
"lenient scoring criteria emphasizing core solution components." Requires expert review or
LLM-as-judge annotators. No ground truth solutions provided in dataset.

Key patterns demonstrated:
- Parsing markdown files with numbered problems
- Handling multilingual content (Chinese + English)
- Open-ended mathematical reasoning tasks
- Multiple task types within one benchmark
- Downloading raw files from GitHub

Fields used: Problem text (extracted from markdown)
Fields skipped: None (data is in markdown format)
"""

import os
import re
from typing import List
from urllib.parse import quote
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class DeepMathCreativeScenario(Scenario):
    """
    DeepMath-Creative benchmark for evaluating mathematical creativity of LLMs.

    Contains 179 problems total:
    - 78 proof/counterexample problems (mixed Chinese/English)
    - 101 existence/counterexample problems (Chinese)

    This example demonstrates:
    - Parsing markdown files with numbered problems
    - Supporting multilingual benchmarks
    - Open-ended generation tasks without ground truth
    - Flexible task type filtering
    """

    name = "deepmath_creative"
    description = "DeepMathLLM/DeepMath"  # Data source
    tags = ["creativity", "mathematics", "reasoning", "multilingual", "chinese"]

    # Raw markdown files from GitHub
    PROOF_PROBLEMS_URL = "https://raw.githubusercontent.com/DeepMathLLM/DeepMath/master/DeepMath-Creative/datasets/" + quote("78道证明题.md")
    COUNTEREXAMPLE_PROBLEMS_URL = "https://raw.githubusercontent.com/DeepMathLLM/DeepMath/master/DeepMath-Creative/datasets/" + quote("101道反例题.md")

    def __init__(self, task_type: str = "all"):
        """
        Args:
            task_type: Which problems to include
                - "all": Both proof and counterexample problems (default, 179 total)
                - "proof": Only proof/counterexample problems (78 problems)
                - "counterexample": Only existence/counterexample problems (101 problems)
        """
        super().__init__()
        self.task_type = task_type

    def _parse_markdown_problems(self, content: str, problem_type: str) -> List[dict]:
        """
        Parse numbered problems from markdown content.

        Handles both Chinese and English number formats:
        - Chinese format: "1、问题文本..."
        - English format: "47. Problem text..."

        Args:
            content: Raw markdown content
            problem_type: "proof" or "counterexample" for labeling

        Returns:
            List of problem dictionaries with id, number, type, and text
        """
        problems = []

        # Pattern matches both formats:
        # - Chinese: \d+、 (number + Chinese enumeration comma)
        # - English: \d+\. (number + period)
        # Captures problem text until next numbered item or end of string
        pattern = r'(\d+)[、\.](.+?)(?=\n\n\d+[、\.]|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)

        for number, text in matches:
            text = text.strip()
            if text:  # Skip empty problems
                problems.append({
                    "problem_id": f"{problem_type}_{number}",
                    "problem_number": int(number),
                    "problem_type": problem_type,
                    "text": text
                })

        return problems

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load and parse problems from DeepMath-Creative markdown files.

        Returns:
            List of Instance objects with problem text as input and empty references
            (since this is open-ended generation requiring expert evaluation)
        """
        instances = []

        # Load proof problems if requested
        if self.task_type in ["all", "proof"]:
            proof_file = os.path.join(output_path, "deepmath_creative_proof.md")
            ensure_file_downloaded(
                source_url=self.PROOF_PROBLEMS_URL,
                target_path=proof_file,
                unpack=False,
            )

            with open(proof_file, "r", encoding="utf-8") as f:
                proof_content = f.read()

            proof_problems = self._parse_markdown_problems(proof_content, "proof")

            for problem in proof_problems:
                # For open-ended generation, we use an empty reference with CORRECT_TAG
                # This signals that evaluation requires external judgment
                instances.append(
                    Instance(
                        input=Input(text=problem["text"]),
                        references=[Reference(output=Output(text=""), tags=[CORRECT_TAG])],
                        split=TEST_SPLIT,
                    )
                )

        # Load counterexample problems if requested
        if self.task_type in ["all", "counterexample"]:
            ce_file = os.path.join(output_path, "deepmath_creative_counterexample.md")
            ensure_file_downloaded(
                source_url=self.COUNTEREXAMPLE_PROBLEMS_URL,
                target_path=ce_file,
                unpack=False,
            )

            with open(ce_file, "r", encoding="utf-8") as f:
                ce_content = f.read()

            ce_problems = self._parse_markdown_problems(ce_content, "counterexample")

            for problem in ce_problems:
                instances.append(
                    Instance(
                        input=Input(text=problem["text"]),
                        references=[Reference(output=Output(text=""), tags=[CORRECT_TAG])],
                        split=TEST_SPLIT,
                    )
                )

        return instances
