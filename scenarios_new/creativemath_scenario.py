"""
HELM Scenario: CreativeMath

Paper: "CreativeMath: Assessing Mathematical Creativity in LLMs"
       https://arxiv.org/abs/2407.14910 (AAAI 2025)
Code: https://github.com/JunyiYe/CreativeMath

CreativeMath evaluates whether LLMs can generate novel mathematical
solutions that are fundamentally different from existing ones. Given a
competition math problem and k reference solutions, the model must produce
a new solution using a different approach.

400 problems across 8 competition levels (AMC 8, AMC 10, AMC 12, AHSME,
AIME, USAJMO, USAMO, IMO), 50 each. Problems have 1-5 existing solutions.
For each problem with n solutions, instances are created at k=1..n
(progressively more reference solutions shown), yielding 605 total instances.

Prompt source: Exact prompt from src/prompts/prompts.py
  (load_novel_solution_generation_prompt). Includes 5 novelty criteria.

Fields used: problem, solutions, competition, problem_id, difficulty
Fields skipped: competition_id

Evaluation: 3-stage LLM-as-judge pipeline (correctness → coarse novelty →
  fine novelty). See annotator_notes.md for details.
"""

import json
import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

DATA_URL = (
    "https://raw.githubusercontent.com/JunyiYe/CreativeMath"
    "/main/data/subset.json"
)

GENERATION_PROMPT = """\
Criteria for evaluating the difference between two mathematical solutions include:
i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;
iv). A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

Given the following mathematical problem:
{problem}

And some typical solutions:
{reference_solutions}

Please output a novel solution distinct from the given ones for this math problem."""


class CreativeMathScenario(Scenario):
    name = "creativemath"
    description = "JunyiYe/CreativeMath"
    tags = ["creativity", "math", "reasoning"]

    def _download_data(self, output_path: str) -> list:
        """Download dataset from GitHub."""
        cache_path = os.path.join(output_path, "creativemath_subset.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

        os.makedirs(output_path, exist_ok=True)
        req = urllib.request.Request(DATA_URL)
        req.add_header("User-Agent", "HELM-CreativeMath/1.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        data = self._download_data(output_path)

        instances = []
        for problem_idx, item in enumerate(data):
            problem = item["problem"]
            solutions = list(item["solutions"].values())
            n = len(solutions)
            competition = item.get("competition", "")

            for k in range(1, n + 1):
                # Build reference solutions string (first k)
                reference_solutions = "\n\n".join(
                    f"Solution {i + 1}:\n{sol}"
                    for i, sol in enumerate(solutions[:k])
                )

                prompt_text = GENERATION_PROMPT.format(
                    problem=problem,
                    reference_solutions=reference_solutions,
                )

                # For open-ended generation, references are empty.
                # The withheld solutions (k+1..n) serve as ground truth
                # for fine-grained novelty evaluation, stored as references
                # for potential use by annotators.
                references = []
                if k < n:
                    for sol in solutions[k:]:
                        references.append(
                            Reference(Output(text=sol), tags=[])
                        )

                instance_id = (
                    f"{competition}_{item['problem_id']}_k{k}"
                )

                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=references,
                        split=TEST_SPLIT,
                        id=instance_id,
                    )
                )

        return instances
