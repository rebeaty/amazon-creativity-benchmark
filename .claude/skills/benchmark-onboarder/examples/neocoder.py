"""NEOCODER: Benchmarking Language Model Creativity on Code Generation

Citation:
    Lu, Y., Wang, D., Li, T., Jiang, D., Khudanpur, S., Jiang, M., & Khashabi, D. (2025).
    Benchmarking Language Model Creativity: A Case Study on Code Generation.
    In Proceedings of NAACL 2025, pp. 2776-2794.

Paper: https://aclanthology.org/2025.naacl-long.141/
"""

import json
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class NeocoderScenario(Scenario):
    """NEOCODER: Code generation creativity benchmark with programming constraints.

    The benchmark evaluates creative problem-solving by requiring models to solve
    competitive programming problems while adhering to constraints on programming
    techniques (e.g., "DO NOT use for loop").

    Dataset contains 199 CodeForces problems, each with 6 variants:
    - 1 original problem (no constraints)
    - 5 variants with increasing programming technique constraints

    Total: 1194 instances (199 problems × 6 variants)

    The evaluation uses the NeoGauge@T metric which combines:
    - Correctness: Code execution against test cases
    - Technique detection: Checking constraint violations (requires external evaluator)
    """

    name = "neocoder"
    description = "JHU-CLSP/NeoCoder"
    tags = ["creativity", "code_generation", "constraint_satisfaction", "problem_solving"]

    DATASET_DOWNLOAD_URL = "https://github.com/JHU-CLSP/NeoCoder.git"

    def __init__(self, constraint_level: str = "all"):
        """Initialize NEOCODER scenario.

        Args:
            constraint_level: Which constraint level to include:
                - "all": All 6 variants per problem (default)
                - "0": Original problems only (no constraints)
                - "1": Level 1 constraints (e.g., no for loop)
                - "2": Level 2 constraints (e.g., no for loop, no while loop)
                - "3": Level 3 constraints
                - "4": Level 4 constraints
                - "5": Maximum constraints (all 5 constraint types)
        """
        super().__init__()
        valid_levels = ["all", "0", "1", "2", "3", "4", "5"]
        if constraint_level not in valid_levels:
            raise ValueError(
                f"constraint_level must be one of {valid_levels}, got: {constraint_level}"
            )
        self.constraint_level = constraint_level

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load NEOCODER dataset and create HELM instances."""
        # Set up data directory
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)

        # Download dataset via git clone if not already present
        repo_dir = os.path.join(data_dir, "NeoCoder")
        if not os.path.exists(repo_dir):
            os.system(f"git clone {self.DATASET_DOWNLOAD_URL} {repo_dir}")

        # Load main dataset
        dataset_path = os.path.join(
            repo_dir, "datasets", "CodeForce", "NeoCoder", "NeoCoder.json"
        )
        with open(dataset_path, "r") as f:
            problems = json.load(f)

        # Load test cases
        test_cases_path = os.path.join(
            repo_dir, "datasets", "CodeForce", "NeoCoder", "test_cases_annotated.json"
        )
        with open(test_cases_path, "r") as f:
            test_cases = json.load(f)

        # Create lookup for test cases by problem_id
        test_cases_by_id = {tc["problem_id"]: tc for tc in test_cases}

        instances = []
        for problem in problems:
            problem_id = problem["problem_id"]
            problem_statements = problem["problem_statements"]
            constraints_list = problem["constraints_list"]

            # Get test cases for this problem
            test_case = test_cases_by_id.get(problem_id)
            if not test_case:
                # Skip problems without test cases
                continue

            # Determine which constraint levels to include
            if self.constraint_level == "all":
                levels_to_include = range(len(problem_statements))
            else:
                level_idx = int(self.constraint_level)
                if level_idx >= len(problem_statements):
                    # Skip if requested level doesn't exist for this problem
                    continue
                levels_to_include = [level_idx]

            # Create instances for each constraint level
            for level_idx in levels_to_include:
                prompt_text = problem_statements[level_idx]
                constraints = constraints_list[level_idx]

                # Format constraints for display
                if constraints == ["this is the og problem"]:
                    constraint_info = "none"
                else:
                    constraint_info = ", ".join(constraints)

                # Create references from test cases
                # Note: For code generation, we include test I/O as references
                # Actual evaluation requires code execution (external to HELM)
                references = []
                for inp, out in zip(test_case["input"], test_case["output"]):
                    # Format input (list of lists) as string
                    input_str = "\n".join([" ".join(line) for line in inp])
                    ref_text = f"Input:\n{input_str}\n\nExpected Output:\n{out}"
                    references.append(
                        Reference(Output(text=ref_text), tags=[CORRECT_TAG])
                    )

                instance = Instance(
                    input=Input(text=prompt_text),
                    references=references,
                    id=f"{problem_id}_level{level_idx}",
                    split="test",
                )
                instances.append(instance)

        return instances


# Example usage and test
if __name__ == "__main__":
    scenario = NeocoderScenario(constraint_level="all")
    instances = scenario.get_instances("/tmp/neocoder_test")
    print(f"Loaded {len(instances)} instances")
    if instances:
        print(f"\nExample instance ID: {instances[0].id}")
        print(f"Prompt preview: {instances[0].input.text[:200]}...")
        print(f"Number of references: {len(instances[0].references)}")
