"""
HELM Scenario: CS4 (Comparing the Skill of Creating Stories by Controlling the Synthesized Constraint Specificity)

Paper: https://arxiv.org/abs/2410.04197 (October 2024)
       "CS4: Measuring the Creativity of Large Language Models Automatically
        by Controlling the Number of Story-Writing Constraints"
Code: https://github.com/anirudhlakkaraju/cs4_benchmark

Task: Generate creative stories that satisfy a varying number of constraints. The benchmark
evaluates LLM creativity by systematically increasing prompt specificity through additional
constraints, preventing models from simply reproducing training data.

Dataset: 500 unique prompts (50 base instructions × 10 variants)
  - Two types: Instruction-based (realistic fiction) and Story-based (writing prompts)
  - Constraint levels: 7, 15, 23, 31, 39 constraints per prompt
  - Total: 250 Instruction-based + 250 Story-based = 500 instances

Evaluation: Automatic metrics (no ground truth stories needed)
  - Constraint Satisfaction: GPT-4-based evaluation of whether each constraint is met
  - Coherence: Overall narrative coherence
  - Diversity: N-gram diversity to assess originality
  - Perplexity: Predictability and fluency
  - QUC (Quality Under Constraints): Quality when constrained
  - RCS: Relative creativity score

Key Finding: LLMs struggle to balance creativity, constraint satisfaction, and coherence
when prompts become highly specific (31-39 constraints).

Prompt format:
  {instruction}

  Constraints:
  {numbered list of constraints}

  Write a story that satisfies all the above constraints.
"""

import csv
import os
import urllib.request
from typing import List, Optional
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Reference, Output,
    CORRECT_TAG, TEST_SPLIT
)


class CS4Scenario(Scenario):
    name = "cs4"
    description = "CS4 Benchmark - Constrained Creative Story Generation"
    tags = ["creativity", "generation", "story", "constraints", "diversity"]

    # Dataset URLs from GitHub
    INSTRUCTION_CONSTRAINTS_URL = "https://raw.githubusercontent.com/anirudhlakkaraju/cs4_benchmark/master/CS4_dataset/Instruction-based%20Constraints.csv"
    INSTRUCTION_STORIES_URL = "https://raw.githubusercontent.com/anirudhlakkaraju/cs4_benchmark/master/CS4_dataset/Instruction-based%20Base%20Stories.csv"
    STORY_CONSTRAINTS_URL = "https://raw.githubusercontent.com/anirudhlakkaraju/cs4_benchmark/master/CS4_dataset/Story-based%20Constraints.csv"
    STORY_STORIES_URL = "https://raw.githubusercontent.com/anirudhlakkaraju/cs4_benchmark/master/CS4_dataset/Story-based%20Base%20Stories.csv"

    def __init__(
        self,
        dataset_type: str = "instruction",
        constraint_levels: Optional[List[int]] = None,
        num_instances: Optional[int] = None
    ):
        """
        Initialize CS4 scenario.

        Args:
            dataset_type: Type of dataset to use - "instruction" (realistic fiction),
                         "story" (writing prompts), or "both" (default: "instruction")
            constraint_levels: List of constraint levels to include (e.g., [7, 15, 23, 31, 39]).
                              If None, includes all constraint levels. (default: None)
            num_instances: Maximum number of instances to include. If None, includes all.
                          (default: None)
        """
        super().__init__()
        self.dataset_type = dataset_type.lower()
        self.constraint_levels = constraint_levels
        self.num_instances = num_instances

        if self.dataset_type not in ["instruction", "story", "both"]:
            raise ValueError(f"dataset_type must be 'instruction', 'story', or 'both', got: {self.dataset_type}")

    def _download_file(self, url: str, output_path: str, filename: str) -> str:
        """Download a file if it doesn't exist."""
        file_path = os.path.join(output_path, filename)
        if not os.path.exists(file_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(url, file_path)
        return file_path

    def _load_constraints_csv(self, file_path: str) -> List[dict]:
        """Load constraints CSV file and return list of instances."""
        instances = []
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up column names (remove BOM and extra spaces)
                cleaned_row = {k.strip(): v for k, v in row.items()}
                instances.append(cleaned_row)
        return instances

    def _format_prompt(self, instruction: str, constraints: str) -> str:
        """Format the prompt for story generation."""
        prompt = f"{instruction}\n\nConstraints:\n{constraints}\n\nWrite a story that satisfies all the above constraints."
        return prompt

    def get_instances(self, output_path: str) -> List[Instance]:
        """Generate instances for CS4 benchmark."""
        all_instances = []

        # Process Instruction-based dataset
        if self.dataset_type in ["instruction", "both"]:
            constraints_path = self._download_file(
                self.INSTRUCTION_CONSTRAINTS_URL,
                output_path,
                "instruction_constraints.csv"
            )

            constraint_data = self._load_constraints_csv(constraints_path)

            for row in constraint_data:
                # Get constraint level
                num_constraints = int(row.get('Number of Constraints', 0))

                # Filter by constraint level if specified
                if self.constraint_levels and num_constraints not in self.constraint_levels:
                    continue

                instruction = row.get('Instruction ', '').strip()  # Note: there's a space in the column name
                constraints = row.get('Constraints', '').strip()

                # Format the input prompt
                prompt = self._format_prompt(instruction, constraints)

                # Create instance
                # No ground truth stories, so references are empty
                # Evaluation will be done via automatic metrics (constraint satisfaction, coherence, etc.)
                instance = Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                    extra_data={
                        "instruction_number": row.get('Instruction Number', ''),
                        "num_constraints": num_constraints,
                        "dataset_type": "instruction",
                        "instruction": instruction,
                        "constraints": constraints
                    }
                )
                all_instances.append(instance)

        # Process Story-based dataset
        if self.dataset_type in ["story", "both"]:
            constraints_path = self._download_file(
                self.STORY_CONSTRAINTS_URL,
                output_path,
                "story_constraints.csv"
            )

            constraint_data = self._load_constraints_csv(constraints_path)

            for row in constraint_data:
                # Get constraint level
                num_constraints = int(row.get('Number of Constraints', 0))

                # Filter by constraint level if specified
                if self.constraint_levels and num_constraints not in self.constraint_levels:
                    continue

                instruction = row.get('Instruction ', '').strip()  # Note: there's a space in the column name
                constraints = row.get('Constraints', '').strip()

                # Format the input prompt
                prompt = self._format_prompt(instruction, constraints)

                # Create instance
                instance = Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                    extra_data={
                        "instruction_number": row.get('Instruction Number', ''),
                        "num_constraints": num_constraints,
                        "dataset_type": "story",
                        "instruction": instruction,
                        "constraints": constraints
                    }
                )
                all_instances.append(instance)

        # Limit number of instances if specified
        if self.num_instances:
            all_instances = all_instances[:self.num_instances]

        return all_instances
