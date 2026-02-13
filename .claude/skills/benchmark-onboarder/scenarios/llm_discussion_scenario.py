"""
HELM Scenario: LLM Discussion Creativity Tests

Paper: https://arxiv.org/abs/2405.06373 (COLM 2024)
Dataset: https://github.com/lawraa/LLM-Discussion

This benchmark includes 4 classic divergent thinking creativity tests:

1. Alternative Uses Test (AUT): Generate creative uses for everyday objects (30 items)
2. Similarities Test: Find ways in which two things are alike (30 pairs)
3. Instances Test: Name all things in a category (30 categories)
4. Scientific Creativity Test: Scientific creativity across 5 question types (30 items)

Prompt format: Open-ended generation
Evaluation: llm_judge
  - Judge model: GPT-4 or GPT-3.5
  - Dimensions:
    * Fluency: Count of unique, relevant responses
    * Flexibility: Variety of distinct categories/perspectives
    * Originality: Novelty of ideas (1-5 scale)
    * Elaboration: Detail and development of responses (1-5 scale)
"""

import json
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    TEST_SPLIT,
)


class LLMDiscussionScenario(Scenario):
    """LLM Discussion Creativity Tests Scenario

    Evaluates creative thinking across 4 divergent thinking tasks:
    - aut: Alternative Uses Test (creative uses for objects)
    - similarities: Similarities Test (ways two things are alike)
    - instances: Instances Test (name things in a category)
    - scientific: Scientific Creativity Test (scientific creativity questions)
    """

    name = "llm_discussion"
    description = "lawraa/LLM-Discussion"
    tags = ["creativity", "divergent_thinking", "idea_generation"]

    VALID_TESTS = ["aut", "similarities", "instances", "scientific", "all"]

    def __init__(self, test: str = "aut"):
        super().__init__()
        if test not in self.VALID_TESTS:
            raise ValueError(f"Invalid test: {test}. Must be one of {self.VALID_TESTS}")
        self.test = test

    def get_instances(self, output_path: str) -> List[Instance]:
        instances = []

        if self.test == "all" or self.test == "aut":
            instances.extend(self._get_aut_instances())

        if self.test == "all" or self.test == "similarities":
            instances.extend(self._get_similarities_instances())

        if self.test == "all" or self.test == "instances":
            instances.extend(self._get_instances_instances())

        if self.test == "all" or self.test == "scientific":
            instances.extend(self._get_scientific_instances())

        return instances

    def _get_aut_instances(self) -> List[Instance]:
        """Alternative Uses Test: Generate creative uses for everyday objects"""
        test_url = "https://raw.githubusercontent.com/lawraa/LLM-Discussion/main/Datasets/AUT/aut_30_test.json"

        with urllib.request.urlopen(test_url) as response:
            data = json.loads(response.read().decode())

        # Use the baseline task formulation (first task)
        baseline_task = data["Task"][0]
        task_lines = baseline_task["Problem"]

        # Get all test objects
        objects = [example["object"] for example in data["Examples"]]

        instances = []
        for obj in objects:
            # Format the prompt with the specific object
            prompt_text = " ".join(task_lines).replace("{object}", obj)

            # Create instance with no references (open-ended, LLM-as-judge)
            instances.append(
                Instance(
                    input=Input(text=prompt_text),
                    references=[],
                    split=TEST_SPLIT,
                )
            )

        return instances

    def _get_similarities_instances(self) -> List[Instance]:
        """Similarities Test: Find ways in which two things are alike"""
        test_url = "https://raw.githubusercontent.com/lawraa/LLM-Discussion/main/Datasets/Similarities/similarities_30_test.json"

        with urllib.request.urlopen(test_url) as response:
            data = json.loads(response.read().decode())

        instances = []
        for prompt in data["Examples"]:
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                )
            )

        return instances

    def _get_instances_instances(self) -> List[Instance]:
        """Instances Test: Name all things in a category"""
        test_url = "https://raw.githubusercontent.com/lawraa/LLM-Discussion/main/Datasets/Instances/instances_30_test.json"

        with urllib.request.urlopen(test_url) as response:
            data = json.loads(response.read().decode())

        instances = []
        for prompt in data["Examples"]:
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                )
            )

        return instances

    def _get_scientific_instances(self) -> List[Instance]:
        """Scientific Creativity Test: Scientific creativity across 5 question types"""
        test_url = "https://raw.githubusercontent.com/lawraa/LLM-Discussion/main/Datasets/Scientific/scientific_30_test.json"

        with urllib.request.urlopen(test_url) as response:
            data = json.loads(response.read().decode())

        instances = []
        # Each task has multiple examples (6 examples per task type)
        for task in data["Task"]:
            for example_prompt in task["Example"]:
                instances.append(
                    Instance(
                        input=Input(text=example_prompt),
                        references=[],
                        split=TEST_SPLIT,
                    )
                )

        return instances
