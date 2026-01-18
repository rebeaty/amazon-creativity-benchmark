"""
HELM Scenario: BRAINTEASER

Paper: https://arxiv.org/abs/2310.05057 (EMNLP 2023)
Code: https://github.com/1171-jpg/BrainTeaser

Prompt format:
  Question: {question}
  A. {choice_0}
  B. {choice_1}
  C. {choice_2}
  D. {choice_3}

Fields used: question, choices, answer
Fields skipped: none
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)

class BrainteaserScenario(Scenario):
    name = "brainteaser"
    description = "yzha/R1_distilled_brain_teasers"
    tags = ["creativity", "reasoning", "multiple_choice"]

    def get_instances(self, output_path):
        dataset = load_dataset("yzha/R1_distilled_brain_teasers", split="train")

        instances = []
        for item in dataset:
            # Format prompt (from paper Section 3.2)
            prompt = f"Question: {item['question']}\n"
            for i, choice in enumerate(item['choices']):
                prompt += f"\n{chr(65+i)}. {choice}"

            # HELM MC pattern: all choices are References
            references = []
            for i, choice in enumerate(item['choices']):
                letter = chr(65 + i)
                is_correct = (choice == item['answer'])
                tags = [CORRECT_TAG] if is_correct else []
                references.append(Reference(Output(text=letter), tags=tags))

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
