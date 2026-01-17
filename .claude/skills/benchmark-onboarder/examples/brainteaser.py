"""
HELM Scenario: BRAINTEASER

Prompt source: Paper Section 3.2, Figure 2
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
