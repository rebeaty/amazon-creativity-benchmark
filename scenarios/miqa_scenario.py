"""
HELM Scenario: MiQA (Metaphorical Inference Questions and Answers)

Paper: https://arxiv.org/abs/2210.07993 (AACL 2022)
Code: https://github.com/google-research/language/tree/master/language/miqa

Task: Binary multiple choice inference on metaphorical statements. Two question types:
  1. "implies": Given a metaphorical premise, which conclusion follows?
  2. "implied_by": Given a literal conclusion, which premise implies it?

Dataset:
  - 150 base examples, each generates 2 questions (300 total)
  - Fields: literal_premise, metaphorical_premise, literal_conclusion, metaphorical_conclusion

Prompt format (from Paper Appendix A - format 2):
  "implies" task:
    "Mp". Which of the following two statements could that imply? (1) Lc (2) Mc

  "implied_by" task (similar format per paper):
    "Lc". Which of the following two statements could imply that? (1) Lp (2) Mp

Note: Paper tested multiple prompt formats; format 2 used here as a clean middle-ground.
Different models performed best with different prompts (see Appendix A).

Subsets:
  "all" - Both question types (300 examples)
  "implies" - Only "implies" questions (150 examples)
  "implied_by" - Only "implied_by" questions (150 examples)

Evaluation: exact_match (binary accuracy)
"""

import csv
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class MiQAScenario(Scenario):
    name = "miqa"
    description = "google-research/language/miqa"
    tags = ["creativity", "metaphor", "inference", "figurative_language"]

    DATA_URL = "https://raw.githubusercontent.com/google-research/language/master/language/miqa/data/metaphor_inference_qa.tsv"

    SUBSETS = ["all", "implies", "implied_by"]

    def __init__(self, subset: str = "all"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"Unknown subset: {subset}. Must be one of {self.SUBSETS}")
        self.subset = subset

    def get_instances(self, output_path: str):
        # Download data
        data_path = os.path.join(output_path, "metaphor_inference_qa.tsv")
        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(self.DATA_URL, data_path)

        instances = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                literal_premise = row['literal_premise']
                metaphorical_premise = row['metaphorical_premise']
                literal_conclusion = row['literal_conclusion']
                metaphorical_conclusion = row['metaphorical_conclusion']

                # Generate "implies" question (Appendix A format 2)
                # M_p -> which follows: M_c (correct) or L_c (incorrect)?
                if self.subset in ["all", "implies"]:
                    # Format: "Mp". Which of the following two statements could that imply? (1) Lc (2) Mc
                    prompt_implies = (
                        f'"{metaphorical_premise.capitalize()}". '
                        f'Which of the following two statements could that imply? '
                        f'(1) {literal_conclusion} (2) {metaphorical_conclusion}'
                    )
                    # Correct answer is (2) - metaphorical conclusion
                    instances.append(Instance(
                        input=Input(text=prompt_implies),
                        references=[
                            Reference(Output(text="1"), tags=[]),
                            Reference(Output(text="2"), tags=[CORRECT_TAG]),
                        ],
                        split=TEST_SPLIT
                    ))

                # Generate "implied_by" question (similar format per Appendix A)
                # L_c is implied by: L_p (correct) or M_p (incorrect)?
                if self.subset in ["all", "implied_by"]:
                    # Format: "Lc". Which of the following two statements could imply that? (1) Lp (2) Mp
                    prompt_implied_by = (
                        f'"{literal_conclusion.capitalize()}". '
                        f'Which of the following two statements could imply that? '
                        f'(1) {literal_premise} (2) {metaphorical_premise}'
                    )
                    # Correct answer is (1) - literal premise
                    instances.append(Instance(
                        input=Input(text=prompt_implied_by),
                        references=[
                            Reference(Output(text="1"), tags=[CORRECT_TAG]),
                            Reference(Output(text="2"), tags=[]),
                        ],
                        split=TEST_SPLIT
                    ))

        return instances
