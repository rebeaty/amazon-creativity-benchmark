"""
HELM Scenario: Simile Generation (SCOPE)

Paper: https://aclanthology.org/2020.emnlp-main.524/ (EMNLP 2020)
Code: https://github.com/tuhinjubcse/SimileGeneration-EMNLP2020

Task: Convert literal sentences into similes. Given a sentence with a literal
expression (marked with brackets in the source data), generate a creative simile
that expresses the same meaning figuratively.

Example:
  Input: "From the day you were born, you've been invincible"
  Output: "From the day you were born, you've been like a well-seasoned superhero."

Dataset:
  - 150 test examples with human-written reference similes
  - Multiple references per example (Human1, Human2, SCOPE model)

Evaluation: open_ended (BLEU, ROUGE against reference similes)

Prompt format:
  Convert the following literal sentence into a simile using "like" or "as":

  Literal: {sentence}
  Simile:

Prompt source: Standard instruction format (paper uses fine-tuned BART)
Fields used: Literal Sense (input), Human1/Human2 (references)
"""

import csv
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class SimileGenerationScenario(Scenario):
    name = "simile_generation"
    description = "tuhinjubcse/SimileGeneration-EMNLP2020"
    tags = ["creativity", "figurative_language", "simile", "style_transfer"]

    DATA_URL = "https://raw.githubusercontent.com/tuhinjubcse/SimileGeneration-EMNLP2020/master/SimileEMNLP.csv"

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str):
        # Download the data file
        data_path = os.path.join(output_path, "SimileEMNLP.csv")
        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(self.DATA_URL, data_path)

        instances = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                literal = row['Literal Sense']
                human1 = row['Human1']
                human2 = row['Human2']

                # Skip if no valid references
                if not human1 or human1.strip() == '------':
                    continue

                # Clean the literal sentence (remove brackets marking target word)
                literal_clean = literal.replace('[', '').replace(']', '')

                # Format prompt
                prompt = (
                    "Convert the following literal sentence into a simile using \"like\" or \"as\":\n\n"
                    f"Literal: {literal_clean}\n"
                    "Simile:"
                )

                # Build references from human-written similes
                references = []
                if human1 and human1.strip() != '------':
                    references.append(Reference(Output(text=human1.strip()), tags=[CORRECT_TAG]))
                if human2 and human2.strip() != '------':
                    references.append(Reference(Output(text=human2.strip()), tags=[CORRECT_TAG]))

                if references:
                    instances.append(Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=TEST_SPLIT
                    ))

        return instances
