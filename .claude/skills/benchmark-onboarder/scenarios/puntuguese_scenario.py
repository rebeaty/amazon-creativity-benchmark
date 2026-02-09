"""
HELM Scenario: Puntuguese

Paper: https://aclanthology.org/2024.lrec-main.1167/ (LREC-COLING 2024)
Dataset: Superar/Puntuguese (HuggingFace)

Task: Binary humor recognition for Portuguese puns. Given a Portuguese text (joke or
micro-edited non-funny version), classify whether it is humorous or not.

Corpus: 4,903 manually-gathered punning one-liners in Brazilian and European Portuguese.
Non-humorous examples created through micro-editing by fluent Portuguese speakers.

Prompt format: Standard binary classification
  Text: {text}
  Is this text humorous? (Yes/No)

Fields used: text, label (1=humorous, 0=non-humorous)
Fields skipped: tokens, labels (token-level pun location annotations; auxiliary task)

Evaluation: exact_match
  - Metric: Accuracy / F1-score
  - Paper reports: 68.9% F1-score baseline

Note: The dataset also includes token-level pun location annotations, but the primary
      creativity task evaluated in the paper is humor recognition.
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class PuntugueseScenario(Scenario):
    name = "puntuguese"
    description = "Superar/Puntuguese"
    tags = ["creativity", "humor", "puns", "portuguese", "multilingual"]

    def get_instances(self, output_path):
        # Load test split
        dataset = load_dataset("Superar/Puntuguese", split="test")

        instances = []
        for item in dataset:
            text = item['text']
            label = item['label']  # 1 = humorous, 0 = non-humorous

            # Build prompt - standard binary classification
            prompt = f"""Text: {text}

Is this text humorous?"""

            # Binary classification: both Yes and No are references
            # Tag the correct answer based on label
            references = [
                Reference(
                    output={"text": "Yes"},
                    tags=[CORRECT_TAG] if label == 1 else []
                ),
                Reference(
                    output={"text": "No"},
                    tags=[CORRECT_TAG] if label == 0 else []
                )
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
