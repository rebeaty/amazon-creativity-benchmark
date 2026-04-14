"""
HELM Scenario: Javanese and Sundanese Story Cloze

Paper: "Culturally-Nuanced Story Generation for Low-Resource Languages:
        Javanese and Sundanese Story Cloze" (arXiv:2502.12932)
Dataset: rifoag/javanese_sundanese_story_cloze on HuggingFace

Task: Given a 4-sentence story premise in Javanese (jv) or Sundanese (su),
choose the correct ending from two options.

Prompt format (standard MC; no task prompt specified in paper):
  Read the following story and choose the most appropriate ending.

  Story: {sentence_1} {sentence_2} {sentence_3} {sentence_4}

  A) {ending_a}
  B) {ending_b}

  Answer:

Note: Correct ending alternates between A/B positions (by index) to avoid positional bias.

Config: human_written (native-authored stories; highest cultural fidelity)
Split: test (1,123 examples: 529 Javanese, 594 Sundanese)
Fields used: sentence_1–4 (premise), correct_ending, incorrect_ending, language
Fields skipped: topic, category (metadata only), generated_by (llm_generated config only)

Other configs available: llm_generated (train only), machine_translated (train+test)
Eval: exact_match (binary MC classification)
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class JavaneseSundaneseStoryClozeScenario(Scenario):
    name = "javanese_sundanese_story_cloze"
    description = "rifoag/javanese_sundanese_story_cloze"
    tags = ["creativity", "story_completion", "multilingual", "low_resource"]

    def get_instances(self, output_path):
        dataset = load_dataset(
            "rifoag/javanese_sundanese_story_cloze",
            "human_written",
            split="test"
        )

        instances = []
        for idx, item in enumerate(dataset):
            premise = (
                f"{item['sentence_1']} "
                f"{item['sentence_2']} "
                f"{item['sentence_3']} "
                f"{item['sentence_4']}"
            )

            # Alternate correct ending between A and B positions to avoid positional bias
            if idx % 2 == 0:
                choice_a = item["correct_ending"]
                choice_b = item["incorrect_ending"]
            else:
                choice_a = item["incorrect_ending"]
                choice_b = item["correct_ending"]

            prompt = (
                "Read the following story and choose the most appropriate ending.\n\n"
                f"Story: {premise}\n\n"
                f"A) {choice_a}\n"
                f"B) {choice_b}\n\n"
                "Answer:"
            )

            references = [
                Reference(
                    Output(text=choice_a),
                    tags=[CORRECT_TAG] if choice_a == item["correct_ending"] else [],
                ),
                Reference(
                    Output(text=choice_b),
                    tags=[CORRECT_TAG] if choice_b == item["correct_ending"] else [],
                ),
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
            ))

        return instances
