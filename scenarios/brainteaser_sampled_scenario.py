"""
HELM Scenario (SAMPLED MIRROR): BrainTeaser — subtask-decomposed +
reproducible 200-item sample for factor analysis / IRT work.

This is a mirror of [brainteaser_scenario.py] that:
  1. Exposes BrainTeaser's two HF configs (SP = Sentence Puzzle,
     WP = Word Puzzle) as separate evaluation units via a `subtask` arg.
  2. Applies the project-wide reproducible sampler ([scenarios/_sample.py])
     so every model run sees exactly the same items.

The original `brainteaser_scenario.py` is intentionally left untouched.
Any existing pipelines that rely on the onboarded `brainteaser` scenario
(single SP pool, no sampling) continue to work.

Paper: https://arxiv.org/abs/2310.05057 (EMNLP 2023)
Code:  https://github.com/1171-jpg/BrainTeaser
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

from scenarios._sample import sampled


_SUBTASK_TO_CONFIG = {
    "sentence_puzzle": "SP",
    "word_puzzle": "WP",
}


class BrainteaserSampledScenario(Scenario):
    name = "brainteaser_sampled"
    description = "tasksource/brainteasers — subtask-split + 200-item sampled mirror"
    tags = ["creativity", "lateral_thinking", "multiple_choice", "sampled"]

    def __init__(self, subtask: str):
        super().__init__()
        if subtask not in _SUBTASK_TO_CONFIG:
            raise ValueError(
                f"subtask must be one of {list(_SUBTASK_TO_CONFIG)}, got {subtask!r}"
            )
        self.subtask = subtask

    def get_instances(self, output_path):
        config = _SUBTASK_TO_CONFIG[self.subtask]
        dataset = load_dataset("tasksource/brainteasers", config, split="train")

        instances = []
        for item in dataset:
            original_choices = [
                item["answer"],
                item["distractor1"],
                item["distractor2"],
                item["distractor(unsure)"],
            ]
            choice_order = item["choice_order"]
            shuffled_choices = [original_choices[i] for i in choice_order]
            correct_idx = item["label"]

            prompt = f"Question: {item['question']}\n"
            for i, choice in enumerate(shuffled_choices):
                prompt += f"\n{chr(65 + i)}. {choice}"

            references = []
            for i in range(4):
                letter = chr(65 + i)
                is_correct = (i == correct_idx)
                tags = [CORRECT_TAG] if is_correct else []
                references.append(Reference(Output(text=letter), tags=tags))

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
            ))

        return sampled(f"brainteaser_subtask={self.subtask}", instances)
