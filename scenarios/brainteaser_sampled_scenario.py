"""
HELM Scenario (SAMPLED MIRROR): BrainTeaser — subtask-decomposed +
reproducible 200-item sample for factor analysis / IRT work.

This is a mirror of [brainteaser_scenario.py] that:
  1. Exposes BrainTeaser's two HF configs (SP = Sentence Puzzle,
     WP = Word Puzzle) as separate evaluation units via a `subtask` arg.
  2. Applies the project-wide reproducible sampler ([scenarios/_sample.py])
     so every model run sees exactly the same items.

Prompting convention follows [analobench_scenario.py] — the in-repo
pattern that produces meaningful MCQ scores with ADAPT_GENERATION +
BasicGenerationMetric exact_match. The explicit "Only generate the
letter without any additional text" instruction is what reliably makes
chat-tuned models emit a single letter.

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

            # Prompt follows analobench pattern — explicit "only generate the letter"
            # forces chat-tuned models to emit a single letter token.
            options_block = "\n".join(
                f"{chr(65 + i)}. {choice}" for i, choice in enumerate(shuffled_choices)
            )
            prompt = (
                f"Which of the following is the correct answer to the puzzle below?\n\n"
                f"Note: Only generate the letter (A, B, C, or D) without any additional text.\n\n"
                f"Puzzle:\n"
                f"{item['question']}\n\n"
                f"Options:\n"
                f"{options_block}\n\n"
                f"Answer:"
            )

            correct_letter = chr(65 + correct_idx)
            references = [
                Reference(Output(text=chr(65 + i)),
                          tags=[CORRECT_TAG] if i == correct_idx else [])
                for i in range(4)
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
            ))

        return sampled(f"brainteaser_subtask={self.subtask}", instances)
