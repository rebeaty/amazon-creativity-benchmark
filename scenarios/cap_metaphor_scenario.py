"""
HELM Scenario: CAP Metaphor (Metaphor Completion)

Part of the CAP (Creativity Assessment Platform) battery used in the UVA
pilot / Study 3 validation. Prompts match the exact items administered to
human participants. Humans produced ~1 response per (participant, item)
— single-response protocol.

LLM protocol: one creative short-phrase completion per stem (1-5 words).
No semicolon splitting downstream; each generation is a single metaphor.

Item list reproduced verbatim from
[human/uva_pilot/scripts/run_all_models_study3.py].
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, TEST_SPLIT,
)


CAP_METAPHOR_STEMS = [
    "The strong wind is",
    "The soft pillow is",
    "The shiny coin is",
    "The clean window is",
    "The heavy rain is",
    "The hot tea is",
    "The bright light is",
    "The fresh fruit is",
    "The dark night is",
    "The young child is",
]


class CapMetaphorScenario(Scenario):
    name = "cap_metaphor"
    description = "CAP Metaphor — Metaphor Completion (UVA pilot Study 3 items)"
    tags = ["creativity", "cap", "figurative_language", "metaphor"]

    def __init__(self, num_repetitions: int = 1):
        super().__init__()
        self.num_repetitions = max(1, int(num_repetitions))

    def get_instances(self, output_path) -> List[Instance]:
        instances: List[Instance] = []
        for rep in range(self.num_repetitions):
            for prompt_id, stem in enumerate(CAP_METAPHOR_STEMS):
                prompt = (
                    "In this task, you'll complete metaphorical comparisons. A METAPHOR "
                    "is a word or phrase that creatively compares one thing to another. "
                    "When thinking of a metaphor, the aim is to come up with a creative "
                    "response: something clever, humorous, or original.\n\n"
                    "Complete this metaphor with a short creative phrase (1-5 words). "
                    "Give ONLY the completion, no explanation.\n\n"
                    f"{stem}"
                )
                instances.append(Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"metaphor_{prompt_id}_rep{rep}",
                    extra_data={
                        "task": "Metaphor",
                        "prompt_id": prompt_id,
                        "stem": stem,
                        "rep": rep,
                    },
                ))
        return instances
