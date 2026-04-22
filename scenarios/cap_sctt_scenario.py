"""
HELM Scenario: CAP SCTT (Scientific Creative Thinking Task)

Part of the CAP (Creativity Assessment Platform) battery used in the UVA
pilot / Study 3 validation. Prompts match the exact items administered to
human participants (n=227, mean 4.42 responses per participant per item).

LLM protocol: 3-6 novel scientific hypotheses or questions per prompt,
semicolon-separated. Post-hoc novelty splits on ``;``.

Item list reproduced verbatim from
[human/uva_pilot/scripts/run_all_models_study3.py].
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, TEST_SPLIT,
)


CAP_SCTT_PROMPTS = [
    "You look outside one night and see stars disappearing one by one from the sky. What hypotheses do you have about why that is?",
    "You notice that the water in one lake is warmer than the water in another lake even though they both get the same amount of sunlight. What hypotheses do you have about why that is?",
    "You are introduced to a robot that can learn and think like humans. What scientific questions could you ask about this?",
    "You travel on a spaceship to a new planet outside of our galaxy. What scientific questions could you ask about this planet?",
    "You travel to a remote island and find people that do not communicate verbally. What scientific questions could you ask about the people of the island?",
]

N_IDEAS_MIN = 3
N_IDEAS_MAX = 6


class CapSCTTScenario(Scenario):
    name = "cap_sctt"
    description = "CAP SCTT — Scientific Creative Thinking Task (UVA pilot Study 3 items)"
    tags = ["creativity", "cap", "scientific_ideation", "sctt"]

    def __init__(self, num_repetitions: int = 1):
        super().__init__()
        self.num_repetitions = max(1, int(num_repetitions))

    def get_instances(self, output_path) -> List[Instance]:
        instances: List[Instance] = []
        for rep in range(self.num_repetitions):
            for prompt_id, stem in enumerate(CAP_SCTT_PROMPTS):
                prompt = (
                    "In this task, you'll think of creative ideas related to scientific "
                    "problems. The goal is to come up with ideas that are novel (original, "
                    "unique, and innovative), while also being scientifically possible "
                    "(practical, efficient, and feasible).\n\n"
                    f"{stem}\n\n"
                    f"Give {N_IDEAS_MIN}-{N_IDEAS_MAX} distinct ideas, separated by "
                    "semicolons (;). Give only the ideas, no numbering or explanation.\n\n"
                    "Answer:"
                )
                instances.append(Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"sctt_{prompt_id}_rep{rep}",
                    extra_data={
                        "task": "SCTT",
                        "prompt_id": prompt_id,
                        "stem": stem,
                        "rep": rep,
                    },
                ))
        return instances
