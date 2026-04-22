"""
HELM Scenario: CAP Design (Real-world Design Problems)

Part of the CAP (Creativity Assessment Platform) battery used in the UVA
pilot / Study 3 validation. Prompts match the exact items administered to
human participants (n=228, mean 3.71 solutions per participant per item).

LLM protocol: 3-6 distinct design solutions per prompt, semicolon-separated.
Post-hoc novelty splits on ``;``.

Item list reproduced verbatim from
[human/uva_pilot/scripts/run_all_models_study3.py].
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, TEST_SPLIT,
)


CAP_DESIGN_PROMPTS = [
    "Develop a design idea to assist people with memory impairments remember important tasks.",
    "Develop a design idea to help people with mobility impairments navigate stairs.",
    "Develop a design idea to reduce traffic congestion in mega cities.",
    "Develop a design idea to increase the use of renewable energy sources.",
    "Develop a design idea to improve access to clean water in remote areas.",
]

N_IDEAS_MIN = 3
N_IDEAS_MAX = 6


class CapDesignScenario(Scenario):
    name = "cap_design"
    description = "CAP Design — Real-world Design Problems (UVA pilot Study 3 items)"
    tags = ["creativity", "cap", "design_thinking", "design"]

    def __init__(self, num_repetitions: int = 1):
        super().__init__()
        self.num_repetitions = max(1, int(num_repetitions))

    def get_instances(self, output_path) -> List[Instance]:
        instances: List[Instance] = []
        for rep in range(self.num_repetitions):
            for prompt_id, stem in enumerate(CAP_DESIGN_PROMPTS):
                prompt = (
                    "In this task, you'll think of solutions to real-world design "
                    "problems. The goal is to come up with ideas that are novel (original, "
                    "unique, and innovative), while also being effective (practical, "
                    "efficient, and feasible).\n\n"
                    f"{stem}\n\n"
                    f"Give {N_IDEAS_MIN}-{N_IDEAS_MAX} distinct solutions, separated by "
                    "semicolons (;). Give only the solutions, no numbering or "
                    "explanation.\n\n"
                    "Answer:"
                )
                instances.append(Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"design_{prompt_id}_rep{rep}",
                    extra_data={
                        "task": "Design",
                        "prompt_id": prompt_id,
                        "stem": stem,
                        "rep": rep,
                    },
                ))
        return instances
