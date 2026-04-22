"""
HELM Scenario: CAP AUT (Alternative Uses Task)

Part of the CAP (Creativity Assessment Platform) battery used in the UVA
pilot / Study 3 validation. Prompts match the exact items administered to
human participants so LLM scores are directly comparable on item-level
(join on ``prompt_id``).

Human protocol: 227 participants, fluency-style. Mean 5.47 ideas per
(participant, item). See
[human/uva_pilot/llm_data/study3_results/human_cap_aut.csv] for the raw
human data.

LLM protocol (this scenario): multi-response per prompt. Model is asked to
produce 3-6 creative uses in one generation, separated by semicolons. The
post-hoc novelty aggregator ([scripts/score_cap_novelty.py]) splits on
``;`` to recover per-idea embeddings.

Prompt + item list reproduced verbatim from
[human/uva_pilot/scripts/run_all_models_study3.py].
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, TEST_SPLIT,
)


CAP_AUT_PROMPTS = [
    "BRICK",
    "KNIFE",
    "PENCIL",
    "BUCKET",
    "SOCK",
]

N_IDEAS_MIN = 3
N_IDEAS_MAX = 6


class CapAutScenario(Scenario):
    name = "cap_aut"
    description = "CAP AUT — Alternative Uses Task (UVA pilot Study 3 items)"
    tags = ["creativity", "cap", "divergent_thinking", "aut"]

    def __init__(self, num_repetitions: int = 1):
        super().__init__()
        self.num_repetitions = max(1, int(num_repetitions))

    def get_instances(self, output_path) -> List[Instance]:
        instances: List[Instance] = []
        for rep in range(self.num_repetitions):
            for prompt_id, obj in enumerate(CAP_AUT_PROMPTS):
                prompt = (
                    "In this task, you'll think of creative uses for everyday objects. "
                    "The goal is to come up with creative ideas that are clever, unusual, "
                    "interesting, uncommon, humorous, innovative, or different from the "
                    "object's typical use.\n\n"
                    f"List {N_IDEAS_MIN}-{N_IDEAS_MAX} creative uses for a {obj}. "
                    "Separate each idea with a semicolon (;). Give only the ideas, no "
                    "numbering or explanation.\n\n"
                    "Answer:"
                )
                instances.append(Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"aut_{prompt_id}_rep{rep}",
                    extra_data={
                        "task": "AUT",
                        "prompt_id": prompt_id,
                        "object": obj,
                        "rep": rep,
                    },
                ))
        return instances
