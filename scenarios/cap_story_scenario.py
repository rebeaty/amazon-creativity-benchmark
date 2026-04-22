"""
HELM Scenario: CAP Story (Short Creative Stories from Word Triads)

Part of the CAP (Creativity Assessment Platform) battery used in the UVA
pilot / Study 3 validation. Prompts match the exact word triads
administered to human participants (n=298, single response per item).

LLM protocol: one creative 3-8 sentence story per triad. Single-response.

Item list reproduced verbatim from
[human/uva_pilot/scripts/run_all_models_study3.py].
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, TEST_SPLIT,
)


CAP_STORY_TRIADS = [
    "pen, paper, story",
    "key, door, lock",
    "bridge, river, cross",
    "mirror, face, reflection",
    "shoe, path, walk",
]


class CapStoryScenario(Scenario):
    name = "cap_story"
    description = "CAP Story — Short creative stories from word triads (UVA pilot Study 3 items)"
    tags = ["creativity", "cap", "narrative", "story"]

    def __init__(self, num_repetitions: int = 1):
        super().__init__()
        self.num_repetitions = max(1, int(num_repetitions))

    def get_instances(self, output_path) -> List[Instance]:
        instances: List[Instance] = []
        for rep in range(self.num_repetitions):
            for prompt_id, triad in enumerate(CAP_STORY_TRIADS):
                prompt = (
                    "In this task, you'll write a short creative story. You'll be given "
                    "three words that you must incorporate into your story. The goal is "
                    "to write a creative story that is novel (original, unique, "
                    "surprising) and effective (engaging, interesting, well-crafted).\n\n"
                    f"Write a short story (3-8 sentences) using these 3 words: {triad}. "
                    "Give ONLY the story, no title or explanation."
                )
                instances.append(Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"story_{prompt_id}_rep{rep}",
                    extra_data={
                        "task": "Story",
                        "prompt_id": prompt_id,
                        "triad": triad,
                        "rep": rep,
                    },
                ))
        return instances
