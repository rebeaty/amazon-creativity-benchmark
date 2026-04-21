"""
HELM Scenario (SAMPLED MIRROR): CS4 — subtask-split + reproducible 200-item
sample for factor analysis / IRT work.

Wraps the original [cs4_scenario.CS4Scenario] which already accepts a
``dataset_type`` argument ("instruction" | "story" | "both"). This mirror:
  1. Exposes the two types as separate evaluation units (disallows "both").
  2. Applies the project-wide reproducible sampler ([scenarios/_sample.py])
     so every model run sees exactly the same 200 items per subtask.

The original `cs4_scenario.py` is intentionally left untouched.

Paper: https://arxiv.org/abs/2410.04197 (October 2024)
Code:  https://github.com/anirudhlakkaraju/cs4_benchmark
"""

from typing import List

from helm.benchmark.scenarios.scenario import Instance

from scenarios.cs4_scenario import CS4Scenario
from scenarios._sample import sampled


_ALLOWED = ("instruction", "story")


class CS4SampledScenario(CS4Scenario):
    name = "cs4_sampled"
    description = "CS4 subtask-split + 200-item sampled mirror"
    tags = ["creativity", "generation", "story", "constraints", "sampled"]

    def __init__(self, dataset_type: str):
        if dataset_type not in _ALLOWED:
            raise ValueError(
                f"dataset_type must be one of {_ALLOWED}, got {dataset_type!r}"
            )
        super().__init__(dataset_type=dataset_type)

    def get_instances(self, output_path: str) -> List[Instance]:
        all_instances = super().get_instances(output_path)
        return sampled(f"cs4_subtask={self.dataset_type}", all_instances)
