"""
HELM Scenario: Design Problems Task (DPT) — Creative Solution Generation

Paper: "Harnessing Large Language Models to Study Science and Engineering
        Creativity" (arXiv:2502.03253, CogSci 2025)
Authors: Patterson, Organisciak, Beaty et al. (Beaty Lab)
Code: https://github.com/Beaty-Lab/CogSci-2025-Scientific-Creativity

Task: Given a real-world STEM design challenge, generate a creative solution.
The benchmark assesses originality, cleverness, uncommonness (remoteness from
conventional solutions), and practical effectiveness — the core dimensions of
scientific and engineering creativity.

Dataset: 16 design problems across three domains (Accessibility, Transportation,
Environment/Sustainability), drawn from the paper's study instrument.
Crowd-validated via 80 STEM-expert human raters on Prolific (830 rated
responses with 5-dimension Likert scores available in the repo for judge
calibration).

Prompt instruction (adapted from paper study protocol, Section 2.1):
  "Think of a new, creative way to solve the following design problem.
   Your solution should be original — something that most people would not
   think of — and practically feasible. Describe your solution clearly in
   2–4 sentences."

Prompt source: Adapted from paper Section 2.1 study instructions.
Fields used: problem (16 unique STEM challenges, embedded in-code)
Fields skipped: response (human/LLM-generated solutions — model outputs)
               originality_rescaled_factor, effectiveness_rescaled_factor
               (human ratings — used for judge calibration only)
Evaluation: llm_judge (see annotator_notes.md)
  5 dimensions: originality, cleverness, uncommonness, effectiveness,
  conciseness — rated 1–5 by LLM judge; calibrated against 830 human ratings
Dataset: 16 instances (one per design problem; no download required)

Parameters:
  domain: "all" | "accessibility" | "transportation" | "environment"
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

# 16 design problems verbatim from the paper's study instrument
# Grouped by the three domains used in the paper (arXiv:2502.03253, Section 2)
_PROBLEMS_BY_DOMAIN = {
    "accessibility": [
        "Assist people with learning impairments to retain information.",
        "Assist people with memory impairments to remember important tasks.",
        "Help people with hearing impairments participate in group conversations.",
        "Help people with mobility impairments navigate stairs.",
        "Help people with speech impairments communicate effectively.",
    ],
    "transportation": [
        "Improve the efficiency of public transportation systems.",
        "Improve the safety of pedestrian crossings.",
        "Reduce traffic congestion in mega cities.",
    ],
    "environment": [
        "Improve access to clean water in remote areas.",
        "Increase the use of renewable energy sources.",
        "Reduce air pollution in cities.",
        "Reduce the amount of litter in public spaces and promote waste reduction and recycling.",
        "Reduce the amount of single-use plastic packaging used in retail products.",
        "Reduce the carbon footprint of office buildings.",
        "Reduce the environmental impact of air travel.",
        "Reduce the risk of accidents caused by distracted driving.",
    ],
}

_DOMAIN_LABELS = {
    "accessibility": "Accessibility & Ability Differences",
    "transportation": "Transportation & Mobility",
    "environment": "Environment & Sustainability",
}

_VALID_DOMAINS = list(_PROBLEMS_BY_DOMAIN.keys()) + ["all"]

# Instruction adapted from paper Section 2.1 study protocol
_INSTRUCTION = (
    "Think of a new, creative way to solve the following design problem. "
    "Your solution should be original — something that most people would not "
    "think of — and practically feasible. Describe your solution clearly in "
    "2–4 sentences."
)

_PROMPT_TEMPLATE = "{instruction}\n\nDesign problem: {problem}"


class DptScenario(Scenario):
    """
    Design Problems Task — creative STEM solution generation.

    16 real-world design challenges across accessibility, transportation, and
    environmental sustainability. The model generates a short creative solution
    for each problem, evaluated by LLM judge on originality, cleverness,
    uncommonness, effectiveness, and conciseness.

    Calibration data: 830 human-rated responses available in the paper's
    GitHub repo for judge calibration. See annotator_notes.md.
    """

    name = "dpt"
    description = "github.com/Beaty-Lab/CogSci-2025-Scientific-Creativity (arXiv:2502.03253)"
    tags = ["creativity", "design_thinking", "stem", "open_ended_generation", "scientific_creativity"]

    def __init__(self, domain: str = "all"):
        super().__init__()
        if domain not in _VALID_DOMAINS:
            raise ValueError(
                f"Unknown domain: {domain!r}. Must be one of {_VALID_DOMAINS}"
            )
        self.domain = domain

    def get_instances(self, output_path: str) -> List[Instance]:
        active_domains = (
            list(_PROBLEMS_BY_DOMAIN.keys())
            if self.domain == "all"
            else [self.domain]
        )

        instances = []
        for domain in active_domains:
            for problem in _PROBLEMS_BY_DOMAIN[domain]:
                prompt = _PROMPT_TEMPLATE.format(
                    instruction=_INSTRUCTION,
                    problem=problem,
                )

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=[],   # LLM-as-judge; no single correct answer
                        split=TEST_SPLIT,
                    )
                )

        return instances  # 16 instances total (5 + 3 + 8)
