"""
HELM Scenario: Creative Process (Verbal Fluency + AUT)

Paper: "Characterising the Creative Process in Humans and Large Language
        Models" (ICCC 2024)
       arXiv: 2405.00899
Code: https://github.com/surabhisnath/Creative_Process

Three creativity tasks measuring semantic space exploration:

  - vf: Verbal Fluency — generate 30 animals.
  - aut_brick: Alternative Uses Test — 30 creative uses for a brick.
  - aut_paperclip: Alternative Uses Test — 20 creative uses for a paperclip.

Each task is a single prompt repeated num_instances times (default 5,
matching the paper's 5 repetitions per model-temperature combo). With
non-zero temperature, each instance produces different output, enabling
measurement of creative process diversity.

The paper's innovation is process analysis: examining HOW models explore
semantic space (jump profiles, clustering patterns) via sentence embeddings,
not just WHAT they produce. Human baseline: 219 participants.

Prompt source: Exact prompts from scripts_LLM/together_call.py in the repo.

Fields used: None (prompts are fixed, not from a dataset)

Evaluation: Custom metrics. See metric_notes.md.
  - Semantic similarity chains (cosine of gte-large embeddings)
  - Category clustering (hierarchical, cutoff=1.14 for VF)
  - Jump profiles (cumulative count of semantic space transitions)
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    TEST_SPLIT,
)

# Exact prompts from scripts_LLM/together_call.py

VF_PROMPT = (
    "Think of 30 animals.\n"
    "Give your responses as a comma-separated list in square braces '[]'. "
    "For example, if your animals are a,b,c,d,.., output: [a,b,c,d,..]\n"
    "Do not give any other words or text.\n\n"
    "Output: ["
)

AUT_BRICK_PROMPT = (
    "Think of 30 creative uses for 'brick'.\n"
    "Answer in short phrases of 1 to 5 words. "
    "Give your responses as a comma-separated list in square braces '[]'. "
    "For example, if your creative uses are a,b,c,d,.., output: [a,b,c,d,..]\n"
    "Do not give any other words or text.\n\n"
    "Output: ["
)

AUT_PAPERCLIP_PROMPT = (
    "Think of 20 creative uses for 'paperclip'.\n"
    "Answer in short phrases of 1 to 5 words. "
    "Give your responses as a comma-separated list in square braces '[]'. "
    "For example, if your creative uses are a,b,c,d,.., output: [a,b,c,d,..]\n"
    "Do not give any other words or text.\n\n"
    "Output: ["
)


class CreativeProcessScenario(Scenario):
    name = "creative_process"
    description = "surabhisnath/Creative_Process"
    tags = ["creativity", "divergent_thinking", "verbal_fluency"]

    TASK_PROMPTS = {
        "vf": VF_PROMPT,
        "aut_brick": AUT_BRICK_PROMPT,
        "aut_paperclip": AUT_PAPERCLIP_PROMPT,
    }

    def __init__(self, task: str = "vf", num_instances: int = 5):
        """
        Args:
            task: One of "vf", "aut_brick", "aut_paperclip".
            num_instances: Number of instances (repetitions) per task.
                Default 5, matching the paper's 5 reps per model-temp combo.
        """
        super().__init__()
        task = task.lower()
        if task not in self.TASK_PROMPTS:
            raise ValueError(
                f"task must be one of {list(self.TASK_PROMPTS)}, got '{task}'"
            )
        self.task = task
        self.num_instances = num_instances

    def get_instances(self, output_path: str) -> List[Instance]:
        prompt_text = self.TASK_PROMPTS[self.task]
        instances = []
        for i in range(self.num_instances):
            instances.append(
                Instance(
                    input=Input(text=prompt_text),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"{self.task}_{i}",
                )
            )
        return instances
