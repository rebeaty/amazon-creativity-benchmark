"""
HELM Scenario: Arena-Hard v0.1

Paper: Arena-Hard-Auto (Li et al., 2024)
Code: https://github.com/lmarena/arena-hard-auto
Dataset: lmarena-ai/arena-hard-auto-v0.1

Arena-Hard v0.1 contains 500 challenging user queries from Chatbot Arena,
focusing on hard problems including coding, math, and creative tasks.

Note: v2.0 (750 prompts with 250 creative writing subset) mentioned in DARLING
paper may not be publicly released yet. This scenario implements v0.1.

Prompt format:
  {content}
  (multi-turn conversation format)

Example:
  "Use ABC notation to write a melody in the style of a folk tune."

Fields used: turns[0]['content'] (first turn prompt), cluster (topic category)
Fields skipped: question_id, category (all marked as 'arena-hard-v0.1')

Evaluation: LLM-as-judge comparing against baseline (typically GPT-4)
            Metric: Win Rate (higher = better)

Dataset source: lmarena-ai/arena-hard-auto-v0.1 on HuggingFace
Related: DARLING paper (arXiv:2509.02534) uses v2.0 with creative writing subset
"""

from typing import List
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
)


class ArenaHardV01Scenario(Scenario):
    """
    Arena-Hard v0.1 benchmark

    500 challenging real-world user queries spanning diverse topics organized
    into 250 clusters. Includes complex coding problems, advanced math, creative
    writing, and other difficult tasks sourced from Chatbot Arena.

    This is v0.1 with 500 examples. The DARLING paper references v2.0 (750 prompts
    with 250 creative writing) which may be a newer unreleased version.
    """

    name = "arena_hard_v01"
    description = "lmarena-ai/arena-hard-auto-v0.1"
    tags = ["creativity", "challenging", "diverse", "instruction_following"]

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load Arena-Hard v0.1 prompts.

        Each instance contains a challenging user query. Models generate responses
        which are compared against baseline (GPT-4) by an LLM judge.
        """

        # Load dataset from HuggingFace
        dataset = load_dataset("lmarena-ai/arena-hard-auto-v0.1")

        instances = []

        for item in dataset['train']:
            question_id = item['question_id']
            cluster = item['cluster']  # Topic category
            turns = item['turns']

            # Use first turn as the prompt (multi-turn not implemented yet)
            if len(turns) > 0:
                prompt_text = turns[0]['content']
            else:
                continue

            # References are empty - this is an LLM-judge benchmark
            references = []

            instances.append(
                Instance(
                    input=Input(text=prompt_text),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"arena_hard_{question_id}",
                )
            )

        return instances
