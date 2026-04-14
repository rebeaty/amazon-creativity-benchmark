"""
HELM Scenario: EQBench Creative Writing v3

Author: Samuel J. Paech, 2023
Code: https://github.com/EQ-bench/creative-writing-bench
Leaderboard: https://eqbench.com/creative_writing_longform.html

EQBench Creative Writing v3 evaluates LLMs on 32 challenging creative writing prompts
across diverse genres. Models generate 3 iterations per prompt (96 total outputs) using
temperature=0.7 and min_p=0.1.

Prompts emphasize difficult creative challenges including humor, romance, unusual
perspectives, and complex character development.

Prompt format:
  {writing_prompt}
  (Plain prompt with optional seed modifiers)

Example:
  "Historical Fiction: Write a scene from a story set during the height of the Roman
   Empire, focusing on a slice of a day in the life of a gladiator..."

Fields used: writing_prompt, seed_modifiers (optional variations), category, title
Fields skipped: None

Evaluation: LLM-as-judge (Claude Sonnet 4) with pairwise comparisons
            Metrics: Elo ratings, rubric scores
            Criteria: character authenticity, originality, plot coherence,
                     emotional engagement, prose quality

Dataset source: https://raw.githubusercontent.com/EQ-bench/creative-writing-bench/main/data/creative_writing_prompts_v3.json
Used in: DARLING paper (arXiv:2509.02534) for creative writing evaluation
"""

from typing import List
import json
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class EQBenchCreativeWritingV3Scenario(Scenario):
    """
    EQBench Creative Writing v3 benchmark

    Evaluates creative writing across 32 challenging prompts spanning multiple genres:
    - Historical fiction
    - Science fiction
    - Fantasy
    - Romance
    - Humor
    - Literary fiction
    - Horror

    Each prompt includes base instructions and optional seed modifiers for variation.
    Models typically generate 3 iterations per prompt with controlled randomness.
    """

    name = "eqbench_creative_writing_v3"
    description = "EQ-bench/creative-writing-bench"
    tags = ["creativity", "creative_writing", "long_form", "diverse_genres"]

    DATASET_DOWNLOAD_URL = "https://raw.githubusercontent.com/EQ-bench/creative-writing-bench/main/data/creative_writing_prompts_v3.json"

    def __init__(self, num_iterations: int = 3):
        """
        Args:
            num_iterations: Number of times to generate for each prompt (default 3 per benchmark)
        """
        super().__init__()
        self.num_iterations = num_iterations

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load EQBench Creative Writing v3 prompts and create instances.

        Each of the 32 prompts is repeated num_iterations times (default 3)
        to create 96 total instances. Generation uses temperature=0.7 and min_p=0.1
        to encourage creativity while maintaining coherence.
        """

        # Download dataset
        data_path = os.path.join(output_path, "creative_writing_prompts_v3.json")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            unpack=False,
        )

        # Load JSON data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instances = []

        # Data is dict with numbered string keys ('1', '2', etc.)
        for prompt_id in sorted(data.keys(), key=lambda x: int(x)):
            prompt_data = data[prompt_id]

            category = prompt_data['category']
            title = prompt_data['title']
            writing_prompt = prompt_data['writing_prompt']
            seed_modifiers = prompt_data.get('seed_modifiers', [])

            # Create num_iterations instances for each prompt
            for iteration in range(self.num_iterations):
                # Base prompt (no modifiers applied in scenario - done at generation time)
                prompt_text = writing_prompt

                # References are empty for open-ended creative writing
                # Evaluation uses LLM judge comparing pairwise, not reference-based
                references = []

                instance_id = f"eqbench_cw_v3_{prompt_id}_iter{iteration+1}"

                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=references,
                        split=TEST_SPLIT,
                        id=instance_id,
                    )
                )

        return instances
