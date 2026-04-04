"""
HELM Scenario: Story Generation (ROCStories and WritingPrompts)

Paper: DeltaScore: Fine-Grained Story Evaluation with Perturbations
        Zhuohan Xie, Jey Han Lau, Alexander Gray Trevor Cohn
        EMNLP 2023
        https://arxiv.org/abs/2303.08991

Code: https://github.com/ZhuohanX/DeltaScore

Dataset: Adapted from DeltaScore's human evaluation dataset
  - ROCStories: 20 short story prompts (5-sentence stories)
  - WritingPrompts: 20 creative writing prompts (longer stories)
  - Total: 40 unique story generation tasks

Task: Generate a complete story given a prompt/title.
      ROCStories prompts are short contexts (e.g., "[FEMALE] and her friends were visiting las vegas.")
      WritingPrompts are more elaborate creative prompts (e.g., "Savour this sunset, gentlemen...")

Prompt format:
  Write a story based on the following prompt:

  {prompt}

  Story:

Evaluation:
  - Primary: Open-ended generation evaluated with BLEU, ROUGE against reference stories
  - Optional: Can be extended with human evaluation on 5 dimensions:
    * Fluency (grammatical correctness, readability)
    * Coherence (logical flow and consistency)
    * Relatedness (relevance to prompt)
    * Logicality (cause-effect relationships make sense)
    * Interestingness (engaging and compelling)

Fields used: title (prompt), reference (ground truth story)

Note: This dataset is adapted from DeltaScore's validation set, which originally contained
      pre-generated stories from multiple models with human ratings. We extract only the
      prompts and reference stories to create a story generation benchmark.

Original DeltaScore paper focused on validating evaluation metrics, not benchmarking LLMs.
This adaptation repurposes the data for LLM story generation evaluation.
"""

import json
import os
from typing import List, Optional
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
    CORRECT_TAG,
)
from helm.common.general import ensure_file_downloaded


class StoryGenerationScenario(Scenario):
    """
    Story generation benchmark using ROCStories and WritingPrompts from DeltaScore.

    Models are tasked with generating complete stories from prompts.
    """

    name = "story_generation"
    description = "ZhuohanX/DeltaScore"  # GitHub repo
    tags = ["creativity", "story-generation", "text-generation", "open-ended"]

    def __init__(self, dataset: str = "both"):
        """
        Args:
            dataset: Which dataset to use. Options: ["roc", "wp", "both"]
                    "roc" = ROCStories only (20 prompts)
                    "wp" = WritingPrompts only (20 prompts)
                    "both" = Both datasets (40 prompts total)
        """
        super().__init__()
        if dataset not in ["roc", "wp", "both"]:
            raise ValueError(f"Invalid dataset: {dataset}. Must be 'roc', 'wp', or 'both'")
        self.dataset = dataset

    def download_dataset(self, output_path: str) -> tuple:
        """Download the ROCStories and WritingPrompts data files."""
        base_url = "https://raw.githubusercontent.com/ZhuohanX/DeltaScore/master/data/crowdsource"

        roc_url = f"{base_url}/roc.jsonl"
        wp_url = f"{base_url}/wp.jsonl"

        roc_path = os.path.join(output_path, "roc.jsonl")
        wp_path = os.path.join(output_path, "wp.jsonl")

        ensure_file_downloaded(source_url=roc_url, target_path=roc_path)
        ensure_file_downloaded(source_url=wp_url, target_path=wp_path)

        return roc_path, wp_path

    def load_unique_prompts(self, file_path: str) -> List[dict]:
        """
        Load unique prompts from a dataset file.

        The file contains multiple model outputs per prompt, but we only need
        each unique prompt once with its reference story.
        """
        unique_prompts = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                title = item['title']
                if title not in unique_prompts:
                    unique_prompts[title] = {
                        'prompt': title,
                        'reference': item['reference']
                    }

        return list(unique_prompts.values())

    def create_prompt(self, story_prompt: str) -> str:
        """Create the prompt for story generation."""
        return (
            "Write a story based on the following prompt:\n\n"
            f"{story_prompt}\n\n"
            "Story:"
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate instances for story generation.

        Creates instances from ROCStories and/or WritingPrompts based on
        the selected dataset parameter.
        """
        # Download datasets
        roc_path, wp_path = self.download_dataset(output_path)

        instances = []
        instance_id = 0

        # Process ROCStories
        if self.dataset in ["roc", "both"]:
            roc_prompts = self.load_unique_prompts(roc_path)

            for prompt_data in roc_prompts:
                prompt_text = self.create_prompt(prompt_data['prompt'])

                # Reference is the human-written story
                references = [
                    Reference(Output(text=prompt_data['reference']), tags=[CORRECT_TAG])
                ]

                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"roc_{instance_id}"
                    )
                )
                instance_id += 1

        # Process WritingPrompts
        if self.dataset in ["wp", "both"]:
            wp_prompts = self.load_unique_prompts(wp_path)

            for prompt_data in wp_prompts:
                prompt_text = self.create_prompt(prompt_data['prompt'])

                references = [
                    Reference(Output(text=prompt_data['reference']), tags=[CORRECT_TAG])
                ]

                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"wp_{instance_id}"
                    )
                )
                instance_id += 1

        return instances
