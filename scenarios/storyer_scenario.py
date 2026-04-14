"""
HELM Scenario: StoryER (Automatic Story Evaluation via Ranking, Rating and Reasoning)

Paper: https://arxiv.org/abs/2210.08459 (EMNLP 2022)
Code: https://github.com/sairin1202/StoryER

Task: Story evaluation across three sub-tasks:
  1. Ranking - Preference scoring between story pairs
  2. Rating - Aspect-based quality ratings (10 dimensions) with confidence scores
  3. Reasoning - Natural language comment generation explaining evaluations

This is a meta-evaluation benchmark that assesses story quality rather than generating
stories. Models learn to predict human judgments about narrative quality across multiple
dimensions including opening, character development, plot, emotion, dialogue, setting,
pacing, and resolution.

Dataset composition:
  - 100k ranked story pairs for preference comparison
  - 46k aspect ratings with confidence scores and comments
  - 10 story aspects evaluated per instance
  - Datasets: WritingPrompts and ScaryStories corpora

Evaluation metrics:
  - Ranking: Accuracy, mean distance between high/low stories
  - Rating: Correlation coefficients (Spearman, Pearson, Kendall tau)
  - Reasoning: Generation quality of explanatory comments

Prompt format:
  Task 1 (Ranking):
    [ranking_prompt]
    Story A: {high_story}
    Story B: {low_story}
    Which story is better?

  Task 2 (Rating):
    [rating_prompt]
    Aspect: {aspect_name}
    Story: {story}
    Rating (0-1):

  Task 3 (Reasoning):
    [reasoning_prompt]
    Aspect: {aspect_name}
    Story: {story}
    Comment:

Prompt source: dataset.py (prompts embedded in data files)
Fields used: high_story, low_story, target (ranking); aspect_story, aspect_rate, comment (rating/reasoning)
Fields skipped: margin, aspect (numeric index), prompt fields (used internally)

Note: Dataset is hosted on Google Drive (not HuggingFace). Manual download required.
This scenario implements Task 3 (Reasoning/Comment Generation) as the primary creative
task, as Tasks 1-2 are evaluative rather than generative.
"""

import os
import json
import pickle
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT, TRAIN_SPLIT
)


class StoryERScenario(Scenario):
    name = "storyer"
    description = "sairin1202/StoryER"
    tags = ["creativity", "story_evaluation", "comment_generation", "meta_evaluation"]

    # 10 story aspect categories from the paper
    ASPECTS = [
        "opening",
        "character_shaping",
        "humor",
        "plot_structure",
        "emotional_depth",
        "dialogue_quality",
        "setting_worldbuilding",
        "tension_pacing",
        "resolution_quality",
        "narrative_coherence"
    ]

    GOOGLE_DRIVE_URLS = {
        "ranking": "https://drive.google.com/drive/folders/1DpPyFVAEOS59E5wC9Ob-0ZV8p8sfNb7J?usp=sharing",
        "rating_reasoning": "https://drive.google.com/file/d/1RXPa64vQSAvf7ZbRUeofkkeesPIK3TCo/view?usp=sharing"
    }

    def __init__(self, task: str = "reasoning", subset: str = "test"):
        """
        Args:
            task: Which StoryER task to evaluate
                  - "ranking": Preference between story pairs (Task 1)
                  - "rating": Aspect-based quality scores (Task 2)
                  - "reasoning": Comment generation (Task 3) - primary creative task
            subset: "train" or "test"
        """
        super().__init__()
        if task not in ["ranking", "rating", "reasoning"]:
            raise ValueError(f"task must be 'ranking', 'rating', or 'reasoning', got '{task}'")
        if subset not in ["train", "test"]:
            raise ValueError(f"subset must be 'train' or 'test', got '{subset}'")

        self.task = task
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load StoryER instances from local files.

        Note: Dataset must be downloaded manually from Google Drive:
        - Ranking data: {GOOGLE_DRIVE_URLS["ranking"]}
        - Rating/Reasoning data: {GOOGLE_DRIVE_URLS["rating_reasoning"]}

        Expected file structure in output_path:
        - test_rank_data.pkl (33.5 MB)
        - test_rate_reason_data.json (7.2 MB)
        - train_rank_data_small.pkl (1.9 MB)
        - train_rate_reason_data_small.json (381 KB)
        """
        instances = []

        if self.task == "ranking":
            instances = self._load_ranking_instances(output_path)
        elif self.task in ["rating", "reasoning"]:
            instances = self._load_rating_reasoning_instances(output_path)

        return instances

    def _load_ranking_instances(self, output_path: str) -> List[Instance]:
        """Load Task 1 (Ranking) instances - preference between story pairs."""
        filename = f"{self.subset}_rank_data{'_small' if self.subset == 'train' else ''}.pkl"
        filepath = os.path.join(output_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Ranking data file not found: {filepath}\n"
                f"Please download from: {self.GOOGLE_DRIVE_URLS['ranking']}"
            )

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instances = []
        split = TRAIN_SPLIT if self.subset == "train" else TEST_SPLIT

        for item in data:
            # Extract ranking pair
            high_story = item.get('high_story', '')
            low_story = item.get('low_story', '')
            target = item.get('target', 1)  # Binary preference (high > low)

            # Format prompt
            prompt = (
                "Compare the following two stories and determine which is better.\n\n"
                f"Story A:\n{high_story}\n\n"
                f"Story B:\n{low_story}\n\n"
                "Which story is better? Answer with 'A' or 'B'."
            )

            # References: A (high_story) is correct
            references = [
                Reference(Output(text="A"), tags=[CORRECT_TAG]),
                Reference(Output(text="B"), tags=[])
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=split
            ))

        return instances

    def _load_rating_reasoning_instances(self, output_path: str) -> List[Instance]:
        """Load Task 2 (Rating) and Task 3 (Reasoning) instances."""
        filename = f"{self.subset}_rate_reason_data{'_small' if self.subset == 'train' else ''}.json"
        filepath = os.path.join(output_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Rating/Reasoning data file not found: {filepath}\n"
                f"Please download from: {self.GOOGLE_DRIVE_URLS['rating_reasoning']}"
            )

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instances = []
        split = TRAIN_SPLIT if self.subset == "train" else TEST_SPLIT

        for item in data:
            # Extract fields
            story = item.get('story', '')
            aspect_idx = item.get('aspect', 0)
            aspect_name = self.ASPECTS[aspect_idx] if aspect_idx < len(self.ASPECTS) else f"aspect_{aspect_idx}"
            aspect_rate = item.get('aspect_rate', 0.5)  # 0-1 normalized rating
            comment = item.get('comment', '')

            if self.task == "rating":
                # Task 2: Predict aspect rating
                prompt = (
                    f"Rate the following story on the aspect: {aspect_name}\n"
                    f"Provide a rating between 0 (poor) and 1 (excellent).\n\n"
                    f"Story:\n{story}\n\n"
                    f"Rating for {aspect_name}:"
                )

                # Reference is the normalized rating value (preserve full precision)
                references = [
                    Reference(Output(text=repr(aspect_rate)), tags=[CORRECT_TAG])
                ]

            else:  # self.task == "reasoning"
                # Task 3: Generate explanatory comment
                prompt = (
                    f"Evaluate the following story on the aspect: {aspect_name}\n"
                    f"Provide a detailed comment explaining your evaluation.\n\n"
                    f"Story:\n{story}\n\n"
                    f"Comment on {aspect_name}:"
                )

                # Reference is the human-written comment
                references = [
                    Reference(Output(text=comment), tags=[CORRECT_TAG])
                ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=split
            ))

        return instances
