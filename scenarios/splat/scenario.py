"""
HELM Scenario: SPLAT (Situation Puzzles for Lateral Thinking)

Paper: https://arxiv.org/abs/2410.06733 (NeurIPS 2024)
Code: https://github.com/chenqi008/LateralThinking

Task: Solve lateral thinking puzzles by inferring the complete scenario from an
incomplete/mysterious story.

Prompt format: Simplified from official multi-turn interactive framework
  Original: Multi-turn Q&A game between player and judge
  HELM version: Direct inference task (single-turn)

Fields used: title, story (puzzle), answer (solution), level of difficulty
Fields skipped: None (all fields used)

Evaluation: open_ended generation
  - Reference: Official answer/explanation
  - Metrics: BLEU, ROUGE, semantic similarity
  - Could also use LLM-as-judge for reasoning quality

Note: The original paper proposes an interactive multi-turn framework where the
      model asks yes/no questions to gather clues. For HELM's single-turn format,
      we simplify to direct explanation generation, which still tests lateral
      thinking and creative reasoning.

Difficulty levels:
  - EASY: 1-3/10 (217 puzzles)
  - MEDIUM: 4-6/10 (648 puzzles)
  - HARD: 7-9/10 (110 puzzles)
"""

import os
import pandas as pd
from typing import Optional
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Reference, Output,
    CORRECT_TAG, TEST_SPLIT
)


class SPLATScenario(Scenario):
    name = "splat"
    description = "chenqi008/LateralThinking"
    tags = ["creativity", "lateral_thinking", "reasoning", "puzzle_solving"]

    def __init__(self, difficulty: str = "all"):
        """
        Args:
            difficulty: Filter by difficulty level
                - "all": All 975 puzzles
                - "easy": Difficulty 1-3/10
                - "medium": Difficulty 4-6/10
                - "hard": Difficulty 7-9/10
        """
        super().__init__()
        self.difficulty = difficulty

    def _get_difficulty_level(self, difficulty_str: str) -> Optional[str]:
        """Extract difficulty level from string like '5/10 MEDIUM'."""
        if pd.isna(difficulty_str):
            return None
        # Extract the category (EASY, MEDIUM, HARD)
        if "EASY" in str(difficulty_str):
            return "easy"
        elif "MEDIUM" in str(difficulty_str):
            return "medium"
        elif "HARD" in str(difficulty_str):
            return "hard"
        return None

    def _load_puzzles(self, repo_path: str) -> pd.DataFrame:
        """Load puzzles from Excel file."""
        file_path = os.path.join(repo_path, 'puzzles.xlsx')

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"SPLAT dataset not found at {file_path}. "
                f"Please clone the repo: git clone https://github.com/chenqi008/LateralThinking.git"
            )

        return pd.read_excel(file_path)

    def get_instances(self, output_path):
        # Try to find the repository in common locations
        possible_paths = [
            '/tmp/splat',
            os.path.join(output_path, '..', '..', '..', 'LateralThinking'),
            './LateralThinking',
            '../LateralThinking'
        ]

        repo_path = None
        for path in possible_paths:
            if os.path.exists(path):
                repo_path = path
                break

        if repo_path is None:
            raise FileNotFoundError(
                "SPLAT repository not found. Please clone it first:\n"
                "git clone https://github.com/chenqi008/LateralThinking.git"
            )

        # Load all puzzles
        df = self._load_puzzles(repo_path)

        # Filter by difficulty if specified
        if self.difficulty != "all":
            df = df[df['level of difficulty'].apply(
                lambda x: self._get_difficulty_level(x) == self.difficulty
            )]

        # Create instances
        instances = []
        for idx, row in df.iterrows():
            title = row['title']
            story = row['story']
            answer = row['answer']
            difficulty = row['level of difficulty']

            # Build prompt - simplified single-turn version
            prompt = f"""You are solving a lateral thinking puzzle. Read the mysterious story below and use creative reasoning to explain what really happened.

Puzzle: {title}

Story:
{story}

Task: Provide a complete explanation that resolves the mystery. Think creatively and consider unconventional interpretations.

Your explanation:"""

            # Reference is the official answer
            references = [Reference(
                Output(text=answer),
                tags=[CORRECT_TAG]
            )]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
