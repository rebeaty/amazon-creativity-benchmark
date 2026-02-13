"""
HELM Scenario: PoetMT (Classical Chinese Poetry Translation)

Paper: https://arxiv.org/abs/2408.09945
Code: https://github.com/andongBlue/PoetMT
Published: EMNLP 2025

Task: Translate classical Chinese poems into English while preserving:
  - Adequacy (信): Accurate meaning and cultural faithfulness
  - Fluency (达): Smooth rhythm and structural alignment
  - Elegance (雅): Poetic beauty and aesthetic depth

This is a creativity benchmark focused on the "elegance" dimension - the ability
to produce poetically beautiful, creative translations that preserve aesthetic value.

Prompt format:
  Translate the following classical Chinese poem into English:

  {chinese_poem}

  Translation:

Evaluation: Open-ended generation (BLEU, ROUGE, F1) + custom GPT-4 metrics
for Beauty of Sound (BS), Beauty of Form (BF), Beauty of Meaning (BM).
See metric_notes.md for GPT-4 evaluation implementation.

Fields used: src (Chinese poem), ref (reference English translation)
Fields skipped: note, english_note (annotations), title, author (metadata)

Dataset: 790 classical Chinese poems
  - Tang Dynasty: 295 poems
  - Song Dynasty: 196 poems
  - Yuan Dynasty: 299 poems
"""

import json
import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)


class PoetMTScenario(Scenario):
    """
    PoetMT: Classical Chinese Poetry Translation Benchmark

    Evaluates translation quality across three dimensions:
    adequacy, fluency, and elegance (creative expression).
    """

    name = "poetmt"
    description = "andongBlue/PoetMT"
    tags = ["creativity", "translation", "poetry"]

    # Poetry dynasties available in the dataset
    VALID_DYNASTIES = ["tang", "song", "yuan", "all"]

    def __init__(self, dynasty: str = "all"):
        """
        Args:
            dynasty: Which dynasty to evaluate. Options:
                - "tang": Tang Dynasty poems (295 examples)
                - "song": Song Dynasty poems (196 examples)
                - "yuan": Yuan Dynasty poems (299 examples)
                - "all": All dynasties combined (790 examples)
        """
        super().__init__()
        if dynasty not in self.VALID_DYNASTIES:
            raise ValueError(
                f"Invalid dynasty '{dynasty}'. Must be one of: {self.VALID_DYNASTIES}"
            )
        self.dynasty = dynasty

    def _download_data(self, output_path: str) -> str:
        """
        Download PoetMT dataset from GitHub if not already present.

        Returns:
            Path to the data directory
        """
        data_dir = os.path.join(output_path, "poetmt_data")

        # Check if data already exists
        if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
            print(f"Data already exists at {data_dir}")
            return data_dir

        # Clone the repository
        import subprocess
        print("Downloading PoetMT dataset from GitHub...")
        os.makedirs(output_path, exist_ok=True)

        repo_url = "https://github.com/andongBlue/PoetMT.git"
        subprocess.run(
            ["git", "clone", repo_url, data_dir],
            check=True,
            capture_output=True
        )
        print("Download complete")

        return data_dir

    def _load_poems(self, data_dir: str, dynasty: str) -> List[dict]:
        """
        Load poems from JSONL files for specified dynasty.

        Args:
            data_dir: Path to the data directory
            dynasty: Dynasty name ("tang", "song", "yuan")

        Returns:
            List of poem dictionaries
        """
        file_path = os.path.join(data_dir, "all_poems", f"{dynasty}.jsonl")

        poems = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                poems.append(json.loads(line.strip()))

        return poems

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate PoetMT instances for the specified dynasty.

        Each instance contains:
        - Input: Classical Chinese poem with translation prompt
        - Reference: Expert English translation
        """
        # Download data if needed
        data_dir = self._download_data(output_path)

        # Load poems based on dynasty selection
        all_poems = []
        if self.dynasty == "all":
            for dynasty in ["tang", "song", "yuan"]:
                all_poems.extend(self._load_poems(data_dir, dynasty))
        else:
            all_poems = self._load_poems(data_dir, self.dynasty)

        instances = []
        for idx, poem in enumerate(all_poems):
            # Extract fields
            chinese_poem = poem['src']
            reference_translation = poem['ref']

            # Skip if no source or reference (though all should have both)
            if not chinese_poem or not reference_translation:
                continue

            # Build prompt
            prompt = (
                "Translate the following classical Chinese poem into English:\n\n"
                f"{chinese_poem}\n\n"
                "Translation:"
            )

            # Create reference
            references = [
                Reference(
                    Output(text=reference_translation),
                    tags=[CORRECT_TAG]
                )
            ]

            # Create instance
            instance_id = f"{self.dynasty}_{idx}"
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=instance_id
                )
            )

        return instances
