"""
HELM Scenario: Vietnamese Poem Quality Scoring System

Paper: https://arxiv.org/abs/2401.01078 (January 2024)
       "Vietnamese Poem Generation & The Prospect Of Cross-Language Poem-To-Poem Translation"
Code: https://github.com/Anshler/poem_generator

Task: Generate Vietnamese poems from natural language prompts across various traditional
Vietnamese poetry genres (lục bát, 4 chữ, 5 chữ, 7 chữ, 8 chữ). Each genre has strict rules
for structure, rhyme, and tone patterns.

Dataset: 480 test instances with Vietnamese prompts
  - Genres: lục bát (alternating 6-8 syllable couplets), 4 chữ, 5 chữ, 7 chữ, 8 chữ
  - Prompts specify genre, topic, keywords
  - Example prompt: "Viết bài thơ lục bát về đam mê làm cha mẹ, ước mơ cho gia đình
    hạnh phúc và con cái thành nhân. Có chứa từ khóa 'gia đình hạnh phúc', 'con cái thành nhân'."
    (Write a lục bát poem about passion for parenting, dreams for a happy family...)

Evaluation: Automatic rule-based scoring
  - Formula: score = L/10 + 3T/10 + 6R/10
  - L = Length score (correct syllable count per line)
  - T = Tone score (correct tonal patterns)
  - R = Rhyme score (correct rhyme patterns at line endings)
  - Scores based on conformance to rigid rules of Vietnamese poetry genres

Vietnamese Poetry Genres:
  - Lục bát: Alternating 6-8 syllable lines with specific rhyme patterns
  - 4 chữ: 4 syllables per line
  - 5 chữ: 5 syllables per line
  - 7 chữ: 7 syllables per line
  - 8 chữ: 8 syllables per line

Note: This benchmark requires Vietnamese language understanding. Evaluation requires
Vietnamese NLP tools (tone detection, rhyme analysis, syllable counting).

Reference: Inspired by fsoft-ailab's SP-GPT2 evaluation method
Data source: https://github.com/fsoft-ailab/Poem-Generator (171,188 Vietnamese poems)
"""

import json
import os
import urllib.request
from typing import Optional, List
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class VietnamesePoemScenario(Scenario):
    name = "vietnamese_poem"
    description = "Vietnamese Poem Generation with Rule-Based Quality Scoring"
    tags = ["creativity", "generation", "poetry", "vietnamese", "multilingual"]

    # Test dataset from GitHub
    DATA_URL = "https://raw.githubusercontent.com/Anshler/poem_generator/master/resource/dataset/dataset_test.json"

    def __init__(
        self,
        genre_filter: Optional[str] = None,
        num_instances: Optional[int] = None
    ):
        """
        Initialize Vietnamese Poem scenario.

        Args:
            genre_filter: Filter by Vietnamese poetry genre. Options:
                         "lục bát", "4 chữ", "5 chữ", "7 chữ", "8 chữ", or None for all.
                         (default: None - includes all genres)
            num_instances: Maximum number of instances to include. If None, includes all.
                          (default: None)
        """
        super().__init__()
        self.genre_filter = genre_filter
        self.num_instances = num_instances

        # Validate genre filter
        valid_genres = ["lục bát", "4 chữ", "5 chữ", "7 chữ", "8 chữ"]
        if self.genre_filter and self.genre_filter not in valid_genres:
            raise ValueError(f"genre_filter must be one of {valid_genres} or None, got: {self.genre_filter}")

    def get_instances(self, output_path: str) -> List[Instance]:
        """Generate instances for Vietnamese poem generation."""
        # Download the data file
        data_path = os.path.join(output_path, "vietnamese_poem_test.jsonl")
        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(self.DATA_URL, data_path)

        # Load test instances (JSONL format - one JSON per line)
        instances_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    instances_data.append(json.loads(line.strip()))

        # Filter by genre if specified
        if self.genre_filter:
            instances_data = [
                item for item in instances_data
                if self.genre_filter in item.get('prompt', '')
            ]

        # Limit number of instances if specified
        if self.num_instances:
            instances_data = instances_data[:self.num_instances]

        # Create HELM instances
        instances = []
        for idx, item in enumerate(instances_data):
            prompt = item.get('prompt', '')
            reference_poem = item.get('completion', '')

            # Each instance prompts for poem generation
            # The reference poem serves as a quality reference
            # Note: Evaluation should use rule-based Vietnamese poetry scoring,
            # not simple text similarity metrics like BLEU
            instances.append(Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=reference_poem), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
                extra_data={
                    "instance_id": idx,
                    "prompt": prompt,
                    "reference_poem": reference_poem
                }
            ))

        return instances
