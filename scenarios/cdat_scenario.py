"""
HELM Scenario: CDAT (Conditional Divergent Association Task)

Paper: https://arxiv.org/abs/2601.20546
Code: https://github.com/knakajima1225/beyond_divergent_creativity
Dataset: https://github.com/knakajima1225/beyond_divergent_creativity/tree/main/cdat/data
Published: EACL 2026 (Findings)

Task: Generate 10 mutually dissimilar nouns that are each semantically associated
with a given cue word. Tests divergent creative thinking by evaluating both
novelty (how different the 10 words are from each other) and appropriateness
(how related each word is to the cue).

This addresses limitations of the original Divergent Association Task (DAT)
which only measures novelty, allowing stochastic outputs to achieve high scores.
CDAT requires both creative divergence AND contextual relevance.

Prompt format:
  Generate 10 mutually dissimilar nouns that are each semantically associated
  with the following cue word: {cue_word}

  Please provide your response as a comma-separated list of exactly 10 nouns.

Expected output:
  noun1, noun2, noun3, noun4, noun5, noun6, noun7, noun8, noun9, noun10

Evaluation: Custom metrics (see metric_notes.md):
  - Appropriateness: Average semantic similarity between each generated word
    and the cue word (using word embeddings)
  - Novelty: Average pairwise semantic distance between the 10 generated words
  - Creativity Score: Combines appropriateness and novelty

No ground truth references - evaluation is based on computational metrics using
FastText/GloVe embeddings to measure semantic relationships.

Dataset: 1,000 cue words from Brown corpus
  - Filtered for frequency and semantic properties
  - Examples: "release", "newspaper", "gratitude", "automobile"
  - Each cue word creates one test instance
"""

import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
)


class CDATScenario(Scenario):
    """
    CDAT: Conditional Divergent Association Task

    Evaluates divergent creative thinking through conditional word generation.
    Models generate 10 dissimilar nouns related to a cue word, testing both
    novelty and appropriateness dimensions of creativity.
    """

    name = "cdat"
    description = "knakajima1225/beyond_divergent_creativity"
    tags = ["creativity", "divergent-thinking", "word-association", "open-ended"]

    def __init__(self, num_cues: int = None):
        """
        Args:
            num_cues: Optional limit on number of cue words to use.
                If None, uses all 1,000 cue words from the dataset.
        """
        super().__init__()
        self.num_cues = num_cues

    def _download_cues(self) -> List[str]:
        """
        Download cue words from GitHub repository.

        Returns:
            List of cue words
        """
        cues_url = "https://raw.githubusercontent.com/knakajima1225/beyond_divergent_creativity/master/cdat/data/cdat_cues_filtered550.txt"

        print(f"Downloading CDAT cue words from GitHub...")
        response = urllib.request.urlopen(cues_url)
        content = response.read().decode('utf-8')

        # Parse cue words (one per line)
        cues = [line.strip() for line in content.split('\n') if line.strip()]

        print(f"Downloaded {len(cues)} cue words")
        return cues

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate CDAT instances from cue words.

        Each instance contains:
        - Input: Prompt with cue word asking for 10 dissimilar related nouns
        - References: Empty (evaluation uses computational metrics, no ground truth)
        """
        # Download cue words
        cues = self._download_cues()

        # Limit if specified
        if self.num_cues is not None:
            cues = cues[:self.num_cues]

        instances = []
        for idx, cue_word in enumerate(cues):
            # Format prompt
            prompt = (
                f"Generate 10 mutually dissimilar nouns that are each semantically "
                f"associated with the following cue word: {cue_word}\\n\\n"
                f"Please provide your response as a comma-separated list of exactly 10 nouns."
            )

            # No references - evaluation is based on semantic distance metrics
            # The computational metrics will measure:
            # 1. Appropriateness: similarity of each word to cue_word
            # 2. Novelty: pairwise distance between the 10 words
            references = []

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"cdat_{cue_word}_{idx}"
                )
            )

        print(f"Created {len(instances)} CDAT instances")
        return instances
