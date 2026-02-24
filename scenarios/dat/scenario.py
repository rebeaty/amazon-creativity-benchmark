"""
HELM Scenario: Divergent Association Task (DAT)

Paper: Chen, H., & Ding, N. (2023). Probing the "Creativity" of Large Language
       Models: Can models produce divergent semantic association?
       Findings of EMNLP 2023, pages 12881-12888.
       https://aclanthology.org/2023.findings-emnlp.858/

Code: https://github.com/DingNLab/probing_creativity
Original DAT: https://github.com/jayolson/divergent-association-task

Dataset: Human baseline data from probing_creativity repository (8,572 responses)

Task: Models generate 10 unrelated nouns. Creativity is measured by semantic
      distance between words using GloVe embeddings.

Prompt format:
  Please write 10 nouns in English that are as irrelevant from each other as
  possible, in all meanings and uses of the words. Please note that the words
  you write should have only single word, only nouns (e.g., things, objects,
  concepts), and no proper nouns (e.g., no specific people or places). Your answer:

Evaluation: Requires custom DATMetric implementation (see metric_notes.md)
            - Computes average cosine distance between all word pairs
            - Uses GloVe 840B 300d embeddings
            - Score range: 0-100 (higher = more creative)

Fields used: Prompt from dataset.py (lines 108-110)
Reference data: Human responses from dataset/human.json (mean DAT: ~78, SD: ~10)
"""

from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    TEST_SPLIT,
)


class DATScenario(Scenario):
    """
    Divergent Association Task (DAT) - A brief measure of verbal creativity.

    The task asks models to generate 10 semantically unrelated nouns. Creativity
    is measured by computing the average cosine distance between all word pairs
    using GloVe embeddings.

    Original task by Olson et al. (2021) in PNAS: "Naming unrelated words predicts creativity"
    LLM evaluation methodology from Chen & Ding (2023) EMNLP.
    """

    name = "dat"
    description = "DingNLab/probing_creativity"  # GitHub repo source
    tags = ["creativity", "divergent_thinking", "generation"]

    def __init__(self, num_trials: int = 100):
        """
        Args:
            num_trials: Number of independent generation trials (default: 100)
                       Each trial is an independent request to generate 10 words.
        """
        super().__init__()
        self.num_trials = num_trials

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate instances for the Divergent Association Task.

        Each instance represents an independent trial where the model generates
        10 unrelated nouns. The same prompt is used for all trials.

        Note: Evaluation requires custom DATMetric implementation that computes
        semantic distances using GloVe embeddings. See metric_notes.md for details.
        """
        # Standard DAT prompt from Chen & Ding (2023)
        prompt = (
            "Please write 10 nouns in English that are as irrelevant from each other as possible, "
            "in all meanings and uses of the words. Please note that the words you write should have only single word, "
            "only nouns (e.g., things, objects, concepts), and no proper nouns (e.g., no specific people or places). "
            "Your answer:"
        )

        instances = []
        for trial_idx in range(self.num_trials):
            # Create instance with empty references (open-ended generation task)
            # The DATMetric will compute creativity scores from generated outputs
            instance = Instance(
                input=Input(text=prompt),
                references=[],  # No ground truth - evaluated by semantic distance metric
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances
