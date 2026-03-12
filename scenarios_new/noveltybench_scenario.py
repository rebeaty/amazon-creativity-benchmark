"""
HELM Scenario: NoveltyBench

Paper: "NoveltyBench: Evaluating Language Models for Humanlike Diversity"
       (Zhang et al., 2025)
       https://arxiv.org/abs/2504.05228
Code: https://github.com/novelty-bench/novelty-bench
Website: https://novelty-bench.github.io/

NoveltyBench evaluates LLMs' ability to produce multiple distinct and high-quality
outputs for the same prompt. The benchmark measures diversity (how different outputs
are) and quality (how good they are).

Dataset structure:
- NB-Curated: 100 manually crafted prompts (randomness, factual knowledge,
  creative writing, subjectivity)
- NB-WildChat: 1,000 prompts from real ChatGPT user interactions

Prompt format:
  {prompt}
  (plain prompt, models generate multiple times)

Example:
  "Tell me a story in five sentences about a girl and her dog."

Fields used: prompt, id
Evaluation: Diversity measured with classifier (deberta-v3-large trained on 1,000
            human annotations). Quality assessed with reward model (Skywork-Reward).
            Models generate multiple outputs per prompt (typically 5-10) and diversity
            is measured across these outputs.

Dataset source: yimingzhang/novelty-bench on HuggingFace
Used in: DARLING paper (arXiv:2509.02534) for diversity evaluation
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


class NoveltyBenchScenario(Scenario):
    """
    NoveltyBench

    Evaluates diversity and quality of multiple generations from the same prompt.
    Current state-of-the-art models generate significantly less diversity than
    human writers. Larger models often exhibit less diversity than smaller ones,
    challenging assumptions about capability scaling.

    Key finding: Capability on standard benchmarks doesn't translate to generative
    diversity - a critical aspect of creative and practical utility.
    """

    name = "noveltybench"
    description = "yimingzhang/novelty-bench"
    tags = ["creativity", "diversity", "quality", "multi_generation"]

    def __init__(self, subset: str = "all"):
        """
        Args:
            subset: Which subset to load - "curated", "wildchat", or "all" (default)
        """
        super().__init__()
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load NoveltyBench prompts.

        For each prompt, models should generate multiple outputs (typically 5-10).
        Diversity is measured across these outputs using a trained classifier,
        and quality is assessed with a reward model.
        """

        # Load dataset from HuggingFace
        dataset = load_dataset("yimingzhang/novelty-bench")

        instances = []

        # Determine which splits to include
        if self.subset == "all":
            splits = ["curated", "wildchat"]
        elif self.subset == "curated":
            splits = ["curated"]
        elif self.subset == "wildchat":
            splits = ["wildchat"]
        else:
            raise ValueError(f"Invalid subset: {self.subset}. Must be 'curated', 'wildchat', or 'all'")

        for split_name in splits:
            split_data = dataset[split_name]

            for item in split_data:
                prompt_id = item['id']
                prompt_text = item['prompt']

                # References are empty - this is a pure generation task
                # Evaluation uses diversity classifier and reward model, not references
                references = []

                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"novelty_{prompt_id}",
                    )
                )

        return instances
