"""
HELM Scenario: FLUTE (Filtered) - Rhetorical Language Understanding

Paper: Rhetorical Text-to-Image Generation via Two-layer Diffusion Policy Optimization (May 2025)
arXiv: https://arxiv.org/abs/2505.22792
Code: https://github.com/zyxxxxx-39/Rhet2Pix
Original FLUTE: https://arxiv.org/abs/2205.12404
Dataset: https://huggingface.co/datasets/ColumbiaNLP/FLUTE

This scenario uses a filtered version of the FLUTE dataset, selecting high-quality
metaphor and simile samples with rhetorical clarity and visual interpretability.

The Rhet2Pix paper filters FLUTE to focus on:
- Metaphors: Figurative comparisons (1,250 examples in FLUTE)
- Similes: Explicit comparisons using "like" or "as" (1,250 examples in FLUTE)

Task: Figurative Natural Language Inference (NLI) with explanations
- Given a premise and hypothesis containing rhetorical language
- Predict: Entailment or Contradiction
- Generate: Textual explanation for the inference

Semantic Dimensions Extracted (for downstream tasks):
1. Rhetorical device (metaphor/simile)
2. Literal subject
3. Metaphorical vehicle
4. Theme
5. Emotional tone
6. Subject keywords
7. Vehicle keywords

Prompt format:
  Premise: {premise}
  Hypothesis: {hypothesis}

  Does the hypothesis follow from the premise?
  Answer with "Entailment" or "Contradiction" and explain your reasoning.

Fields used: premise, hypothesis, label, explanation, type (Metaphor/Simile)
Fields skipped: id (identifier), split (always train), idiom (not applicable)

Evaluation: Exact match for classification + BLEU/ROUGE for explanation generation
"""

from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
)
from datasets import load_dataset


class FLUTEFilteredScenario(Scenario):
    """
    FLUTE (Filtered) - Rhetorical Language Understanding

    Filters the original FLUTE dataset to include only metaphor and simile examples,
    following the filtering approach in the Rhet2Pix paper.

    Evaluates LLMs on understanding figurative language through:
    1. Natural Language Inference (entailment vs. contradiction)
    2. Explanation generation for inference decisions
    """

    name = "flute_filtered"
    description = "Metaphor and simile understanding from FLUTE (ColumbiaNLP/FLUTE)"
    tags = ["creativity", "figurative_language", "metaphor", "simile", "nli", "explanation"]

    FIGURATIVE_TYPES = ["Metaphor", "Simile"]

    def __init__(self, include_explanations: bool = True):
        """
        Args:
            include_explanations: If True, evaluate both classification and explanation
        """
        super().__init__()
        self.include_explanations = include_explanations

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load FLUTE dataset and filter for metaphors and similes.

        Returns instances for figurative NLI with optional explanation generation.
        """

        # Load FLUTE dataset from HuggingFace
        dataset = load_dataset("ColumbiaNLP/FLUTE", split="train")

        print(f"Loaded {len(dataset)} total FLUTE examples")

        # Filter for metaphors and similes
        filtered_examples = [
            ex for ex in dataset
            if ex['type'] in self.FIGURATIVE_TYPES
        ]

        print(f"Filtered to {len(filtered_examples)} metaphor/simile examples")
        print(f"  Metaphors: {len([ex for ex in filtered_examples if ex['type'] == 'Metaphor'])}")
        print(f"  Similes: {len([ex for ex in filtered_examples if ex['type'] == 'Simile'])}")

        # Create instances
        instances = []
        for ex in filtered_examples:
            # Build prompt
            if self.include_explanations:
                prompt = self._build_prompt_with_explanation(ex)
                # Reference includes both label and explanation
                reference_text = f"{ex['label']}\n\nExplanation: {ex['explanation']}"
            else:
                prompt = self._build_prompt_classification_only(ex)
                # Reference is just the label
                reference_text = ex['label']

            # Create instance
            instance = Instance(
                input=Input(text=prompt),
                references=[Reference(output=reference_text, tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )

            # Add metadata
            instance.id = f"{ex['type'].lower()}_{ex['id']}"
            instance.extra_data = {
                'type': ex['type'],
                'premise': ex['premise'],
                'hypothesis': ex['hypothesis'],
                'label': ex['label'],
                'explanation': ex['explanation'],
                'original_id': ex['id'],
            }

            instances.append(instance)

        return instances

    def _build_prompt_with_explanation(self, ex: dict) -> str:
        """Build prompt asking for both classification and explanation."""
        return f"""Premise: {ex['premise']}
Hypothesis: {ex['hypothesis']}

Does the hypothesis follow from the premise? Answer with "Entailment" or "Contradiction" and explain your reasoning."""

    def _build_prompt_classification_only(self, ex: dict) -> str:
        """Build prompt for classification only."""
        return f"""Premise: {ex['premise']}
Hypothesis: {ex['hypothesis']}

Does the hypothesis follow from the premise? Answer with "Entailment" or "Contradiction"."""
