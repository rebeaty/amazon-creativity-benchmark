"""
HELM Scenario: New Yorker Cartoon Caption Contest Humor 'Understanding' Benchmarks

Paper: https://arxiv.org/abs/2209.06293 (ACL 2023 Best Paper)
       "Do Androids Laugh at Electric Sheep? Humor 'Understanding' Benchmarks
        from The New Yorker Caption Contest"
Authors: Jack Hessel, Ana Marasovic, Jena D. Hwang, Lillian Lee, Jeff Da,
         Rowan Zellers, Robert Mankoff, and Yejin Choi
Dataset: https://huggingface.co/datasets/jmhessel/newyorker_caption_contest
Website: www.capcon.dev

Task: Evaluate humor understanding through three progressively sophisticated tasks
using New Yorker cartoon caption contest data.

Three Tasks:
1. Matching: Select the correct caption from 5 choices for a given cartoon
2. Ranking: Choose the funnier caption between 2 options for a cartoon
3. Explanation: Generate an explanation for why a caption is funny

Dataset: Uses textual descriptions of cartoons for text-only model evaluation
  - Matching: 9,792 train, 531 val, 528 test
  - Ranking: 9,576 train, 507 val, 513 test
  - Explanation: 2,340 train, 130 val, 131 test

Evaluation:
  - Matching & Ranking: Accuracy (multiple choice)
  - Explanation: Quality of generated explanations (requires human evaluation or
    comparison to reference explanations)

Key Features:
  - Text-only evaluation using multifaceted visual scene descriptions
  - 5-fold cross-validation supported (configurations _1 through _4)
  - Tests sophisticated multimodal humor understanding
  - Significant human-AI gap: Best models ~62% vs humans ~94% on matching

Note: This implementation uses textual descriptions (image_description,
image_location, image_uncanny_description) instead of cartoon images, allowing
evaluation with text-only language models.
"""

from datasets import load_dataset
from typing import List, Optional
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, VALID_SPLIT, TEST_SPLIT, TRAIN_SPLIT
)


class NewYorkerHumorScenario(Scenario):
    name = "newyorker_humor"
    description = "New Yorker Cartoon Caption Contest Humor Understanding Benchmarks"
    tags = ["creativity", "humor", "caption", "commonsense", "explanation"]

    def __init__(
        self,
        task: str = "matching",
        cross_val_fold: Optional[int] = None,
        use_uncanny_description: bool = True
    ):
        """
        Initialize New Yorker Humor Understanding scenario.

        Args:
            task: One of "matching", "ranking", or "explanation"
            cross_val_fold: Cross-validation fold (1-4) or None for default split
            use_uncanny_description: If True, include the "uncanny" description along
                                    with standard description (recommended for better
                                    humor understanding)
        """
        super().__init__()

        if task not in ["matching", "ranking", "explanation"]:
            raise ValueError(f"task must be 'matching', 'ranking', or 'explanation', got: {task}")

        if cross_val_fold is not None and cross_val_fold not in [1, 2, 3, 4]:
            raise ValueError(f"cross_val_fold must be None or 1-4, got: {cross_val_fold}")

        self.task = task
        self.cross_val_fold = cross_val_fold
        self.use_uncanny_description = use_uncanny_description

        # Construct dataset configuration name
        if cross_val_fold is None:
            self.config_name = task
        else:
            self.config_name = f"{task}_{cross_val_fold}"

    def _format_cartoon_description(self, item: dict) -> str:
        """Format the textual description of the cartoon."""
        parts = []

        # Location description
        if item.get('image_location'):
            parts.append(f"Location: {item['image_location']}")

        # Standard description
        if item.get('image_description'):
            parts.append(f"Scene: {item['image_description']}")

        # Uncanny/humor-focused description (often captures what makes it funny)
        if self.use_uncanny_description and item.get('image_uncanny_description'):
            parts.append(f"Notable: {item['image_uncanny_description']}")

        # Entities in the scene
        if item.get('entities'):
            entities = item['entities']
            if entities:
                parts.append(f"Entities: {', '.join(entities)}")

        # Questions about the scene (help guide understanding)
        if item.get('questions'):
            questions = item['questions']
            if questions:
                parts.append(f"Questions: {' | '.join(questions)}")

        return "\n".join(parts)

    def get_instances(self, output_path: str) -> List[Instance]:
        """Generate instances for New Yorker Humor benchmarks."""
        # Load dataset from HuggingFace
        dataset = load_dataset('jmhessel/newyorker_caption_contest', self.config_name)

        instances = []

        # Process each split
        for split_name, helm_split in [
            ('train', TRAIN_SPLIT),
            ('validation', VALID_SPLIT),
            ('test', TEST_SPLIT)
        ]:
            if split_name not in dataset:
                continue

            for idx, item in enumerate(dataset[split_name]):
                if self.task == "matching":
                    instance = self._create_matching_instance(item, idx, helm_split)
                elif self.task == "ranking":
                    instance = self._create_ranking_instance(item, idx, helm_split)
                else:  # explanation
                    instance = self._create_explanation_instance(item, idx, helm_split)

                instances.append(instance)

        return instances

    def _create_matching_instance(self, item: dict, idx: int, split: str) -> Instance:
        """Create instance for matching task (5-way multiple choice)."""
        cartoon_desc = self._format_cartoon_description(item)
        caption_choices = item['caption_choices']
        label = item['label']

        # Format as multiple choice (A, B, C, D, E)
        choice_letters = ['A', 'B', 'C', 'D', 'E']
        choices_text = "\n".join([
            f"{letter}. {caption}"
            for letter, caption in zip(choice_letters, caption_choices)
        ])

        prompt = (
            f"Cartoon Description:\n{cartoon_desc}\n\n"
            f"Which of the following captions best matches this cartoon?\n\n"
            f"{choices_text}\n\n"
            f"Answer (A, B, C, D, or E):"
        )

        # Create references for each choice
        references = []
        for i, letter in enumerate(choice_letters[:len(caption_choices)]):
            tags = [CORRECT_TAG] if label == letter else []
            references.append(Reference(Output(text=letter), tags=tags))

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=split,
            extra_data={
                "instance_id": item.get('instance_id', f"{split}_{idx}"),
                "contest_number": item.get('contest_number'),
                "task": "matching",
                "caption_choices": caption_choices,
                "correct_answer": label
            }
        )

    def _create_ranking_instance(self, item: dict, idx: int, split: str) -> Instance:
        """Create instance for ranking task (2-way comparison)."""
        cartoon_desc = self._format_cartoon_description(item)
        caption_choices = item['caption_choices']
        label = item['label']

        # For ranking, we have exactly 2 captions to compare
        prompt = (
            f"Cartoon Description:\n{cartoon_desc}\n\n"
            f"Which caption is funnier for this cartoon?\n\n"
            f"A. {caption_choices[0]}\n"
            f"B. {caption_choices[1]}\n\n"
            f"Answer (A or B):"
        )

        # Create references
        ref_a = Reference(Output(text="A"), tags=[CORRECT_TAG] if label == "A" else [])
        ref_b = Reference(Output(text="B"), tags=[CORRECT_TAG] if label == "B" else [])
        references = [ref_a, ref_b]

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=split,
            extra_data={
                "instance_id": item.get('instance_id', f"{split}_{idx}"),
                "contest_number": item.get('contest_number'),
                "task": "ranking",
                "caption_choices": caption_choices,
                "correct_answer": label,
                "winner_source": item.get('winner_source')
            }
        )

    def _create_explanation_instance(self, item: dict, idx: int, split: str) -> Instance:
        """Create instance for explanation task (open-ended generation)."""
        cartoon_desc = self._format_cartoon_description(item)
        caption = item['caption_choices'][0]  # For explanation, there's typically one caption
        reference_explanation = item['label']  # The gold explanation

        prompt = (
            f"Cartoon Description:\n{cartoon_desc}\n\n"
            f"Caption: \"{caption}\"\n\n"
            f"Explain why this caption is funny for this cartoon:"
        )

        # For explanation task, we provide the reference explanation
        references = [Reference(Output(text=reference_explanation), tags=[CORRECT_TAG])]

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=split,
            extra_data={
                "instance_id": item.get('instance_id', f"{split}_{idx}"),
                "contest_number": item.get('contest_number'),
                "task": "explanation",
                "caption": caption,
                "reference_explanation": reference_explanation
            }
        )
