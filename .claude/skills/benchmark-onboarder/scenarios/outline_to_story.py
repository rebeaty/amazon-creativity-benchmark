"""
HELM Scenario: Outline to Story (O2S)

Paper: Outline to Story: Fine-grained Controllable Story Generation from Cascaded Events
       https://arxiv.org/abs/2101.00822

Dataset: WritingPrompts (euclaise/writingprompts on HuggingFace)
GitHub: https://github.com/fangleai/Outline2Story
License: Not specified

Task: Controllable Story Generation from Outlines
Generate creative multi-paragraph stories from writing prompts that serve as outlines/guidance.

The O2S paper proposes generating stories from "cascaded events" - a sequence of outline
events that guide paragraph generation. The WritingPrompts dataset provides natural
prompts/outlines paired with human-written stories, serving as a benchmark for this task.

Dataset: 272,600 train, 15,620 validation, 15,138 test examples
Source: Reddit's r/WritingPrompts community

Prompt format: Prompts are natural writing prompts from Reddit, e.g.:
  "[WP] Leonardo DiCaprio in a fit of rage begins to torpedo his own career..."
  "[CW] Kill the writer in first-person narrative."
  "[EU] Sean Bean has a hard time leaving his role as Eddard Stark..."

Fields used: prompt (outline/guidance), story (reference)
Fields skipped: None (dataset only has 2 fields)

Evaluation: Open-ended generation (BLEU, ROUGE, F1 against reference stories)
Note: Stories are typically multi-paragraph creative fiction (500-2000 words)
"""

from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
    VALID_SPLIT,
)
from datasets import load_dataset


class OutlineToStoryScenario(Scenario):
    """
    Outline to Story (O2S): Controllable Story Generation

    Evaluates models' ability to generate creative multi-paragraph stories
    from writing prompts that serve as outlines or guidance.
    """

    name = "outline_to_story"
    description = "euclaise/writingprompts"
    tags = ["creativity", "story_generation", "long_form_generation"]

    def __init__(self, split: str = "test", max_instances: int = 0):
        """
        Args:
            split: "test" (15,138 examples) or "validation" (15,620 examples)
                  Default is "test"
            max_instances: Maximum number of instances to use (0 = use all)
                          Useful for limiting dataset size during testing
        """
        super().__init__()
        self.split = split
        self.max_instances = max_instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load WritingPrompts dataset and create O2S instances.

        Each instance contains:
        - Prompt: Writing prompt serving as outline/guidance
        - Reference: Human-written story based on the prompt
        """
        # Load WritingPrompts dataset from HuggingFace
        # Map split names: "validation" in dataset -> VALID_SPLIT in HELM
        helm_split = VALID_SPLIT if self.split == "validation" else TEST_SPLIT
        dataset = load_dataset("euclaise/writingprompts", split=self.split)

        # Limit dataset size if max_instances specified
        if self.max_instances > 0:
            dataset = dataset.select(range(min(self.max_instances, len(dataset))))

        instances = []
        for idx, example in enumerate(dataset):
            # Writing prompt serves as the outline/guidance
            prompt_text = example["prompt"].strip()

            # Human-written story is the reference
            story_text = example["story"].strip()

            # Create prompt for the model
            # The model should generate a story based on the writing prompt
            input_text = f"Write a creative story based on this prompt:\n\n{prompt_text}"

            # Create reference from human story
            references = [
                Reference(
                    Output(text=story_text),
                    tags=[CORRECT_TAG]
                )
            ]

            # Create instance
            instances.append(
                Instance(
                    input=Input(text=input_text),
                    references=references,
                    split=helm_split,
                    id=f"o2s_{self.split}_{idx}",
                )
            )

        return instances
