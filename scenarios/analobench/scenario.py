"""
HELM Scenario: AnaloBench (Analogical Reasoning Benchmark)

Paper: https://arxiv.org/abs/2402.12370
Code: https://github.com/JHU-CLSP/AnaloBench
Data: https://huggingface.co/datasets/jhu-clsp/AnaloBench

Task: Given a target story/sentence, identify the most analogous story from
four options (Task T1). Tests abstract analogical reasoning ability.

Dataset:
  - T1S1-Subset: 340 single-sentence examples (proverbs)
  - T1S10-Subset: 340 examples with 10-sentence stories
  - T1S30-Subset: 340 examples with 30-sentence stories
  - Full versions: 24.4k examples each

Prompt format (from code/t1.py):
  Which of the following is the most analogous story to the target story?

  Note: Only generate the index without any additional text.

  Target Story:
  {sentence_or_story}

  Options:
  {options}

  Answer:

Subsets:
  "s1_subset" - Single sentence, 340 examples
  "s10_subset" - 10 sentences, 340 examples
  "s30_subset" - 30 sentences, 340 examples
  "s1_full" - Single sentence, 24.4k examples
  "s10_full" - 10 sentences, 24.4k examples
  "s30_full" - 30 sentences, 24.4k examples

Evaluation: exact_match (4-way accuracy: A/B/C/D)
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class AnaloBenchScenario(Scenario):
    name = "analobench"
    description = "jhu-clsp/AnaloBench"
    tags = ["creativity", "analogical_reasoning", "abstract_reasoning"]

    SUBSETS = ["s1_subset", "s10_subset", "s30_subset", "s1_full", "s10_full", "s30_full"]

    SUBSET_TO_CONFIG = {
        "s1_subset": "T1S1-Subset",
        "s10_subset": "T1S10-Subset",
        "s30_subset": "T1S30-Subset",
        "s1_full": "T1S1-Full",
        "s10_full": "T1S10-Full",
        "s30_full": "T1S30-Full",
    }

    # Prompt from code/t1.py (lines 43-55)
    PROMPT_TEMPLATE = """Which of the following is the most analogous story to the target story?

Note: Only generate the index without any additional text.

Target Story:
{target}

Options:
{options}

Answer:"""

    def __init__(self, subset: str = "s1_subset"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"Unknown subset: {subset}. Must be one of {self.SUBSETS}")
        self.subset = subset

    def get_instances(self, output_path: str):
        config = self.SUBSET_TO_CONFIG[self.subset]
        dataset = load_dataset("jhu-clsp/AnaloBench", config, split="train")

        instances = []
        for item in dataset:
            # For S1, use Sentence; for S10/S30, use Story field
            if "s1" in self.subset:
                target = item['Sentence']
            else:
                target = item.get('Story', item['Sentence'])

            options = item['Options']
            label = item['Label']  # A, B, C, or D

            # Format prompt
            prompt = self.PROMPT_TEMPLATE.format(
                target=target,
                options=options
            )

            # Create references for all 4 options
            references = [
                Reference(Output(text="A"), tags=[CORRECT_TAG] if label == "A" else []),
                Reference(Output(text="B"), tags=[CORRECT_TAG] if label == "B" else []),
                Reference(Output(text="C"), tags=[CORRECT_TAG] if label == "C" else []),
                Reference(Output(text="D"), tags=[CORRECT_TAG] if label == "D" else []),
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
