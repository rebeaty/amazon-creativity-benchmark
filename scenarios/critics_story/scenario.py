"""
HELM Scenario: Creative Story Plan Generation (CritiCS / DOC-StoryGen-v2)

Paper: "Collective Critics for Creative Story Generation"
       EMNLP 2024, arXiv:2410.02428
Code: https://github.com/EMNLP-2024-CritiCS/Collective-Critics-for-Creative-Story-Generation
Dataset: https://huggingface.co/datasets/llm-aes/doc-storygen-v2

Creative story plan generation from narrative premises. Given a 30-100 word
premise describing a scenario, setting, and characters, generate a detailed
story plan (settings, characters, plot outline with multiple paragraphs).

The CritiCS paper used 300 premises from the DOC pipeline (Yang et al., 2023).
This scenario uses the doc-storygen-v2 dataset which provides 6,932 unique
premises from the same methodology (a superset), each with two machine-generated
story plans and human pairwise annotations on 6 creativity dimensions.

Subsets:
  - generation (default): Open-ended story plan generation from premises.
    Each premise is a generation prompt; no reference output (LLM-as-judge eval).
    6,932 unique premises.
  - judgment: Story plan quality judgment. Given a premise and two plans,
    select which is more interesting/coherent/creative. Uses human annotations
    from doc-storygen-v2 as ground truth. 7,000 annotated pairs.

Prompt source:
  - generation: Standard instruction + premise (no exact prompt in paper;
    CritiCS uses multi-agent pipeline, not single prompt). Instruction adapted
    from DOC plan format.
  - judgment: Heavily adapted from CritiCS evaluation/evaluation_prompt.py
    (persona_comparision template). Simplified from 4 questions to 1
    (interestingness only), added premise display, removed premise-alignment
    verdict block, and fixed typos from original.

Fields used: premise, plan1, plan2, Q1 (interesting), Q3 (engaging), Q4 (suspense),
  Q5 (character), Q6 (ending)
Fields skipped: worker_id, task_id, task_response_id, Q2 (free-text explanation)

Human annotation dimensions (Q1-Q6, pairwise: Plot A / Plot B / Both / Neither):
  Q1: Which plot is more interesting?
  Q3: Which could make a more interesting book or movie?
  Q4: Which has better suspense or surprise?
  Q5: Which characters do you identify with or care about more?
  Q6: Which has a better ending?
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)

# Generation instruction adapted from DOC story plan format
_GENERATION_INSTRUCTION = (
    "Given the following story premise, write a detailed story plan. "
    "Include the setting, main characters with brief descriptions, "
    "and a paragraph-by-paragraph plot outline.\n\n"
    "Premise: {premise}\n\n"
    "Story Plan:"
)

# Pairwise judgment prompt adapted from CritiCS evaluation/evaluation_prompt.py
# (persona_comparision template), heavily simplified: added premise display,
# reduced from 4 questions to 1 (interestingness only), removed premise-alignment
# verdict, and fixed typos (engag→engage, Creative::→Creative:, B/ C→B / C).
_JUDGMENT_INSTRUCTION = (
    "Here are two storyline excerpts.\n"
    "You shouldn't be concerned about the completeness of the plot.\n\n"
    "Premise:\n{premise}\n\n"
    "Storyline A:\n{plan_a}\n\n"
    "Storyline B:\n{plan_b}\n\n"
    "Answer the following question:\n"
    "Overall, which story do you prefer/find more interesting? A / B / C\n\n"
    "Metrics explanation:\n"
    "(1) Interesting: How does the story plan engage and captivate readers, "
    "making them find the narrative interesting?\n"
    "(2) Coherent: How well are the paragraphs organized and connected "
    "with each other?\n"
    "(3) Creative: How does the originality and inventiveness of the "
    "storyline offer a fresh perspective compared to typical narratives?\n\n"
    'Output your final verdict by strictly following this format: '
    '"[[A]]" if storyline A is better, "[[B]]" if storyline B is better, '
    'and "[[C]]" for a tie.\n'
    "Answer:"
)

# Map human annotations to standardized labels
_LABEL_MAP = {
    "Plot A": "A",
    "Plot B": "B",
}


class CriticsStoryScenario(Scenario):
    name = "critics_story"
    description = "llm-aes/doc-storygen-v2"
    tags = ["creativity", "story_generation", "planning", "open_ended"]

    SUBSETS = ["generation", "judgment"]

    def __init__(self, subset: str = "generation"):
        """
        Args:
            subset: "generation" for open-ended story plan generation,
                    "judgment" for pairwise quality comparison.
        """
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(
                f"subset must be one of {self.SUBSETS}, got '{subset}'"
            )
        self.subset = subset

    def _build_generation_instances(self, dataset):
        """Build open-ended generation instances from unique premises."""
        seen_premises = set()
        instances = []
        for item in dataset:
            premise = (item["premise"] or "").strip()
            if not premise or premise in seen_premises:
                continue
            seen_premises.add(premise)

            prompt = _GENERATION_INSTRUCTION.format(premise=premise)
            instances.append(Instance(
                input=Input(text=prompt),
                references=[],
                split=TEST_SPLIT,
                id=f"critics_gen_{item['id']}",
            ))
        return instances

    def _build_judgment_instances(self, dataset):
        """Build pairwise judgment instances with human ground truth."""
        instances = []
        for idx, item in enumerate(dataset):
            q1 = (item["Q1"] or "").strip()
            label = _LABEL_MAP.get(q1)
            if label is None:
                # Skip "Both" and "Neither" — no clear ground truth
                continue

            prompt = _JUDGMENT_INSTRUCTION.format(
                premise=item["premise"],
                plan_a=item["plan1"],
                plan_b=item["plan2"],
            )

            references = [
                Reference(Output(text="A"), tags=[CORRECT_TAG] if label == "A" else []),
                Reference(Output(text="B"), tags=[CORRECT_TAG] if label == "B" else []),
                Reference(Output(text="C"), tags=[]),
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=f"critics_judge_{idx}",
            ))
        return instances

    def get_instances(self, output_path):
        dataset = load_dataset("llm-aes/doc-storygen-v2", split="train")

        if self.subset == "generation":
            return self._build_generation_instances(dataset)
        else:
            return self._build_judgment_instances(dataset)
