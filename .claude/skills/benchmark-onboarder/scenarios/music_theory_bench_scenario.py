"""
HELM Scenario: MusicTheoryBench

Paper: ChatMusician: Understanding and Generating Music Intrinsically with LLM
       https://arxiv.org/abs/2402.16153
Code: https://github.com/hf-lin/ChatMusician
Dataset: https://huggingface.co/datasets/m-a-p/MusicTheoryBench

Task: College-level music theory understanding across knowledge and reasoning domains.
      Questions test theoretical knowledge (chords, scales, rhythm) and analytical
      reasoning (modulation analysis, harmonic progressions). Some questions include
      ABC notation for musical examples.

Dataset: 372 questions crafted by a professional college music teacher
- Music Knowledge: 269 questions covering 30 topics (notes, rhythm, chords, etc.)
- Music Reasoning: 98 questions requiring multi-step logical analysis
- Test split: 367 examples (269 knowledge, 98 reasoning)
- Dev split: 5 examples for few-shot prompting

Prompt format: From eval/configs/datasets/music_theory_bench/music_theory_bench_ppl_zero_shot.py
  Read the following questions from the four options (A, B, C and D) given in each
  question. Choose the best option.

  {stem}
  A. {option_A}
  B. {option_B}
  C. {option_C}
  D. {option_D}
  Answer:

Fields used: stem, options (A/B/C/D), answer, subject, abc_score
Fields skipped: id, instruction (redundant with our prompt), split, analysis (explanation)
Note: abc_score contains ABC notation when musical examples are needed; empty string otherwise

Evaluation: exact_match (accuracy on multiple choice)
"""

from datasets import load_dataset
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
    VALID_SPLIT,
)


class MusicTheoryBenchScenario(Scenario):
    """MusicTheoryBench: College-level Music Theory Understanding

    Evaluates models' understanding of music theory concepts through multiple-choice
    questions on music knowledge and reasoning. Includes ABC notation for musical examples.
    """

    name = "music_theory_bench"
    description = "m-a-p/MusicTheoryBench"
    tags = ["creativity", "music", "music_theory", "multiple_choice"]

    def __init__(self, subject: str = "all"):
        """
        Args:
            subject: Filter by subject ('knowledge', 'reasoning', or 'all')
        """
        super().__init__()
        if subject not in ["knowledge", "reasoning", "all"]:
            raise ValueError(f"Invalid subject: {subject}. Must be 'knowledge', 'reasoning', or 'all'")
        self.subject = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        # Load both splits
        dataset_test = load_dataset("m-a-p/MusicTheoryBench", split="test")
        dataset_dev = load_dataset("m-a-p/MusicTheoryBench", split="dev")

        instances = []

        # Process dev split (for few-shot examples if needed)
        for item in dataset_dev:
            if self.subject == "all" or item["subject"] == self.subject:
                instances.append(self._create_instance(item, VALID_SPLIT))

        # Process test split
        for item in dataset_test:
            if self.subject == "all" or item["subject"] == self.subject:
                instances.append(self._create_instance(item, TEST_SPLIT))

        return instances

    def _create_instance(self, item: dict, split: str) -> Instance:
        """Create an instance from a MusicTheoryBench question"""

        # Build prompt following the config file format
        prompt = "Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n\n"

        # Add the question stem
        prompt += item["stem"]

        # If there's ABC notation, include it inline (it's already part of the stem in most cases)
        # The abc_score field is sometimes empty, sometimes contains notation

        prompt += "\n"

        # Add options
        options = item["options"]
        prompt += f"A. {options['A']}\n"
        prompt += f"B. {options['B']}\n"
        prompt += f"C. {options['C']}\n"
        prompt += f"D. {options['D']}\n"
        prompt += "Answer:"

        # HELM MC pattern: all choices become References, only correct one tagged
        correct_answer = item["answer"]
        references = []
        for letter in ["A", "B", "C", "D"]:
            is_correct = (letter == correct_answer)
            tags = [CORRECT_TAG] if is_correct else []
            references.append(Reference(Output(text=letter), tags=tags))

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=split,
        )
