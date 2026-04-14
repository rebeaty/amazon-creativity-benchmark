"""
HELM Scenario: GraphRAG-Bench

Paper: GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation
       arXiv:2506.02404 (https://arxiv.org/abs/2506.02404)
Code: https://github.com/jeremycp3/GraphRAG-Bench
Dataset: jeremycp3/GraphRAG-Bench (HuggingFace)

GraphRAG-Bench evaluates domain-specific reasoning across 5 question types from 20 computer science textbooks.
The benchmark includes Fill-in-Blank (FB), Multiple-Choice (MC), Multi-Select (MS), True/False (TF), and Open-Ended (OE) questions.

Prompt format:
  For open-ended: Question text (no special formatting)
  For MC/MS: Question text followed by choices (formatted in scenario)
  For FB: Question with blank (________)
  For TF: Statement to evaluate

Fields used: Question, Answer, Rationale, Level-1 Topic, Level-2 Topic, Choices (for MC/MS only)
Fields skipped: None

Note: Dataset has inconsistent columns across question types (MC/MS have 'Choices' field, others don't).
      Must load each question type separately and combine.
"""

import json
import os
from typing import Dict, List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)
from helm.common.general import ensure_file_downloaded


class GraphRAGBenchScenario(Scenario):
    """
    GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation

    Dataset evaluates domain-specific reasoning across computer science topics with 5 question types:
    - Fill-in-Blank (FB): 105 questions
    - Multiple-Choice (MC): 217 questions
    - Multi-Select (MS): 111 questions
    - True/False (TF): 316 questions
    - Open-Ended (OE): 269 questions

    Total: 1,018 questions from 20 computer science textbooks covering 16 disciplines.
    """

    name = "graphragbench"
    description = "jeremycp3/GraphRAG-Bench"
    tags = ["creativity", "domain_reasoning", "computer_science"]

    QUESTION_TYPES = ["FB", "MC", "MS", "TF", "OE"]

    # URLs for the question files from HuggingFace
    BASE_URL = "https://huggingface.co/datasets/jeremycp3/GraphRAG-Bench/resolve/main/questions"

    def __init__(self, question_type: str = "all"):
        """
        Initialize GraphRAG-Bench scenario.

        Args:
            question_type: Type of questions to include. Options:
                - "all": All question types (default)
                - "FB": Fill-in-Blank only
                - "MC": Multiple-Choice only
                - "MS": Multi-Select only
                - "TF": True/False only
                - "OE": Open-Ended only
        """
        super().__init__()
        if question_type not in ["all"] + self.QUESTION_TYPES:
            raise ValueError(
                f"Invalid question_type: {question_type}. "
                f"Must be 'all' or one of {self.QUESTION_TYPES}"
            )
        self.question_type = question_type

    def _download_questions(self, output_path: str, qtype: str) -> str:
        """Download question file for a specific type."""
        url = f"{self.BASE_URL}/{qtype}.jsonl"
        filename = f"{qtype}.jsonl"
        target_path = os.path.join(output_path, filename)
        ensure_file_downloaded(source_url=url, target_path=target_path)
        return target_path

    def _load_questions(self, filepath: str) -> List[Dict]:
        """Load questions from a JSONL file."""
        questions = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        return questions

    def _create_fb_instance(self, item: Dict, idx: int) -> Instance:
        """Create instance for Fill-in-Blank question."""
        prompt = item["Question"]
        answer = item["Answer"]

        return Instance(
            input=Input(text=prompt),
            references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
            split=TEST_SPLIT,
            extra_data={
                "id": f"FB_{idx}",
                "level_1_topic": item["Level-1 Topic"],
                "level_2_topic": item["Level-2 Topic"],
                "rationale": item["Rationale"],
                "question_type": "fill_in_blank"
            }
        )

    def _create_mc_instance(self, item: Dict, idx: int) -> Instance:
        """Create instance for Multiple-Choice question."""
        question = item["Question"]
        choices = item["Choices"]
        answer = item["Answer"]

        # Format prompt with choices
        prompt = f"{question}\n\n"
        for choice_key in sorted(choices.keys()):
            prompt += f"{choice_key}) {choices[choice_key]}\n"
        prompt += "\nAnswer:"

        # Create references for all choices, tag correct one
        references = []
        for choice_key in sorted(choices.keys()):
            tags = [CORRECT_TAG] if choice_key == answer else []
            references.append(Reference(Output(text=choice_key), tags=tags))

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=TEST_SPLIT,
            extra_data={
                "id": f"MC_{idx}",
                "level_1_topic": item["Level-1 Topic"],
                "level_2_topic": item["Level-2 Topic"],
                "rationale": item["Rationale"],
                "question_type": "multiple_choice"
            }
        )

    def _create_ms_instance(self, item: Dict, idx: int) -> Instance:
        """Create instance for Multi-Select question."""
        question = item["Question"]
        choices = item["Choices"]
        answer = item["Answer"]

        # Parse answer (e.g., "ABD" -> ["A", "B", "D"])
        correct_choices = list(answer)

        # Format prompt with choices
        prompt = f"{question}\n\n"
        for choice_key in sorted(choices.keys()):
            prompt += f"{choice_key}) {choices[choice_key]}\n"
        prompt += "\nAnswer (select all that apply):"

        # Create reference with comma-separated correct choices
        # Sort to ensure consistent ordering
        correct_answer = ", ".join(sorted(correct_choices))

        return Instance(
            input=Input(text=prompt),
            references=[Reference(Output(text=correct_answer), tags=[CORRECT_TAG])],
            split=TEST_SPLIT,
            extra_data={
                "id": f"MS_{idx}",
                "level_1_topic": item["Level-1 Topic"],
                "level_2_topic": item["Level-2 Topic"],
                "rationale": item["Rationale"],
                "question_type": "multi_select",
                "correct_choices": correct_choices
            }
        )

    def _create_tf_instance(self, item: Dict, idx: int) -> Instance:
        """Create instance for True/False question."""
        question = item["Question"]
        answer = item["Answer"]

        prompt = f"{question}\n\nAnswer (True or False):"

        # Create references for both options, tag correct one
        references = []
        for option in ["True", "False"]:
            tags = [CORRECT_TAG] if option == answer else []
            references.append(Reference(Output(text=option), tags=tags))

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=TEST_SPLIT,
            extra_data={
                "id": f"TF_{idx}",
                "level_1_topic": item["Level-1 Topic"],
                "level_2_topic": item["Level-2 Topic"],
                "rationale": item["Rationale"],
                "question_type": "true_false"
            }
        )

    def _create_oe_instance(self, item: Dict, idx: int) -> Instance:
        """Create instance for Open-Ended question."""
        question = item["Question"]
        answer = item["Answer"]

        return Instance(
            input=Input(text=question),
            references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
            split=TEST_SPLIT,
            extra_data={
                "id": f"OE_{idx}",
                "level_1_topic": item["Level-1 Topic"],
                "level_2_topic": item["Level-2 Topic"],
                "rationale": item["Rationale"],
                "question_type": "open_ended"
            }
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        """Generate instances for GraphRAG-Bench."""
        instances = []

        # Determine which question types to load
        if self.question_type == "all":
            types_to_load = self.QUESTION_TYPES
        else:
            types_to_load = [self.question_type]

        # Load and process each question type
        for qtype in types_to_load:
            filepath = self._download_questions(output_path, qtype)
            questions = self._load_questions(filepath)

            # Create instances based on question type
            for idx, item in enumerate(questions):
                if qtype == "FB":
                    instance = self._create_fb_instance(item, idx)
                elif qtype == "MC":
                    instance = self._create_mc_instance(item, idx)
                elif qtype == "MS":
                    instance = self._create_ms_instance(item, idx)
                elif qtype == "TF":
                    instance = self._create_tf_instance(item, idx)
                elif qtype == "OE":
                    instance = self._create_oe_instance(item, idx)

                instances.append(instance)

        return instances
