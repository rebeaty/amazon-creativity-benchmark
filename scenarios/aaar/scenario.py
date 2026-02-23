"""
HELM Scenario: AAAR-1.0 (Assessing AI's Potential to Assist Research)

Paper: AAAR-1.0: Assessing AI's Potential to Assist Research
       https://arxiv.org/abs/2410.22394
Code:  https://github.com/RenzeLou/AAAR-1.0
Data:  https://huggingface.co/datasets/Reza8848/AAAR-1.0

Subtasks implemented:
  - experiment_design: Given a partial research paper, generate a list of
    suggested experiments and explanations (100 instances).
  - paper_weakness: Given a research paper, generate reviewer-style weaknesses
    (993 instances from ICLR 2023).

Subtasks skipped:
  - equation_inference: 4-way MC equation selection — evaluates technical
    correctness of knowledge retrieval, not generative capability.
  - review_critique: Binary classification hosted on a separate GitHub repo
    (jiangshdd/ReviewCritique); evaluates judgment, not open-ended generation.

Prompt format (experiment_design) — from scripts/prompt_templates.py Exp_eval:
  You are an expert in Machine Learning and Natural Language Processing (NLP).
  Your responsibility is to help the user design experiments and develop new ideas.

  Below is the content of a research paper:
  {paper_content}

  Please carefully understand the motivation and methodology of this paper, then
  generate a list of experiment ideas that the authors should conduct to validate
  their approach. Format your response as a numbered list.

Prompt format (paper_weakness) — from scripts/prompt_templates.py Weakness_eval:
  You are an expert in Machine Learning and Natural Language Processing (NLP).
  Your responsibility is to help the user review a paper.

  Below is the content of a research paper:
  {paper_content}

  Please identify the weaknesses of this paper. Format your response as a
  numbered list. If the given context is irrelevant to research, just generate
  'No research content'.

Fields used (experiment_design):
  input (list of LaTeX lines → joined + truncated to 2800 words),
  output["What experiments do you suggest doing?"] (gold reference)
Fields used (paper_weakness):
  input.abstractText, input.title, input.sections (formatted paper text),
  output (list of per-reviewer weakness lists, flattened as gold reference)
Fields skipped:
  paper_info metadata (authors, comments), review_scores, acceptance label

Evaluation:
  experiment_design — LLM-as-judge entailment (GPT-4) + SentenceBERT F1;
                      see annotator_notes.md
  paper_weakness    — SentenceBERT soft-F1 (S-F1, En-F1); see metric_notes.md
"""

import json
import os
from typing import List

from huggingface_hub import snapshot_download

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

# Paper content is truncated to this word count to stay within model context,
# matching the default used in the original evaluation scripts.
_MAX_WORDS = 2800


class AaarScenario(Scenario):
    """
    AAAR-1.0: two open-ended research-generation subtasks.

    subtask="experiment_design" — suggest experiments for a given paper (100 instances)
    subtask="paper_weakness"    — identify weaknesses of a given paper (993 instances)
    """

    name = "aaar"
    description = "Reza8848/AAAR-1.0"
    tags = ["creativity", "research_generation", "open_ended"]

    SUBTASKS = ["experiment_design", "paper_weakness"]

    def __init__(self, subtask: str):
        super().__init__()
        assert subtask in self.SUBTASKS, (
            f"subtask must be one of {self.SUBTASKS}, got '{subtask}'"
        )
        self.subtask = subtask

    def get_instances(self, output_path: str) -> List[Instance]:
        repo_dir = snapshot_download(
            repo_id="Reza8848/AAAR-1.0",
            repo_type="dataset",
            cache_dir=output_path,
        )

        if self.subtask == "experiment_design":
            return self._load_experiment_design(repo_dir)
        else:
            return self._load_paper_weakness(repo_dir)

    # ------------------------------------------------------------------
    # ExperimentDesign
    # ------------------------------------------------------------------

    def _load_experiment_design(self, repo_dir: str) -> List[Instance]:
        task_dir = os.path.join(repo_dir, "Experiment_Design")
        instances = []

        for paper_id in sorted(os.listdir(task_dir)):
            data_file = os.path.join(task_dir, paper_id, "data_text.json")
            if not os.path.exists(data_file):
                continue

            with open(data_file, encoding="utf-8") as f:
                item = json.load(f)

            # Paper content is stored as a list of LaTeX source lines.
            paper_text = "\n".join(item.get("input", []))
            words = paper_text.split()
            if len(words) > _MAX_WORDS:
                paper_text = " ".join(words[:_MAX_WORDS])

            prompt = (
                "You are an expert in Machine Learning and Natural Language "
                "Processing (NLP). Your responsibility is to help the user "
                "design experiments and develop new ideas.\n\n"
                "Below is the content of a research paper:\n"
                f"{paper_text}\n\n"
                "Please carefully understand the motivation and methodology of "
                "this paper, then generate a list of experiment ideas that the "
                "authors should conduct to validate their approach. Format your "
                "response as a numbered list."
            )

            output_dict = item.get("output", {})
            experiments = output_dict.get("What experiments do you suggest doing?", [])
            gold_text = "\n".join(experiments).strip()

            references = (
                [Reference(Output(text=gold_text), tags=[CORRECT_TAG])]
                if gold_text else []
            )

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
            ))

        return instances

    # ------------------------------------------------------------------
    # PaperWeakness
    # ------------------------------------------------------------------

    def _load_paper_weakness(self, repo_dir: str) -> List[Instance]:
        task_dir = os.path.join(repo_dir, "Paper_Weakness", "ICLR_2023")
        instances = []

        for paper_id in sorted(os.listdir(task_dir)):
            data_file = os.path.join(task_dir, paper_id, "data_text.json")
            if not os.path.exists(data_file):
                continue

            with open(data_file, encoding="utf-8") as f:
                item = json.load(f)

            # Format paper text: abstract + title + sections
            # (matches subtask3_review_model_prediction.close_source.py)
            paper_input = item.get("input", {})
            abstract = paper_input.get("abstractText", "")
            title = paper_input.get("title", "")
            sections = paper_input.get("sections", [])
            main_text = "\n".join(
                f"{s.get('heading', '')} {s.get('text', '')}".strip()
                for s in sections
            )
            paper_text = f"{abstract}\n\n{title}\n\n{main_text}".strip()

            prompt = (
                "You are an expert in Machine Learning and Natural Language "
                "Processing (NLP). Your responsibility is to help the user "
                "review a paper.\n\n"
                "Below is the content of a research paper:\n"
                f"{paper_text}\n\n"
                "Please identify the weaknesses of this paper. Format your "
                "response as a numbered list. If the given context is irrelevant "
                "to research, just generate 'No research content'."
            )

            # output is a list-of-lists: each inner list is one reviewer's weaknesses.
            # Flatten all reviewers into a single reference string.
            all_weaknesses: List[str] = []
            for reviewer_weaknesses in item.get("output", []):
                all_weaknesses.extend(reviewer_weaknesses)
            gold_text = "\n".join(all_weaknesses).strip()

            references = (
                [Reference(Output(text=gold_text), tags=[CORRECT_TAG])]
                if gold_text else []
            )

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
            ))

        return instances
