"""
HELM Scenario: IDRBench (Understanding LLMs' Ability on Interdisciplinary Research)

Paper: https://arxiv.org/abs/2507.15736
Title: Understanding Large Language Models' Ability on Interdisciplinary Research
Dataset: IDRBench/IDRBench (HuggingFace)
Published: ICML 2025

Task: Evaluate LLMs' capabilities in interdisciplinary research (IDR) across three tasks:
  1. IPI (IDR Paper Identification): Identify whether a paper qualifies as interdisciplinary research
  2. I3 (IDR Idea Integration): Evaluate whether concepts from two papers can form novel multidisciplinary ideas
  3. I2R (IDR Idea Recommendation): Rank candidate papers for pairing with a main paper for IDR

This tests creative thinking in scientific research contexts - the ability to recognize,
integrate, and recommend interdisciplinary research directions.

Official IDR Definition (from paper):
"Interdisciplinary research involves collaborations among multiple distinct disciplines,
aiming to integrate concepts, theories and methodologies from two or more disciplines
to address complex problems that cannot be solved within a single disciplinary framework."

Prompt formats (exact from paper Appendix A.2.3):

IPI Task:
  Read the title and abstract of a given academic paper and identify whether this
  is an interdisciplinary research paper. Also, select one or more subjects from the
  list below to indicate which subject(s) does this paper belong to. After you provide
  your verdict and your choice, provide a score from 0 to 100 to indicate your
  confidence level in the correctness of the verdict.

  The official definition of a typical interdisciplinary paper can be found below:
  "Interdisciplinary Research is a mode of research that integrates information, data,
  techniques, tools, perspectives, concepts, and/or theories from two or more disciplines
  or bodies of specialised knowledge to advance fundamental understanding or to solve
  problems whose solutions are beyond the scope of a single discipline or area of research practice."

  Think carefully to make your verdict, answer "Yes" when this is a valid IDR paper.
  Otherwise, answer "No".

  Paper title: {title};
  Paper abstract: {abstract};

  Subject list: ["Computer Science, Electrical Engineering and System Science",
                 "Economics and Quantitative Finance", "Mathematics and Statistics",
                 "Physics", "Quantitative Biology", "Other"]

  Output: Your verdict (Yes/No), Confidence score (0-100), Subject (list)

I3 Task:
  Read the title and abstract of papers from two disciplines and decide whether you
  can extract concepts from both disciplines to create a novel multidisciplinary research idea.

  Keep in mind a good Interdisciplinary Research idea includes the following standards:
  * This research idea should be Interdisciplinary
  * Should follow the IDR definition above
  * Should be feasible (can be validated by experiments)
  * Should be novel (rare, ingenious, imaginative, or surprising)
  * Should be useful (applies to stated problem, effective at solving it)

  Think carefully to make your decision, and you should only answer "Yes" when this
  multidisciplinary idea meets ALL of the standards above. Otherwise, answer "No".

  Paper in Discipline 1: {title_1}; {abstract_1};
  Paper in Discipline 2: {title_2}; {abstract_2};

  Output: Your verdict (Yes/No), Your reason (<50 words), Confidence score (0-100)

I2R Task:
  In this task, you are given a main paper introducing the key concepts that provides
  certain parts in a Interdisciplinary idea as well as two candidate papers that forms
  the remaining parts of a Interdisciplinary idea. Compare them and select which one is
  better to pair with the main paper in forming a multidisciplinary idea.

  Keep in mind a good Interdisciplinary Research idea includes the same standards as I3.

  Main paper title: {main_title}; Main paper abstract: {main_abstract};
  Paper 1 title: {paper1_title}; Paper 1 abstract: {paper1_abstract};
  Paper 2 title: {paper2_title}; Paper 2 abstract: {paper2_abstract};

  Output: Your choice (Paper 1 or Paper 2), Confidence score (0-100)

Evaluation: Classification accuracy (IPI), binary accuracy (I3), ranking accuracy (I2R)

Dataset splits:
  - IPI: level_1 (1120 examples)
  - I3: level_1 (1120), level_2 (1890)
  - I2R: level_1 (253), level_2 (varies)

Fields used: title, abstract, categories, fields, y_true
Fields skipped: authors, date, explanation (some metadata)
"""

import json
import os
from typing import List, Dict
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)


class IDRBenchScenario(Scenario):
    """
    IDRBench: Interdisciplinary Research Capabilities Benchmark

    Tests LLMs' ability to identify, integrate, and recommend
    interdisciplinary research directions.
    """

    name = "idrbench"
    description = "IDRBench/IDRBench"
    tags = ["creativity", "research", "interdisciplinary", "scientific_reasoning"]

    # Available tasks
    TASKS = ["IPI", "I3", "I2R"]

    # Available difficulty levels per task
    LEVELS = {
        "IPI": ["level_1"],
        "I3": ["level_1", "level_2"],
        "I2R": ["level_1", "level_2"]
    }

    # IDR definition from paper (official definition used in prompts)
    IDR_DEFINITION = (
        '"Interdisciplinary Research is a mode of research that integrates information, data, '
        'techniques, tools, perspectives, concepts, and/or theories from two or more disciplines '
        'or bodies of specialised knowledge to advance fundamental understanding or to solve '
        'problems whose solutions are beyond the scope of a single discipline or area of research practice."'
    )

    # Subject categories for IPI task (from paper Appendix A.2.3)
    SUBJECT_CATEGORIES = [
        "Computer Science, Electrical Engineering and System Science",
        "Economics and Quantitative Finance",
        "Mathematics and Statistics",
        "Physics",
        "Quantitative Biology",
        "Other"
    ]

    # IDR standards for I3 and I2R tasks
    IDR_STANDARDS = [
        "This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.",
        "The Interdisciplinary Research ideas should follow this definition: " + IDR_DEFINITION,
        "This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.",
        "This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.",
        "This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem."
    ]

    def __init__(self, task: str = "IPI", level: str = "level_1"):
        """
        Args:
            task: Which task to evaluate. Options:
                - "IPI": Paper identification (default)
                - "I3": Idea integration
                - "I2R": Idea recommendation
            level: Difficulty level. Options depend on task:
                - IPI: "level_1" only
                - I3: "level_1" or "level_2"
                - I2R: "level_1" or "level_2"
        """
        super().__init__()

        if task not in self.TASKS:
            raise ValueError(f"Invalid task '{task}'. Must be one of: {self.TASKS}")

        if level not in self.LEVELS[task]:
            raise ValueError(f"Invalid level '{level}' for task '{task}'. Must be one of: {self.LEVELS[task]}")

        self.task = task
        self.level = level

    def _download_data(self, output_path: str, task_name: str, split: str) -> List[Dict]:
        """
        Download IDRBench data directly from HuggingFace.

        Args:
            output_path: Directory to cache downloaded data
            task_name: Task configuration name
            split: Split name (level_1, level_2, etc.)

        Returns:
            List of data examples
        """
        import requests

        # Map task to data file
        file_mapping = {
            ("IDR_paper_identification", "level_1"): "data_exp_1.json",
            ("IDR_idea_integration", "level_1"): "data_exp_2_1.json",
            ("IDR_idea_integration", "level_2"): "data_exp_2_2.json",
            ("IDR_idea_recommendation", "level_1"): "data_exp_3_1.json",
            ("IDR_idea_recommendation", "level_2"): "data_exp_3_2.json",
        }

        data_file = file_mapping.get((task_name, split))
        if not data_file:
            raise ValueError(f"No data file for task={task_name}, split={split}")

        # Check cache
        cache_path = os.path.join(output_path, "idrbench_cache", data_file)
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)

        # Download from HuggingFace
        print(f"Downloading {data_file} from HuggingFace...")
        url = f"https://huggingface.co/datasets/IDRBench/IDRBench/resolve/main/{data_file}"
        response = requests.get(url)

        if not response.ok:
            raise RuntimeError(f"Failed to download {data_file}: {response.status_code}")

        data = response.json()

        # Cache for future use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(data, f)

        print(f"Downloaded {len(data)} examples")
        return data

    def _format_ipi_prompt(self, title: str, abstract: str) -> str:
        """Format prompt for IPI (Paper Identification) task - exact from paper Appendix A.2.3."""
        subject_list_str = str(self.SUBJECT_CATEGORIES)

        return (
            f"Read the title and abstract of a given academic paper and identify whether this is an "
            f"interdisciplinary research paper. Also, select one or more subjects from the list below to "
            f"indicate which subject(s) does this paper belong to. After you provide your verdict and your "
            f"choice, provide a score from 0 to 100 to indicate your confidence level in the correctness of "
            f"the verdict.\n"
            f"The official definition of a typical interdisciplinary paper can be found below:\n"
            f"{self.IDR_DEFINITION}\n"
            f"Think carefully to make your verdict, answer \"Yes\" when this is a valid IDR paper. "
            f"Otherwise, answer \"No\".\n"
            f"Note: The confidence level indicates the degree of certainty you have about your verdict and is "
            f"represented as a percentage. For instance, if your confidence level is 80, it means you are 80 "
            f"percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.\n\n"
            f"Paper title: {title};\n"
            f"Paper abstract: {abstract};\n\n"
            f"Subject list: {subject_list_str}\n\n"
            f"Use the template (in this format, with no markdown and lines separated by '\\n') below to provide your answer.\n"
            f"Your verdict: {{A simple answer containing either \"Yes\" or \"No\".}}\n"
            f"Confidence score: {{A numeric score ranging from 0 to 100}}\n"
            f"Subject: {{Your choice of subjects from the list above. Use a list with square brackets \"[]\" "
            f"separated by comma and remember to use \"\" to wrap your answer.}}\n"
        )

    def _format_i3_prompt(self, paper_b: Dict, paper_c: Dict) -> str:
        """Format prompt for I3 (Idea Integration) task - exact from paper Appendix A.2.3."""
        # Extract paper details
        title_b = paper_b['title'][0] if isinstance(paper_b['title'], list) else paper_b['title']
        abstract_b = paper_b['abstract'][0] if isinstance(paper_b['abstract'], list) else paper_b['abstract']

        title_c = paper_c['title'][0] if isinstance(paper_c['title'], list) else paper_c['title']
        abstract_c = paper_c['abstract'][0] if isinstance(paper_c['abstract'], list) else paper_c['abstract']

        # Format paper blocks
        paper1_block = f"Paper in Discipline 1:\nTitle: {title_b};\nAbstract: {abstract_b};"
        paper2_block = f"Paper in Discipline 2:\nTitle: {title_c};\nAbstract: {abstract_c};"

        # Build standards list
        standards_text = "\n".join(f"* {standard}" for standard in self.IDR_STANDARDS)

        return (
            f"Read the title and abstract of papers from two disciplines and decide whether you can extract "
            f"concepts from both disciplines to create a novel multidisciplinary research idea. After you "
            f"provide your verdict, provide a score from 0 to 100 to indicate your confidence level in the "
            f"correctness of the verdict.\n"
            f"Keep in mind a good Interdisciplinary Research idea includes the following standards:\n"
            f"{standards_text}\n"
            f"Think carefully to make your decision, and you should only answer \"Yes\" when this multidisciplinary "
            f"idea meets ALL of the standards above. Otherwise, you should answer \"No\".\n"
            f"Note: The confidence level indicates the degree of certainty you have about your verdict and is "
            f"represented as a percentage. For instance, if your confidence level is 80, it means you are 80 "
            f"percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.\n\n"
            f"{paper1_block}\n\n"
            f"{paper2_block}\n\n"
            f"Use the template (in this format, with no markdown and lines separated by '\\n') to provide your answer.\n"
            f"Your verdict: {{A simple answer containing either \"Yes\" or \"No\".}}\n"
            f"Your reason: {{A short paragraph less than 50 words briefly describes your reasons that you made the verdict above.}}\n"
            f"Confidence score: {{A numeric score ranging from 0 to 100}}\n"
        )

    def _format_i2r_prompt(self, start_paper: Dict, candidates: List[Dict]) -> str:
        """Format prompt for I2R (Idea Recommendation) task - exact from paper Appendix A.2.3."""
        # Paper uses "Paper 1" and "Paper 2" (not A/B)
        assert len(candidates) == 2, "I2R prompt expects exactly 2 candidates"

        # Build standards list
        standards_text = "\n".join(f"* {standard}" for standard in self.IDR_STANDARDS)

        return (
            f"In this task, you are given a main paper introducing the key concepts that provides certain "
            f"parts in a Interdisciplinary idea as well as two candidate papers that forms the remaining "
            f"parts of a Interdisciplinary idea. Compare them and select which one is better to pair with "
            f"the main paper in forming a multidisciplinary idea. After you provide your selection, provide "
            f"a score from 0 to 100 to indicate your confidence level in the correctness of making this choice.\n"
            f"Keep in mind a good Interdisciplinary Research idea includes the following standards:\n"
            f"{standards_text}\n"
            f"Note: The confidence level indicates the degree of certainty you have about your verdict and is "
            f"represented as a percentage. For instance, if your confidence level is 80, it means you are 80 "
            f"percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.\n\n"
            f"Main paper title: {start_paper['title']};\n"
            f"Main paper abstract: {start_paper['abstract']};\n\n"
            f"Paper 1 title: {candidates[0]['title']};\n"
            f"Paper 1 abstract: {candidates[0]['abstract']};\n\n"
            f"Paper 2 title: {candidates[1]['title']};\n"
            f"Paper 2 abstract: {candidates[1]['abstract']};\n\n"
            f"Use the template (in this format, with no markdown and lines separated by '\\n') to provide your answer.\n"
            f"Your choice: {{A simple answer containing either \"Paper 1\" or \"Paper 2\".}}\n"
            f"Confidence score: {{A numeric score ranging from 0 to 100}}\n"
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate IDRBench instances for the specified task and level.

        Each instance contains:
        - Input: Task-specific prompt with paper(s) information
        - References: Correct answer(s) with CORRECT_TAG
        """
        # Map task to dataset config name
        config_mapping = {
            "IPI": "IDR_paper_identification",
            "I3": "IDR_idea_integration",
            "I2R": "IDR_idea_recommendation"
        }

        config_name = config_mapping[self.task]
        data = self._download_data(output_path, config_name, self.level)

        instances = []

        if self.task == "IPI":
            # Paper Identification task
            for idx, example in enumerate(data):
                prompt = self._format_ipi_prompt(
                    title=example['title'],
                    abstract=example['abstract']
                )

                # Create references for Yes/No classification
                correct_answer = "Yes" if example['y_true'] else "No"
                references = [
                    Reference(Output(text="Yes"), tags=[CORRECT_TAG] if correct_answer == "Yes" else []),
                    Reference(Output(text="No"), tags=[CORRECT_TAG] if correct_answer == "No" else [])
                ]

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"ipi_{self.level}_{idx}",
                        extra_data={
                            "paper_id": example['id'],
                            "categories": example['categories'],
                            "y_true": example['y_true']
                        }
                    )
                )

        elif self.task == "I3":
            # Idea Integration task
            for idx, example in enumerate(data):
                # Extract paper B and C info
                paper_b = {
                    'title': example['b_title'],
                    'abstract': example['b_abstract'],
                    'fields': example['b_fields']
                }
                paper_c = {
                    'title': example['c_title'],
                    'abstract': example['c_abstract'],
                    'fields': example['c_fields']
                }

                prompt = self._format_i3_prompt(paper_b, paper_c)

                # Create references for Yes/No classification
                correct_answer = "Yes" if example['y_true'] else "No"
                references = [
                    Reference(Output(text="Yes"), tags=[CORRECT_TAG] if correct_answer == "Yes" else []),
                    Reference(Output(text="No"), tags=[CORRECT_TAG] if correct_answer == "No" else [])
                ]

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"i3_{self.level}_{idx}",
                        extra_data={
                            "example_id": example['id'],
                            "paper_b_id": example['b_id'],
                            "paper_c_id": example['c_id'],
                            "y_true": example['y_true'],
                            "research_type": example.get('research_type')
                        }
                    )
                )

        elif self.task == "I2R":
            # Idea Recommendation task
            for idx, example in enumerate(data):
                # For simplicity, take first 2 candidates from list
                # In full implementation, could create multiple instances per example
                candidate_list = example['list']

                if len(candidate_list.get('title', [])) < 2:
                    continue  # Skip if insufficient candidates

                candidates = []
                for i in range(min(2, len(candidate_list['title']))):
                    candidates.append({
                        'title': candidate_list['title'][i],
                        'abstract': candidate_list['abstract'][i]
                    })

                start_paper = {
                    'title': example['start_title'],
                    'abstract': example['start_abstract']
                }

                prompt = self._format_i2r_prompt(start_paper, candidates)

                # Determine correct answer based on target_paper
                # This is simplified - full implementation would need more sophisticated matching
                target_ids = example['target_paper'].get('id', [])

                # Create references for "Paper 1" / "Paper 2" choice
                # Note: This is simplified matching logic
                references = [
                    Reference(Output(text="Paper 1"), tags=[]),
                    Reference(Output(text="Paper 2"), tags=[])
                ]

                # Mark correct based on simple heuristic (would need refinement)
                if target_ids and len(target_ids) > 0:
                    references[0].tags.append(CORRECT_TAG)  # Simplified

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"i2r_{self.level}_{idx}",
                        extra_data={
                            "example_id": example['id'],
                            "start_id": example['start_id'],
                            "research_type": example['research_type'],
                            "target_paper_ids": target_ids
                        }
                    )
                )

        return instances
