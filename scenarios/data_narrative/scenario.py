"""
HELM Scenario: DataNarrative

Paper: DataNarrative: Automated Data-Driven Storytelling with Visualizations and Texts
       Mohammed Saidul Islam, Md Tahmid Rahman Laskar, Md Rizwan Parvez, Enamul Hoque, Shafiq Joty
       EMNLP 2024
       https://arxiv.org/abs/2408.05346
       https://aclanthology.org/2024.emnlp-main.1073/

Code/Data: https://github.com/saidul-islam98/DataNarrative

Task: Generate narrative text describing data insights from tabular data.
      Given a data table and a topic/intent, models must produce coherent narrative
      paragraphs that accurately describe trends, patterns, and key statistics.

Multi-Stage Framework (paper describes iterative refinement):
  1. Reflection generation (Figure 18) - analyze data tables
  2. Reflection revision (Figures 19-20) - check factual accuracy
  3. Outline generation (Figure 21) - create story structure
  4. Outline revision (Figures 22-23) - verify theme consistency
  5. Narration generation (Figure 24) - write final story
  6. Narration revision (Figures 25-26) - final refinement

For HELM evaluation, we use simplified single-turn generation:

Prompt format (simplified from Figure 24):
  Generate a narrative paragraph describing insights from the following data table.
  Focus on key trends, patterns, and significant data points.

  Topic: {topic_name}
  Intent: {intent}

  Data Table:
  {table}

  Narrative:

Fields used: table, paragraph (reference), topic_name, intent
Fields available but not used:
  - chart (image filename - multimodal extension possible)
  - vis_spec (visualization JSON - not needed for text-only task)
  - annotation (data point highlights)
  - reflection, outline, narration_int (intermediate outputs from multi-agent framework)

Evaluation:
  - Primary: Open-ended generation (BLEU, ROUGE) comparing to reference paragraphs
  - Paper evaluation: Model-based (GPT-4 judge) and human evaluation on:
    * Factual accuracy
    * Coherence
    * Comprehensiveness
    * Theme consistency

Dataset: 1,917 examples from 3 sources (Tableau, Pew Research, GapMinder)
  - Train: 226 examples (Tableau)
  - Test: 1,691 examples
    * GapMinder: 42 (demographic/economic trends)
    * Pew: 1,590 (social/political topics)
    * Tableau: 59 (various domains)

Note: Each story may have multiple paragraph-table pairs. We create one instance
      per paragraph-table segment for fine-grained evaluation.
"""

import json
import os
from typing import List
from urllib.request import urlopen

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
)
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class DataNarrativeScenario(Scenario):
    """
    DataNarrative: Automated data-driven storytelling benchmark.

    Evaluates LLM ability to generate narrative text from tabular data,
    focusing on factual accuracy and coherent explanation of data insights.
    """

    name = "data_narrative"
    description = "saidul-islam98/DataNarrative"
    tags = ["creativity", "data_storytelling", "text_generation", "data_to_text"]

    # GitHub raw content URLs
    GITHUB_BASE = "https://raw.githubusercontent.com/saidul-islam98/DataNarrative/main"

    TRAIN_FILES = {
        "tableau": "Train/Tableau/tableau_train.json"
    }

    TEST_FILES = {
        "gapminder": "Test/GapMinder/gapminder_test.json",
        "pew": "Test/Pew/pew_test.json",
        "tableau": "Test/Tableau/tableau_test.json"
    }

    def __init__(self, source: str = "all"):
        """
        Args:
            source: Which test source to use. Options: ["gapminder", "pew", "tableau", "all"]
                   "all" = All test sources (1,691 examples)
                   "gapminder" = GapMinder only (42 examples)
                   "pew" = Pew Research only (1,590 examples)
                   "tableau" = Tableau only (59 examples)
        """
        super().__init__()
        if source not in ["gapminder", "pew", "tableau", "all"]:
            raise ValueError(f"Invalid source: {source}. Must be 'gapminder', 'pew', 'tableau', or 'all'")
        self.source = source

    def _build_prompt(self, topic_name: str, intent: str, table: str) -> str:
        """
        Build simplified prompt for narrative generation.
        Based on Figure 24 but adapted for single-turn generation.
        """
        return (
            f"Generate a narrative paragraph describing insights from the following data table.\n"
            f"Focus on key trends, patterns, and significant data points.\n\n"
            f"Topic: {topic_name}\n"
            f"Intent: {intent}\n\n"
            f"Data Table:\n{table}\n\n"
            f"Narrative:"
        )

    def _download_json(self, url: str, local_path: str) -> dict:
        """Download and parse JSON file from GitHub."""
        ensure_file_downloaded(source_url=url, target_path=local_path)
        with open(local_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_instances(self, data: dict, split: str, source_name: str) -> List[Instance]:
        """Extract instances from a DataNarrative JSON file."""
        instances = []

        # Get metadata
        topic_names = data.get("topic_name", {})
        intents = data.get("intent", {})
        paragraph_table_pairs = data.get("paragraph_table_pair", {})

        # Iterate through all story IDs
        for story_id in paragraph_table_pairs.keys():
            topic = topic_names.get(story_id, "Unknown Topic")
            intent = intents.get(story_id, "")

            # Each story has multiple paragraph-table segments
            segments = paragraph_table_pairs[story_id]

            for seg_idx, segment in enumerate(segments):
                content = segment.get("content", {})

                paragraph = content.get("paragraph", "")
                table = content.get("table", "")

                # Skip if missing critical fields
                if not paragraph or not table:
                    continue

                # Build prompt
                prompt = self._build_prompt(topic, intent, table)

                # Create instance
                instance_id = f"{source_name}_{story_id}_{seg_idx}"

                instances.append(Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=paragraph), tags=[])],
                    split=split,
                    id=instance_id
                ))

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load DataNarrative dataset and create HELM instances."""
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []

        # Load training data
        for source_name, file_path in self.TRAIN_FILES.items():
            url = f"{self.GITHUB_BASE}/{file_path}"
            local_path = os.path.join(data_path, f"train_{source_name}.json")

            data = self._download_json(url, local_path)
            instances.extend(self._extract_instances(data, TRAIN_SPLIT, f"train_{source_name}"))

        # Load test data based on source parameter
        test_files = {}
        if self.source == "all":
            test_files = self.TEST_FILES
        else:
            test_files = {self.source: self.TEST_FILES[self.source]}

        for source_name, file_path in test_files.items():
            url = f"{self.GITHUB_BASE}/{file_path}"
            local_path = os.path.join(data_path, f"test_{source_name}.json")

            data = self._download_json(url, local_path)
            instances.extend(self._extract_instances(data, TEST_SPLIT, f"test_{source_name}"))

        return instances
