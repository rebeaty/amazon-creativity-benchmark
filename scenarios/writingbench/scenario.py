"""
HELM Scenario: WritingBench

Paper: Not yet published (from X-PLUG team)
Code: https://github.com/X-PLUG/WritingBench
Dataset: GitHub repository (no HuggingFace dataset available)

Task: Evaluates LLM writing capabilities across 1,000 real-world queries spanning
6 primary domains and 100 fine-grained subdomains. Covers diverse writing scenarios
including academic papers, business documents, legal texts, literary works, educational
materials, and marketing content.

Dataset composition:
  - 1,000 queries total (445 Chinese, 555 English)
  - 6 primary domains:
    * Academic & Engineering (167 queries)
    * Finance & Business (210 queries)
    * Politics & Law (201 queries)
    * Literature & Arts (183 queries)
    * Education (111 queries)
    * Advertising & Marketing (128 queries)
  - 100 fine-grained subdomains (e.g., Paper Outline, Abstract, Introduction,
    Contract Drafting, Literary Analysis, etc.)

Evaluation: LLM-as-judge using rubric-based scoring. Each query has 5 instance-specific
evaluation criteria with detailed scoring rubrics (1-2, 3-4, 5-6, 7-8, 9-10 point ranges).
Evaluators assign a 10-point scale score per criterion with justifications.

Judge models: Claude models (LLM-as-a-Judge) or finetuned critic model (paper provides both)

Prompt format:
  No standardized template - each query is a complete writing task request in natural language.
  Queries are domain-specific and include context, requirements, and constraints.

Prompt source: Queries are original contributions from the benchmark paper, designed to
reflect real-world writing scenarios across multiple domains.

Fields used: index, domain1, domain2, lang, query, checklist
Fields skipped: None (all fields are metadata or evaluation criteria)

Note: This is an LLM-as-judge benchmark with instance-specific evaluation criteria.
The 'checklist' field contains 5 criteria per query, each with a name, description,
and detailed scoring rubrics for 5 score ranges (1-2, 3-4, 5-6, 7-8, 9-10).
References are empty as this is a pure generation task evaluated by judges.

Evaluation methodology detailed in repository's prompt.py and evaluate_benchmark.py files.
"""

import json
import os
import tempfile
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Reference,
    TEST_SPLIT
)
from helm.common.general import ensure_file_downloaded


class WritingBenchScenario(Scenario):
    name = "writingbench"
    description = "X-PLUG/WritingBench"
    tags = ["creativity", "writing", "multilingual", "open_ended"]

    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/X-PLUG/WritingBench/main"
    DATA_FILE = "benchmark_query/benchmark_all.jsonl"

    def __init__(
        self,
        domain: str = "all",
        language: str = "all"
    ):
        """
        Args:
            domain: Filter by primary domain. Options:
                - "all": All 1,000 queries (default)
                - "academic": Academic & Engineering (167 queries)
                - "finance": Finance & Business (210 queries)
                - "politics": Politics & Law (201 queries)
                - "literature": Literature & Arts (183 queries)
                - "education": Education (111 queries)
                - "marketing": Advertising & Marketing (128 queries)
            language: Filter by language. Options:
                - "all": Both languages (1,000 queries, default)
                - "en": English only (555 queries)
                - "zh": Chinese only (445 queries)
        """
        super().__init__()
        self.domain = domain
        self.language = language

        # Domain mappings
        self.domain_mapping = {
            "academic": "Academic & Engineering",
            "finance": "Finance & Business",
            "politics": "Politics & Law",
            "literature": "Literature & Arts",
            "education": "Education",
            "marketing": "Advertising & Marketing"
        }

    def download_dataset(self, output_path: str) -> str:
        """Download the dataset from GitHub."""
        data_url = f"{self.GITHUB_RAW_BASE}/{self.DATA_FILE}"
        data_file = os.path.join(output_path, "benchmark_all.jsonl")

        ensure_file_downloaded(
            source_url=data_url,
            target_path=data_file
        )

        return data_file

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download dataset
        data_file = self.download_dataset(output_path)

        # Load all examples
        instances = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                # Apply domain filter
                if self.domain != "all":
                    expected_domain = self.domain_mapping.get(self.domain)
                    if item['domain1'] != expected_domain:
                        continue

                # Apply language filter
                if self.language != "all":
                    if item['lang'] != self.language:
                        continue

                # Extract query (the writing task)
                query = item['query']

                # Create instance with empty references (open-ended generation)
                # Evaluation will be done by LLM-as-judge using the checklist criteria
                instance = Instance(
                    input=Input(text=query),
                    references=[],
                    split=TEST_SPLIT
                )

                # Store metadata for evaluation
                instance.extra_data = {
                    "index": item['index'],
                    "domain1": item['domain1'],
                    "domain2": item['domain2'],
                    "language": item['lang'],
                    "checklist": item['checklist']  # 5 evaluation criteria with rubrics
                }

                instances.append(instance)

        return instances
