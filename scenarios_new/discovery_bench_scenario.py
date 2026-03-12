"""
HELM Scenario: DiscoveryBench

Paper: DiscoveryBench: Towards Data-Driven Discovery with Large Language Models (ICLR 2025)
       https://arxiv.org/abs/2407.01725
Code: https://github.com/allenai/discoverybench

DiscoveryBench evaluates LLMs on data-driven scientific discovery: given domain
knowledge, dataset metadata, and a research question, the model must generate
a hypothesis about relationships in the data.

Prompt format:
  You are a data scientist tasked with discovering patterns and relationships
  in data.

  Domain: {domain}
  Background Knowledge: {domain_knowledge}

  Available Data:
  {dataset descriptions with column names and types}

  Discovery Question: {question}

  Based on the domain knowledge and dataset information provided, formulate
  a specific hypothesis...

Prompt source: Standard format (paper evaluates via code execution pipeline;
  adapted here for text-only hypothesis generation)
Fields used: domain, domain_knowledge, datasets (name, description, columns), queries (question, true_hypothesis)
Fields skipped: workflow_tags (internal pipeline metadata)

Notes:
  - Real test split has no ground truth (held out for leaderboard), using real train (25 instances)
  - Prompts include dataset schema (column names + descriptions) but not actual data rows
  - Gold hypotheses may contain specific coefficients derived from data analysis;
    evaluation captures variable identification and relationship direction
  - Domains: sociology, biology, humanities, economics, engineering, meta science
"""

import json
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class DiscoveryBenchScenario(Scenario):
    name = "discovery_bench"
    description = "allenai/discoverybench"
    tags = ["creativity", "scientific_reasoning", "hypothesis_generation"]

    def _format_datasets(self, datasets_json: str) -> str:
        """Format dataset metadata into a readable description."""
        datasets_info = json.loads(datasets_json)
        parts = []
        for ds in datasets_info:
            lines = [f"Dataset: {ds['name']}"]
            if ds.get("description"):
                lines.append(f"Description: {ds['description']}")
            if "columns" in ds and "raw" in ds["columns"]:
                lines.append("Columns:")
                for col in ds["columns"]["raw"]:
                    desc = col.get("description", "")
                    if desc:
                        lines.append(f"  - {col['name']}: {desc}")
                    else:
                        lines.append(f"  - {col['name']}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def get_instances(self, output_path: str):
        # Real test split has no ground truth; use train split (25 instances with gold hypotheses)
        dataset = load_dataset("allenai/discoverybench", split="train")

        instances = []
        for item in dataset:
            queries = json.loads(item["queries"])
            dataset_desc = self._format_datasets(item["datasets"])

            for query in queries:
                hypothesis = query.get("true_hypothesis")
                if not hypothesis:
                    continue

                prompt = (
                    "You are a data scientist tasked with discovering patterns "
                    "and relationships in data.\n\n"
                    f"Domain: {item['domain']}\n\n"
                    f"Background Knowledge:\n{item['domain_knowledge']}\n\n"
                    f"Available Data:\n{dataset_desc}\n\n"
                    f"Discovery Question: {query['question']}\n\n"
                    "Based on the domain knowledge and dataset information provided, "
                    "formulate a specific hypothesis about the relationship or pattern "
                    "in the data. Include the relevant variables and the nature of "
                    "their relationship."
                )

                references = [
                    Reference(Output(text=hypothesis), tags=[CORRECT_TAG])
                ]

                instances.append(Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                ))

        return instances
