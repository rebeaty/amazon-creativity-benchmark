"""
HELM Scenario: GraphRAG-Bench (Creative Generation subset)

Paper: "When to use Graphs in RAG: A Comprehensive Analysis for Graph
       Retrieval-Augmented Generation" (arXiv:2506.05690)
Data: https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench

GraphRAG-Bench evaluates Graph Retrieval-Augmented Generation systems across
four question types and two knowledge domains (novel literature, medical).
This scenario implements the Creative Generation subset — tasks that require
the model to produce imaginative responses grounded in source material context
(e.g., "Write a diary entry as a fish witnessing John Curgenven's boat").

The dataset has 233 Creative Generation questions total:
  - novel:   67 questions (from 20 literary novels)
  - medical: 166 questions (from medical knowledge corpus)

Note: The benchmark originally targets GraphRAG pipelines (retrieval context
is fetched at inference time from a knowledge graph). In this HELM scenario
we evaluate zero-shot generation from the question alone, matching the
creative capability evaluation described in the paper.

Note: The HuggingFace datasets library fails to load this dataset due to
inconsistent evidence_triple/evidence_relations column schemas. Raw JSON
files are fetched directly from the HuggingFace repository.

Other question types (Fact Retrieval, Complex Reasoning, Contextual
Summarize) are available in the raw JSON files but are not included here
as they do not constitute creativity tasks.

Prompt source: No explicit prompt template specified in the paper;
  the question field is used directly as the model input.

Fields used: question, answer, question_type, source, id
Fields skipped: evidence, evidence_triple/evidence_relations (retrieval
  support fields, not part of model input in zero-shot setting)

Evaluation: LLM-as-judge (answer_correctness + coverage_score via RAGAS).
  See annotator_notes.md for details.
"""

import json
import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

QUESTION_URLS = {
    "novel": (
        "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench"
        "/resolve/main/Datasets/Questions/novel_questions.json"
    ),
    "medical": (
        "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench"
        "/resolve/main/Datasets/Questions/medical_questions.json"
    ),
}


class GraphRAGBenchScenario(Scenario):
    name = "graphrag_bench"
    description = "GraphRAG-Bench/GraphRAG-Bench"
    tags = ["creativity", "generation", "grounded_generation"]

    def __init__(self, domain: str = "novel"):
        """
        Args:
            domain: "novel", "medical", or "all".
                    Selects which knowledge domain(s) to include.
        """
        super().__init__()
        assert domain in ("novel", "medical", "all"), (
            f"domain must be 'novel', 'medical', or 'all', got '{domain}'"
        )
        self.domain = domain

    def _download(self, url: str, dest: str) -> None:
        if not os.path.exists(dest):
            urllib.request.urlretrieve(url, dest)

    def get_instances(self, output_path: str) -> List[Instance]:
        domains = ["novel", "medical"] if self.domain == "all" else [self.domain]

        instances = []
        for domain in domains:
            dest = os.path.join(output_path, f"{domain}_questions.json")
            self._download(QUESTION_URLS[domain], dest)

            with open(dest) as f:
                items = json.load(f)

            for item in items:
                if item["question_type"] != "Creative Generation":
                    continue

                instances.append(
                    Instance(
                        input=Input(text=item["question"]),
                        references=[
                            Reference(Output(text=item["answer"]), tags=[CORRECT_TAG])
                        ],
                        split=TEST_SPLIT,
                        extra_data={
                            "id": item["id"],
                            "domain": domain,
                            "source": item["source"],
                            "question_type": item["question_type"],
                        },
                    )
                )

        return instances
