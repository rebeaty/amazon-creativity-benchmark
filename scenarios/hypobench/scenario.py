"""
HELM Scenario: HypoBench

Paper: HypoBench: A Novel Benchmark for Evaluating Hypothesis Generation
       by Large Language Models
       Liu et al., 2025. arXiv:2504.11524
Website: https://chicagohai.github.io/HypoBench/
Code:    https://github.com/ChicagoHAI/hypothesis-generation
Data:    https://github.com/ChicagoHAI/HypoBench-datasets

HypoBench evaluates the ability of LLMs to generate plausible hypotheses
explaining observed phenomena across 12 diverse tasks (7 real-world + 5 synthetic).

This scenario implements the 7 real-world tasks:
  - deceptive_reviews : Hotel review authenticity (truthful vs. deceptive)
  - headline_binary   : News headline click prediction (which of two gets more clicks)
  - gptgc_detect      : AI-generated content detection (GPT-written vs. human)
  - llamagc_detect    : AI-generated content detection (LLaMA-written vs. human)
  - dreaddit          : Stress detection from Reddit posts
  - persuasive_pairs  : Persuasive language (which of two arguments is more persuasive)
  - retweet           : Tweet popularity prediction (which of two gets more retweets)

Each instance presents num_observations labeled training examples and asks the model
to generate hypotheses explaining the observed patterns. References are known_hypotheses
drawn from the research literature (from each task's metadata.json).

Prompt format: Extracted from config.yaml batched_generation templates (paper Section 3).
  System: Task-specific expert persona + hypothesis generation instructions
  User:   Labeled observations + "Generate {N} hypotheses."

Fields used:   task-specific text features + label (from train JSON), known_hypotheses (metadata)
Fields skipped: OOD data, val/test splits, prompt field (writing prompts, not task prompts)

Evaluation: open_ended — ROUGE-L / BERTScore against known_hypotheses from literature.
Note: The paper's primary evaluation applies generated hypotheses as decision rules and
measures prediction accuracy on a held-out test set (multi-step pipeline). See
metric_notes.md for HDR (Hypothesis Discovery Rate) used on synthetic tasks.

Note: Synthetic tasks (admission, election, preference, shoe, marine) have exact
ground_truth_hypotheses and require HDR evaluation — see metric_notes.md.
Note: journal_cross and journal_same use nested subdirectory structures and are excluded.
"""

import json
import random
import re
import urllib.request
from typing import Any, Dict, List

import yaml

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

_BASE_URL = "https://raw.githubusercontent.com/ChicagoHAI/HypoBench-datasets/main/real"

VALID_TASKS = [
    "deceptive_reviews",
    "headline_binary",
    "gptgc_detect",
    "llamagc_detect",
    "dreaddit",
    "persuasive_pairs",
    "retweet",
]


class HypoBenchScenario(Scenario):
    name = "hypobench"
    description = "https://github.com/ChicagoHAI/HypoBench-datasets"
    tags = ["creativity", "hypothesis_generation", "scientific_reasoning"]

    def __init__(self, task: str = "deceptive_reviews", num_observations: int = 10, seed: int = 42):
        """
        Args:
            task: Which real-world task to evaluate. One of VALID_TASKS.
            num_observations: Number of labeled training examples to include in the prompt.
            seed: Random seed for sampling observations.
        """
        super().__init__()
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{task}'")
        self.task = task
        self.num_observations = num_observations
        self.seed = seed

    def _fetch(self, url: str) -> str:
        with urllib.request.urlopen(url) as response:
            return response.read().decode("utf-8")

    def _substitute(self, template: str, row: Dict[str, Any]) -> str:
        """Replace ${key} placeholders with values from row dict."""
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: str(row.get(m.group(1), m.group(0))),
            template,
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        base = f"{_BASE_URL}/{self.task}"

        config = yaml.safe_load(self._fetch(f"{base}/config.yaml"))
        metadata = json.loads(self._fetch(f"{base}/metadata.json"))

        templates = config.get("prompt_templates", {})
        obs_template = templates.get("observations", {}).get("multi_content", "").strip()
        batched = templates.get("batched_generation", {})
        system_prompt = batched.get("system", "").strip()
        user_template = batched.get("user", "").strip()

        # Load training data (columnar format: {field: [val, ...]})
        train_filename = config.get("train_data_path", "").lstrip("./")
        raw = json.loads(self._fetch(f"{base}/{train_filename}"))
        keys = list(raw.keys())
        n = len(raw[keys[0]])
        rows = [{k: raw[k][i] for k in keys} for i in range(n)]

        # Sample observations
        rng = random.Random(self.seed)
        sample = rng.sample(rows, min(self.num_observations, len(rows)))
        observations = "\n\n".join(
            self._substitute(obs_template, row).strip() for row in sample
        )

        # Build full prompt
        num_hypotheses = 5
        user_prompt = self._substitute(user_template, {
            "observations": observations,
            "num_hypotheses": str(num_hypotheses),
        })
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Known hypotheses from literature as references
        hyp_key = "known_hypotheses" if "known_hypotheses" in metadata else "ground_truth_hypotheses"
        references = [
            Reference(Output(text=h), tags=[CORRECT_TAG])
            for h in metadata.get(hyp_key, [])
        ]

        return [
            Instance(
                input=Input(text=full_prompt),
                references=references,
                split=TEST_SPLIT,
            )
        ]
