"""
HELM Scenario: TinyFabulist (Fable Generation Evaluation Framework)

Paper: "TF1-EN-3M: Three Million Synthetic Moral Fables for Training
       Small, Open Language Models"
       https://arxiv.org/abs/2504.20605
Code: https://github.com/klusai/tinyfabulist

Prompt format:
  System message sets the role as a creative fable writer.
  Each prompt specifies a 6-slot fable scaffold:
    - Main Character: a {trait} {character}
    - Setting: a {setting}
    - Challenge: {conflict}
    - Outcome: {resolution}
    - Teaching: {moral}
  Plus formatting instructions (age group, vocabulary, ~250 words).

Prompt source: Generator prompt template in tinyfabulist/conf/generator_prompts.yaml;
  100 benchmark prompts materialized in data/14.04.2025/prompts/tf_prompts_c100_*.jsonl
  (the exact set used to evaluate 11 models in the paper, Table 1).
Fields used: prompt (generator prompt with narrative slots)
Fields skipped: fable (model-generated output), llm_name, llm_*_tokens,
                llm_inference_time, host_*, generation_datetime, pipeline_version

Source: "benchmark" mode uses the 100 curated prompts from the GitHub repo
(the exact evaluation set with published baselines for 11 models).
Source: "full" mode uses the 100K test split from HuggingFace for larger-scale eval.

Evaluation: LLM-as-judge on 4 dimensions (Grammar & Style, Creativity &
Originality, Moral Clarity, Adherence to Prompt) on a 1-10 scale, plus
age group classification (A-E). See annotator_notes.md.
"""

import json
import os
import urllib.request
import zipfile

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    TEST_SPLIT,
)

REPO_ZIP_URL = (
    "https://github.com/klusai/tinyfabulist/archive/refs/heads/main.zip"
)

# Relative path within the extracted repo to the 100-prompt benchmark file
BENCHMARK_PROMPTS_GLOB = (
    "tinyfabulist-main/data/14.04.2025/prompts/tf_prompts_c100_dt250402-085509.jsonl"
)

SYSTEM_MESSAGE = (
    "You are a world-class creative assistant that generates captivating "
    "and morally-driven fables based on structured inputs.\n"
    "Each fable must be:\n"
    "  - Imaginative and coherent.\n"
    "  - Appropriate for a wide audience, including young readers.\n"
    "  - Structured around a classic fable format "
    "(character, setting, conflict, resolution, and moral).\n\n"
    "Age groups are defined as:\n"
    "  - A: 3 years or under\n"
    "  - B: 4-7 years\n"
    "  - C: 8-11 years\n"
    "  - D: 12-15 years\n"
    "  - E: 16 years or above"
)


class TinyFabulistScenario(Scenario):
    name = "tinyfabulist"
    description = "klusai/tinyfabulist"
    tags = ["creativity", "story_generation", "fable"]

    def __init__(self, source: str = "benchmark"):
        """
        Args:
            source: Which prompt set to use.
                "benchmark" (default): 100 curated prompts from the paper's
                    evaluation (with published baselines for 11 models).
                "full": 100K test split from HuggingFace dataset.
        """
        super().__init__()
        if source not in ("benchmark", "full"):
            raise ValueError(
                f"Invalid source '{source}'. Must be 'benchmark' or 'full'."
            )
        self.source = source

    def _download_repo(self, output_path: str) -> str:
        """Download and extract the tinyfabulist repository."""
        repo_dir = os.path.join(output_path, "tinyfabulist-main")
        if os.path.exists(repo_dir):
            return output_path

        zip_path = os.path.join(output_path, "tinyfabulist.zip")
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)

        return output_path

    def _load_benchmark_prompts(self, output_path: str) -> list[str]:
        """Load the 100 benchmark prompts from the GitHub repo."""
        base = self._download_repo(output_path)
        filepath = os.path.join(base, BENCHMARK_PROMPTS_GLOB)

        prompts = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                items = json.loads(line)
                if isinstance(items, list) and len(items) > 0:
                    item = items[0]
                    if item.get("prompt_type") == "generator_prompt":
                        prompts.append(item["content"])
        return prompts

    def _load_hf_prompts(self) -> list[str]:
        """Load prompts from the HuggingFace test split."""
        from datasets import load_dataset

        ds = load_dataset("klusai/ds-tf1-en-3m", split="test")
        # Extract unique prompts (dedup by prompt_hash)
        seen = set()
        prompts = []
        for item in ds:
            h = item["prompt_hash"]
            if h not in seen:
                seen.add(h)
                prompts.append(item["prompt"])
        return prompts

    def get_instances(self, output_path: str) -> list[Instance]:
        if self.source == "benchmark":
            prompts = self._load_benchmark_prompts(output_path)
        else:
            prompts = self._load_hf_prompts()

        instances = []
        for idx, prompt in enumerate(prompts):
            # Prepend system message context to the generation prompt
            full_prompt = f"{SYSTEM_MESSAGE}\n\n{prompt}"

            instances.append(
                Instance(
                    input=Input(text=full_prompt),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"fable_{idx}",
                )
            )

        return instances
