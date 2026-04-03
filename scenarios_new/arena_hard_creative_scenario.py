"""
HELM Scenario: Arena-Hard v2.0 Creative Writing

Paper: "From Crowdsourced Data to High-Quality Benchmarks:
        Arena-Hard and BenchBuilder Pipeline" (arXiv:2406.11939)
Data:  https://github.com/lmarena/arena-hard-auto
       (data/arena-hard-v2.0/question.jsonl)
Authors: LMArena (lmarena-ai)

Task: Open-ended creative writing generation. Given a diverse creative
writing prompt (poems, lyrics, scripts, dialogues, stories, jokes, and more),
produce a high-quality creative response. Prompts sourced from real user
queries on Chatbot Arena — multilingual, multi-genre, and unrestricted in
form and topic.

Arena-Hard v2.0 (750 total prompts) includes a dedicated creative writing
subset of 250 prompts (category: "creative_writing"). This scenario loads
only that creative writing subset, making it a focused creativity benchmark.

Prompt format: Verbatim user prompts from Chatbot Arena (no system prompt
added; prompts are self-contained creative writing requests).

Dataset: lmarena/arena-hard-auto, data/arena-hard-v2.0/question.jsonl
  filtered to category == "creative_writing" → 250 instances
  Fields: uid, category, subcategory, prompt
  URL: https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/
       data/arena-hard-v2.0/question.jsonl

Fields used:   prompt (verbatim user query), uid (for tracking)
Fields skipped: category, subcategory (used only as filter)
Prompt source: Verbatim from dataset (user-submitted Chatbot Arena prompts)
Evaluation: llm_judge (see annotator_notes.md)
  Original benchmark uses pairwise comparison (GPT-4.1 / Gemini-2.5 as judge).
  This scenario uses single-response quality assessment.

Note: The benchmarks.json URL (facebookresearch/darling) is incorrect.
  Correct data source: github.com/lmarena/arena-hard-auto
"""

import json
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

_DATA_URL = (
    "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
    "data/arena-hard-v2.0/question.jsonl"
)

_CREATIVE_WRITING_CATEGORY = "creative_writing"


class ArenaHardCreativeScenario(Scenario):
    """
    Arena-Hard v2.0 Creative Writing subset — 250 open-ended creative writing
    prompts sourced from real Chatbot Arena user queries.

    Covers diverse creative forms: poetry (multilingual), rap, song lyrics,
    short stories, scripts, dialogues, jokes, and more. No prescribed format
    or length — purely open-ended creative generation.

    Evaluated by LLM-as-judge on overall creative quality. Original benchmark
    uses pairwise win-rate (ELO-style) with GPT-4.1 / Gemini-2.5 judges.
    """

    name = "arena_hard_creative"
    description = "github.com/lmarena/arena-hard-auto (Arena-Hard v2.0 creative_writing subset)"
    tags = ["creativity", "open_ended_generation", "creative_writing", "multilingual"]

    def get_instances(self, output_path: str) -> List[Instance]:
        with urllib.request.urlopen(_DATA_URL) as response:
            lines = response.read().decode("utf-8").splitlines()

        instances = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("category") != _CREATIVE_WRITING_CATEGORY:
                continue

            instances.append(
                Instance(
                    input=Input(text=record["prompt"]),
                    references=[],   # LLM-as-judge; no gold reference
                    split=TEST_SPLIT,
                )
            )

        return instances  # 250 creative writing instances
