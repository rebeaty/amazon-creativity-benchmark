"""
HELM Scenario: Pron vs Prompt — Creative Story Synopsis Generation

Paper: "Pron vs Prompt: Can Large Language Models already Challenge a World-Class
       Fiction Author at Creative Text Writing?"
       arXiv:2407.01119, EMNLP 2024, pages 19654-19670
Code:  https://github.com/grmarco/pron-vs-prompt

Task: Given an imaginary movie/story title, write a creative synopsis of ~600 words
with literary value, appealing to both critics and general audiences.

The paper pitted GPT-4 against award-winning Spanish novelist Patricio Pron across
60 imaginary movie titles. Expert literature critics evaluated outputs using a rubric
grounded in Boden's creativity dimensions (novelty, surprise, value).

Dataset: data/synopses_texts.csv (GitHub raw URL)
         60 titles: 30 by Patricio Pron, 30 AI-generated (GPT-4 Turbo)
         Languages: Spanish titles + English translations; synopses in Spanish/English

Fields used:
  english_title  — English translation of title (language="en", default)
  title          — Original Spanish title (language="es")
  title_origin   — "patricio" (novelist-written) or "machine" (AI-generated)

Fields skipped:
  patricio — human novelist synopsis (model output reference; for calibration only)
  gpt4     — GPT-4 synopsis (model output; for calibration only)
  claude   — Claude synopsis (model output; for calibration only)

Prompt source: Verbatim from Notebook 0 (0_GPT4_prompts.ipynb) in the repo.
  System context + user instruction used for the GPT-4 English condition.
  Note: Spanish condition prompt not released; English prompt used for both by default.

Evaluation: llm_judge (expert rubric: attractiveness, originality, relevance,
            creativity, literary quality; 0-3 scale per dimension)
            See annotator_notes.md for judge configuration and rubric.

Note: Originally flagged "not_suitable" as "research study with no test set for
  new models." The 60 imaginary titles ARE a fixed reusable prompt set for any
  new model — the same pattern as ArenaHard v2.0 creative writing prompts.
  Expert assessments in expert_assessment.csv calibrate the LLM judge.

Parameters:
  language:     "en" (English titles, default) | "es" (Spanish titles)
  title_origin: "all" (default) | "patricio" | "machine"
"""

import csv
import io
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Reference,
    Scenario,
)

_DATA_URL = (
    "https://raw.githubusercontent.com/grmarco/pron-vs-prompt/main/data/synopses_texts.csv"
)

# Verbatim system context from 0_GPT4_prompts.ipynb (English condition)
_SYSTEM_CONTEXT = (
    "We are going to do an experiment in which we are going to compare your creative writing"
    " skills with those of a prestigious novelist, Patricio Pron. The task is to generate"
    " synopses for movie titles that do not exist. The synopses must be creative and appealing"
    " to both critics and the general audience, and must have literary value in and of"
    " themselves.\n\n"
    "Here are some details of the novelist you will be competing with:\n"
    "Patricio Pron (Rosario, December 9, 1975) is a writer and literary critic. Granta magazine"
    " selected him in 2010 as one of the 22 best young writers in Spanish. He won the"
    " twenty-second Alfaguara Novel Prize in 2019 for his work Mañana tendremos otros nombres."
)

# Verbatim user instruction from 0_GPT4_prompts.ipynb
_USER_TEMPLATE = (
    'The proposed title is: "{title}". Please write a synopsis of about 600 words for that'
    " title that meets the above specifications."
)

_VALID_LANGUAGES = ("en", "es")
_VALID_ORIGINS = ("all", "patricio", "machine")


class PronVsPromptScenario(Scenario):
    """
    Pron vs Prompt: creative ~600-word story synopsis from an imaginary movie title.

    60 titles across two origins (Patricio Pron, novelist; or GPT-4 generated).
    Evaluated via LLM judge on attractiveness, originality, relevance, creativity,
    and literary quality (0-3 scale, see annotator_notes.md).

    Expert assessments from 3 literature critics are available in the repo for
    judge calibration (data/expert_assessment.csv).
    """

    name = "pron_vs_prompt"
    description = "github.com/grmarco/pron-vs-prompt (arXiv:2407.01119)"
    tags = ["creativity", "creative_writing", "fiction", "literary_quality", "open_ended_generation"]

    def __init__(self, language: str = "en", title_origin: str = "all"):
        super().__init__()
        if language not in _VALID_LANGUAGES:
            raise ValueError(f"language must be one of {_VALID_LANGUAGES!r}, got {language!r}")
        if title_origin not in _VALID_ORIGINS:
            raise ValueError(
                f"title_origin must be one of {_VALID_ORIGINS!r}, got {title_origin!r}"
            )
        self.language = language
        self.title_origin = title_origin

    def get_instances(self, output_path: str) -> List[Instance]:
        with urllib.request.urlopen(_DATA_URL) as resp:
            content = resp.read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(content))

        instances = []
        for row in reader:
            origin = row.get("title_origin", "").strip()
            if self.title_origin != "all" and origin != self.title_origin:
                continue

            title = (
                row["english_title"].strip()
                if self.language == "en"
                else row["title"].strip()
            )
            if not title:
                continue

            prompt = f"{_SYSTEM_CONTEXT}\n\n{_USER_TEMPLATE.format(title=title)}"

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],  # Open-ended creative generation; LLM-as-judge only
                    split=TEST_SPLIT,
                )
            )

        return instances  # 60 total (all); 30 (patricio); 30 (machine)
