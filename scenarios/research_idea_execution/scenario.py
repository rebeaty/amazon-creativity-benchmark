"""
HELM Scenario: Research Idea Peer Review

Paper: "The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus
        Human Research Ideas" (arXiv:2506.20803)
Code:  https://github.com/NoviScl/AI-Researcher
OSF:   https://osf.io/ckxtp

Task: Given a structured NLP research idea, generate an expert peer review
assessing it on five dimensions: novelty, excitement, feasibility, expected
effectiveness, and overall quality. This operationalises the paper's execution
study evaluation rubric (Appendix A, 1–10 scale) as a text-generation task,
with LLM-as-judge replacing the 58-expert human panel.

The task tests a model's analytical creativity — the ability to reason deeply
about a research proposal, identify its strengths and gaps, and produce a
coherent, calibrated scientific critique. This is directly analogous to the
human expert evaluation in the paper.

Data sources (two tiers, loaded in order):
  Tier 1 — Google Drive zip (primary):
    Human-authored research ideas from the ideation study.
    File ID: 1Z2Nd7WNNks-eCoqUgPzx1_ovYqU8OiPx
    File:    Ideation_Study_Human_Ideas.zip (~137 KB)
    Ideas follow 5-field structure: Problem, Existing Methods, Motivation,
    Proposed Method, Experiment Plan.

  Tier 2 — GitHub fallback (always loaded):
    ai_researcher/prompts/idea_examples_method.json
    10 demonstration ideas (known-good format, same 5-field structure).
    Used as guaranteed instances regardless of zip parse success.

Prompt format (adapted from paper Appendix A rubric — no LLM prompt specified
in the paper; this scenario uses the evaluation rubric as the prompt basis):

  You are an expert NLP researcher reviewing a research proposal. Evaluate the
  following research idea and write a peer review covering five dimensions
  (1 = very poor, 10 = excellent): Novelty, Excitement, Feasibility, Expected
  Effectiveness, and Overall. For each dimension provide a score and rationale.

  [Structured idea text with all 5 fields]

Fields used:   Title (if present), Problem, Existing Methods, Motivation,
               Proposed Method, Experiment Plan
Fields skipped: id_title_mapping IDs, reviewer metadata (familiarity, experience)

Expert reviews: data_points_all_anonymized.json (398 records, novelty_score +
  novelty_rationale) are used for judge calibration in annotator_notes.md.
  NOTE: reviews are not linked 1-to-1 to ideas in the JSON; they are used as
  quality exemplars for the LLM judge, not as per-instance gold references.

Evaluation: llm_judge (see annotator_notes.md)
"""

import json
import os
import subprocess
import urllib.request
import zipfile
from typing import List, Optional

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

_HUMAN_IDEAS_FILE_ID = "1Z2Nd7WNNks-eCoqUgPzx1_ovYqU8OiPx"
_ZIP_NAME = "Ideation_Study_Human_Ideas.zip"
_FALLBACK_URL = (
    "https://raw.githubusercontent.com/NoviScl/AI-Researcher/main"
    "/ai_researcher/prompts/idea_examples_method.json"
)

# Known field names for the structured idea format (with variations)
_FIELD_ALIASES = {
    "problem": ["Problem", "problem", "Problem Statement", "problem_statement"],
    "existing_methods": [
        "Existing Methods", "existing_methods", "Existing Work",
        "Related Work", "Background",
    ],
    "motivation": ["Motivation", "motivation"],
    "proposed_method": [
        "Proposed Method", "proposed_method", "Method", "Approach",
        "Proposed Approach",
    ],
    "experiment_plan": [
        "Experiment Plan", "experiment_plan", "Experiments",
        "Experimental Plan", "Evaluation",
    ],
}

_PROMPT_TEMPLATE = """\
You are an expert NLP researcher reviewing a research proposal. Evaluate the \
following research idea and write a detailed peer review.

{idea_text}

Assess this idea on each of the following dimensions (1 = very poor, \
10 = excellent) and provide a brief rationale for each score:

1. Novelty (1-10): How original is this idea compared to existing work?
2. Excitement (1-10): How impactful and exciting would this research be?
3. Feasibility (1-10): Can this be realistically implemented and evaluated?
4. Expected Effectiveness (1-10): How likely is this approach to succeed?
5. Overall (1-10): Comprehensive assessment of the proposal's quality.

Write your review in this format:
Novelty: [score]
Novelty Rationale: [1-2 sentence explanation]

Excitement: [score]
Excitement Rationale: [1-2 sentence explanation]

Feasibility: [score]
Feasibility Rationale: [1-2 sentence explanation]

Expected Effectiveness: [score]
Expected Effectiveness Rationale: [1-2 sentence explanation]

Overall: [score]
Overall Rationale: [2-3 sentence summary of strengths and weaknesses]"""


def _get_field(idea: dict, canonical: str) -> Optional[str]:
    """Retrieve a field from an idea dict using known alias variants."""
    for alias in _FIELD_ALIASES.get(canonical, [canonical]):
        if alias in idea and idea[alias]:
            return str(idea[alias]).strip()
    return None


def _format_idea(idea: dict, title: Optional[str] = None) -> str:
    """Render a structured idea dict as readable text for the prompt."""
    lines = []
    if title:
        lines.append(f"**Title:** {title}\n")
    elif "Title" in idea and idea["Title"]:
        lines.append(f"**Title:** {idea['Title']}\n")

    for label, canonical in [
        ("Problem", "problem"),
        ("Existing Methods", "existing_methods"),
        ("Motivation", "motivation"),
        ("Proposed Method", "proposed_method"),
        ("Experiment Plan", "experiment_plan"),
    ]:
        val = _get_field(idea, canonical)
        if val:
            lines.append(f"**{label}:**\n{val}")

    return "\n\n".join(lines)


def _load_fallback_ideas(fallback_path: str) -> List[dict]:
    """Load demonstration ideas from the GitHub JSON file."""
    if not os.path.exists(fallback_path):
        urllib.request.urlretrieve(_FALLBACK_URL, fallback_path)
    with open(fallback_path, encoding="utf-8") as f:
        data = json.load(f)
    # File is a list of idea dicts or a dict of idea dicts
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Could be {"idea_name": {fields...}, ...}
        items = list(data.values())
        # If values are dicts with idea fields, return them
        if items and isinstance(items[0], dict):
            return items
    return []


def _load_zip_ideas(zip_path: str, extract_dir: str) -> List[dict]:
    """Extract and parse JSON idea files from the human ideas zip."""
    ideas = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            json_files = [n for n in zf.namelist() if n.endswith(".json")]
            zf.extractall(extract_dir)

        for rel_path in json_files:
            full_path = os.path.join(extract_dir, rel_path)
            try:
                with open(full_path, encoding="utf-8") as f:
                    idea = json.load(f)
                # Keep only if it has at least one recognisable idea field
                if any(
                    alias in idea
                    for aliases in _FIELD_ALIASES.values()
                    for alias in aliases
                ):
                    ideas.append(idea)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    except (zipfile.BadZipFile, FileNotFoundError):
        pass
    return ideas


class ResearchIdeaExecutionScenario(Scenario):
    """
    Research Idea Peer Review — LLM generates expert-style critique.

    Each instance is a structured NLP research idea (problem statement,
    motivation, proposed method, experiment plan). The model writes a
    peer review on five dimensions matching the paper's evaluation rubric.
    Evaluated by LLM-as-judge comparing to the 398 expert reviews collected
    in the ideation study.
    """

    name = "research_idea_execution"
    description = "NoviScl/AI-Researcher (arXiv:2506.20803)"
    tags = ["creativity", "scientific_reasoning", "peer_review", "research_ideation"]

    def _download_zip(self, zip_path: str) -> None:
        """Download the human ideas zip from Google Drive via gdown."""
        subprocess.run(
            ["pip", "install", "-q", "gdown"],
            check=True, capture_output=True,
        )
        subprocess.run(
            [
                "gdown",
                f"https://drive.google.com/uc?id={_HUMAN_IDEAS_FILE_ID}",
                "-O", zip_path,
            ],
            check=True, capture_output=True,
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        os.makedirs(output_path, exist_ok=True)

        zip_path = os.path.join(output_path, _ZIP_NAME)
        extract_dir = os.path.join(output_path, "human_ideas")
        fallback_path = os.path.join(output_path, "idea_examples_method.json")

        # --- Tier 1: Download and parse human ideas zip ---
        zip_ideas: List[dict] = []
        if not os.path.exists(zip_path):
            try:
                self._download_zip(zip_path)
            except Exception:
                pass  # Fall through to tier-2 fallback

        if os.path.exists(zip_path):
            zip_ideas = _load_zip_ideas(zip_path, extract_dir)

        # --- Tier 2: GitHub demonstration examples (always loaded) ---
        fallback_ideas = _load_fallback_ideas(fallback_path)

        # Combine: zip ideas first, then fallback (deduplicated by title if possible)
        all_ideas = zip_ideas + fallback_ideas
        if not all_ideas:
            raise RuntimeError(
                "No research ideas could be loaded from Google Drive zip "
                f"({_ZIP_NAME}) or GitHub fallback. Check network access."
            )

        instances = []
        seen_titles: set = set()

        for idea in all_ideas:
            idea_text = _format_idea(idea)
            if not idea_text.strip():
                continue

            # Deduplicate by title
            title = idea.get("Title", idea.get("title", ""))
            if title and title in seen_titles:
                continue
            if title:
                seen_titles.add(title)

            prompt = _PROMPT_TEMPLATE.format(idea_text=idea_text)

            # No per-instance gold reference — evaluation is LLM-as-judge
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                )
            )

        return instances
