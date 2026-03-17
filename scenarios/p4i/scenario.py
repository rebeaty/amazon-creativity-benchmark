"""
HELM Scenario: P4I (PoemForImage) - Summary-to-Poem Generation

Paper: "PoemTale Diffusion: Minimising Information Loss in Poem to Image
        Generation with Multi-Stage Prompt Refinement"
       https://arxiv.org/abs/2507.13708
Code: https://github.com/SofeeyaJ/PoemTale-Diffusion

Original task direction: poem -> image (diffusion model generation).
The repo's generated images are not published, so the multimodal reverse
task (image -> poem) is not feasible.

Adapted as text-only creative writing: Given a prose summary of a poem's
content, themes, and emotions, the model must compose an original poem.
The original poem text serves as the reference. This tests the model's
ability to transform analytical prose into creative verse — capturing mood,
imagery, and poetic structure.

Dataset: Segmented_Poems.csv from the PoemTale-Diffusion GitHub repo.
3,008 poems sourced from online and offline resources, each annotated with:
  - text: Prose description/analysis of the poem
  - ctext: The actual poem text (used as reference)
  - our_summary: Shorter summary of the poem's themes
  - Emotions: Per-segment emotion labels
  - Segments: Poem split into semantic segments
  - NER Tags: Named entity tags

Prompt format (not from paper; paper targets diffusion models):
  Uses our_summary as the creative brief + emotion labels for tone guidance.

Fields used: our_summary (input context), Emotions (input tone hint),
             ctext (reference poem)
Fields skipped: text (too detailed — would make task trivial), Segments,
                NER Tags, Poem Link, Title, Poet

Evaluation: open_ended (BLEU, ROUGE) + LLM-as-judge for poetic quality.
See annotator_notes.md for judge configuration.
"""

import csv
import os
import subprocess
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)


REPO_URL = "https://github.com/SofeeyaJ/PoemTale-Diffusion.git"

_PROMPT_TEMPLATE = (
    "Write a poem based on the following description. Your poem should use "
    "vivid imagery, poetic devices (metaphor, rhythm, alliteration, etc.), "
    "and capture the emotional tone indicated.\n\n"
    "Description: {summary}\n"
    "Emotional tone: {emotions}\n\n"
    "Poem:"
)


class P4IScenario(Scenario):
    """
    P4I: Summary-to-poem creative writing benchmark.

    Given a prose summary and emotional tone, generate an original poem.
    3,008 instances evaluated against reference poems.
    """

    name = "p4i"
    description = "SofeeyaJ/PoemTale-Diffusion"
    tags = ["creativity", "poetry", "generation"]

    def __init__(self, max_instances: int = 0):
        """
        Args:
            max_instances: Cap on number of instances. 0 = all (default).
        """
        super().__init__()
        self.max_instances = max_instances

    def _clone_repo(self, output_path: str) -> str:
        """Clone the PoemTale-Diffusion repo if not present."""
        repo_dir = os.path.join(output_path, "PoemTale-Diffusion")
        if os.path.isdir(repo_dir):
            return repo_dir

        os.makedirs(output_path, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, repo_dir],
            check=True,
            capture_output=True,
        )
        return repo_dir

    @staticmethod
    def _clean_emotions(raw: str) -> str:
        """Parse emotion list string and return comma-separated unique labels."""
        # raw is like "['joy', 'neutral', 'sadness']"
        try:
            import ast
            emotions = ast.literal_eval(raw)
            if isinstance(emotions, list):
                # Deduplicate while preserving order
                seen = set()
                unique = []
                for e in emotions:
                    e = str(e).strip()
                    if e and e not in seen:
                        seen.add(e)
                        unique.append(e)
                return ", ".join(unique)
        except (ValueError, SyntaxError):
            pass
        return raw.strip("[]' ")

    def get_instances(self, output_path: str) -> List[Instance]:
        repo_dir = self._clone_repo(output_path)
        csv_path = os.path.join(repo_dir, "Dataset", "Segmented_Poems.csv")

        instances: List[Instance] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary = row.get("our_summary", "").strip()
                poem_text = row.get("ctext", "").strip()
                emotions_raw = row.get("Emotions", "").strip()

                # Skip rows missing essential fields
                if not summary or not poem_text:
                    continue

                emotions = self._clean_emotions(emotions_raw)

                prompt = _PROMPT_TEMPLATE.format(
                    summary=summary,
                    emotions=emotions if emotions else "unspecified",
                )

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=[
                            Reference(
                                Output(text=poem_text), tags=[CORRECT_TAG]
                            )
                        ],
                        split=TEST_SPLIT,
                    )
                )

                if self.max_instances and len(instances) >= self.max_instances:
                    break

        return instances
