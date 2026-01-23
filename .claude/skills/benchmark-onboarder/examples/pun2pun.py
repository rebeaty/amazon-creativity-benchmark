"""
HELM Scenario: PUN2PUN

Paper: https://aclanthology.org/2025.acl-srw.23/
Code: https://github.com/rexera/Pun2Pun

Task: Cross-lingual pun translation from English to Chinese while preserving or
recreating the pun effect in the target language.

Prompt format: Based on official inference code (infer/translation/openai_infer.py)

Pun types:
  - Homographic: Same word with two meanings (e.g., "parted" = separated/hair parting)
  - Homophonic: Words that sound the same but have different meanings

Fields used: sentence (from graphic.json/phonic.json)
Fields skipped: index (metadata), pun word annotations, explanations (those are for other sub-tasks)

Evaluation: open_ended generation (no ground-truth translation)
  - Official metrics: LLM-as-judge with Hit and Overlap (Ovl) metrics
  - For HELM: Could use open_ended metrics (BLEU, ROUGE) or implement custom annotator

Note: This implements the Translation sub-task only. The benchmark includes 4 tasks
      (classification, locating, explanation, translation), but translation is the
      primary creativity task. English→Chinese direction implemented.
"""

import json
import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Reference, Output,
    TEST_SPLIT
)


class Pun2PunScenario(Scenario):
    name = "pun2pun"
    description = "rexera/Pun2Pun"
    tags = ["creativity", "humor", "pun", "translation", "multilingual"]

    def __init__(self, pun_type: str = "all", direction: str = "en2zh"):
        """
        Args:
            pun_type: One of ['graphic', 'phonic', 'all']
                - graphic: Homographic puns (same spelling, different meanings)
                - phonic: Homophonic puns (same sound, different meanings)
                - all: Both types combined
            direction: Translation direction
                - en2zh: English to Chinese (default)
                - zh2en: Chinese to English
        """
        super().__init__()
        self.pun_type = pun_type
        self.direction = direction

    def _load_puns_from_repo(self, repo_path: str, lang: str, pun_type: str) -> List[dict]:
        """Load puns from local git repo clone."""
        file_path = os.path.join(repo_path, 'data', 'textual', lang, f'{pun_type}.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Pun2Pun dataset not found at {file_path}. "
                f"Please clone the repo: git clone https://github.com/rexera/Pun2Pun.git"
            )

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_pun_definition(self) -> str:
        """Return pun type definitions (translated from Chinese in official code)."""
        return """Homophonic pun: Two words that sound the same or similar but have different meanings, forming a pun.
Homographic pun: The same word that can be understood in two different meanings, or two words that are identical or similar in form but have different meanings, forming a pun."""

    def get_instances(self, output_path):
        # Determine repo path - try common locations
        possible_paths = [
            '/tmp/pun2pun',
            os.path.join(output_path, '..', '..', '..', 'Pun2Pun'),
            './Pun2Pun',
            '../Pun2Pun'
        ]

        repo_path = None
        for path in possible_paths:
            if os.path.exists(path):
                repo_path = path
                break

        if repo_path is None:
            raise FileNotFoundError(
                "Pun2Pun repository not found. Please clone it first:\n"
                "git clone https://github.com/rexera/Pun2Pun.git"
            )

        # Determine source and target languages
        if self.direction == "en2zh":
            source_lang, target_lang = "en", "Chinese"
            lang_code = "en"
        elif self.direction == "zh2en":
            source_lang, target_lang = "zh", "English"
            lang_code = "zh"
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        # Load puns based on type
        puns = []
        if self.pun_type in ["graphic", "all"]:
            puns.extend(self._load_puns_from_repo(repo_path, lang_code, "graphic"))
        if self.pun_type in ["phonic", "all"]:
            puns.extend(self._load_puns_from_repo(repo_path, lang_code, "phonic"))

        # Create instances
        instances = []
        for item in puns:
            # Build prompt following official format
            definition = self._get_pun_definition()

            prompt = f"""{definition}

Your task:
Translate this pun into {target_lang} while preserving the original pun effect or creating a new pun in the target language.

Original pun:
{item['sentence']}

Please output only the translated sentence."""

            # For open-ended generation, reference is typically empty
            # or we could include the source as context
            references = [Reference(
                Output(text=""),  # No ground-truth translation available
                tags=[]
            )]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
