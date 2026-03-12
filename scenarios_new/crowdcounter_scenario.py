"""
HELM Scenario: CROWDCOUNTER - Type-Specific Counterspeech Generation

Paper: CrowdCounter: A benchmark type-specific multi-target counterspeech dataset
       Punyajoy Saha, Abhilash Datta, Abhik Jana, Animesh Mukherjee
       CoNLL 2024
       https://arxiv.org/abs/2410.01400

Code: https://github.com/hate-alert/CrowdCounter

Dataset: 3,425 hate speech-counterspeech pairs across 6 counterspeech types
  - Train: 2,047 examples
  - Val: 100 examples
  - Test: 1,288 examples

Task: Generate strategic counterspeech responses to hate speech that foster understanding
      or discourage harmful behavior. Models can generate either vanilla (unconditioned)
      or type-specific counterspeech.

Six counterspeech types:
  1. contradiction: Point out inconsistencies or contradictions
  2. empathy_affiliation: Respond with friendly, empathetic, peaceful tone
  3. humour: Use humor, caricature, or sarcasm
  4. questions: Pose thoughtful questions to encourage reflection
  5. shaming: Leverage societal stigma to prompt reconsideration
  6. warning-of-consequences: Emphasize potential harm caused by hate speech

Prompt format (from repository prompts.py):
  Vanilla: "Counterspeech is a strategic response to hate speech, aiming to foster
           understanding or discourage harmful behavior. A good counterspeech to this
           hate speech - \"{hate_speech}\" is:"

  Type-specific: Includes definitions of all 6 types, then:
                "A \"{type}\" type good counterspeech to this hate speech -
                 {hate_speech} is:"

Evaluation:
  - Primary: Open-ended generation evaluated with BLEU, ROUGE against ground truth
  - Paper metrics: Relevance, diversity (Self-BLEU, Distinct-n), quality (perplexity)
  - Optional: Type classification accuracy (does generated counterspeech match target type?)

Fields used: hatespeech (input), counterspeech (reference), required_types (type label)
Fields available: total_types (all applicable types including additional ones)

Note: Data in GitHub repository, not HuggingFace. Files are in JSONL format.
"""

import json
import os
from typing import List, Optional
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
    VALID_SPLIT,
    TRAIN_SPLIT,
)
from helm.common.general import ensure_directory_exists


class CrowdCounterScenario(Scenario):
    """
    CROWDCOUNTER: Type-specific counterspeech generation benchmark.

    Models generate strategic responses to hate speech across 6 counterspeech types.
    """

    name = "crowdcounter"
    description = "hate-alert/CrowdCounter"  # GitHub repo
    tags = ["creativity", "counterspeech", "hate-speech", "text-generation", "open-ended"]

    # Counterspeech type definitions from prompts.py
    TYPE_DEFINITIONS = {
        "contradiction": "Point out any inconsistencies or contradictions in the hate speech. Explain and rationalize past behavior or encourage the individual to reflect on their statements.",
        "empathy_affiliation": "Respond with a friendly, empathetic, or peaceful tone to counteract hostility or violence in the original message.",
        "humour": "Utilize humor, ranging from conciliatory to provocative, through forms like caricature and sarcasm to address the hate speech.",
        "questions": "Pose thoughtful questions to challenge the hate speaker's sources of information or encourage self-reflection.",
        "shaming": "Leverage societal stigma associated with offensive terms to prompt speakers to reconsider their statements. Provide insight into the reasons behind the hateful nature of the speech.",
        "warning-of-consequences": "Emphasize the potential harm caused by hate speech, highlighting its ability to incite real-world actions. Stress the visibility of online content to the speaker's offline network."
    }

    # Normalize type names (dataset uses underscores and hyphens)
    TYPE_NORMALIZATION = {
        "empathy_affiliation": "empathy affiliation",
        "warning-of-consequences": "warning of consequences"
    }

    def __init__(self, prompt_type: str = "vanilla", use_types: bool = True):
        """
        Args:
            prompt_type: Prompt style to use. Options: ["vanilla", "type_specific"]
                        "vanilla" = Generate any counterspeech
                        "type_specific" = Generate counterspeech of specific type
            use_types: Whether to include type information in prompts (only for type_specific)
        """
        super().__init__()
        if prompt_type not in ["vanilla", "type_specific"]:
            raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 'vanilla' or 'type_specific'")

        self.prompt_type = prompt_type
        self.use_types = use_types

    def download_dataset(self, output_path: str) -> str:
        """Clone the CrowdCounter repository to get the dataset."""
        import subprocess

        repo_path = os.path.join(output_path, "CrowdCounter")

        if not os.path.exists(repo_path):
            print(f"Cloning CrowdCounter repository to {repo_path}...")
            subprocess.run(
                ["git", "clone", "https://github.com/hate-alert/CrowdCounter", repo_path],
                check=True
            )

        return os.path.join(repo_path, "Datasets", "CrowdCounter")

    def load_dataset(self, file_path: str) -> List[dict]:
        """Load counterspeech data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def create_vanilla_prompt(self, hate_speech: str) -> str:
        """Create vanilla (unconditioned) counterspeech prompt."""
        return (
            "Counterspeech is a strategic response to hate speech, aiming to foster "
            "understanding or discourage harmful behavior. "
            f"A good counterspeech to this hate speech - \"{hate_speech}\" is:"
        )

    def create_type_specific_prompt(self, hate_speech: str, cs_type: str) -> str:
        """Create type-specific counterspeech prompt with type definitions."""
        # Build type definitions section
        type_defs = "Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior. Different types of counterspeech include:\n\n"

        for i, (type_name, definition) in enumerate(self.TYPE_DEFINITIONS.items(), 1):
            display_name = self.TYPE_NORMALIZATION.get(type_name, type_name)
            type_defs += f"{i}. {display_name}: {definition}\n\n"

        # Normalize type for display
        display_type = self.TYPE_NORMALIZATION.get(cs_type, cs_type)

        return (
            f"{type_defs}"
            f"A \"{display_type}\" type good counterspeech to this hate speech - {hate_speech} is:"
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate instances for counterspeech generation.

        Creates instances from train/val/test splits based on prompt type.
        """
        # Download dataset
        dataset_path = self.download_dataset(output_path)

        instances = []

        # Load all splits
        splits = {
            TRAIN_SPLIT: os.path.join(dataset_path, "Train.json"),
            VALID_SPLIT: os.path.join(dataset_path, "Val.json"),
            TEST_SPLIT: os.path.join(dataset_path, "Test.json")
        }

        for split_name, file_path in splits.items():
            data = self.load_dataset(file_path)

            for idx, item in enumerate(data):
                hate_speech = item['hatespeech']
                counterspeech = item['counterspeech']
                cs_type = item['required_types']

                # Create prompt based on type
                if self.prompt_type == "vanilla":
                    prompt_text = self.create_vanilla_prompt(hate_speech)
                else:  # type_specific
                    prompt_text = self.create_type_specific_prompt(hate_speech, cs_type)

                # Reference is the ground truth counterspeech
                references = [
                    Reference(Output(text=counterspeech), tags=[cs_type])
                ]

                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=references,
                        split=split_name,
                        id=f"crowdcounter_{split_name}_{idx}"
                    )
                )

        return instances
