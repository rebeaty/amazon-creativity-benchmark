"""
HELM Scenario: Unfun Corpus

Paper: Getting Serious about Humor: Crafting Humor Datasets with Unfunny Large Language Models
        Zachary Horvitz, Jingru Chen, Rahul Aditya, Harshvardhan Srivastava,
        Robert West, Zhou Yu, Kathleen McKeown
        ACL 2024
        https://arxiv.org/abs/2403.00794

Code: https://github.com/zacharyhorvitz/Getting-Serious-With-LLMs

Dataset: Paired satirical and "unfunned" headlines from The Onion
  - Test: 375 examples
  - Validation: 186 examples
  - Total: 561 evaluation instances

Task: "Unfunning" - Edit satirical headlines to make them realistic/serious.
      This benchmark evaluates humor understanding and manipulation (removing humor)
      rather than humor generation.

Prompt format (from data_generation/prompts/unfun_dataset/few-shot/ and hit_llm_generation_v2.py):
  Chat-style (primary):
    System: "You are a helpful assistant that edits humorous headlines to make them realistic."
    User: {satirical_headline}
    (No explicit "Humorous headline:" or "Realistic headline:" labels)

  Completion-style (alternative):
    "The following humorous headlines can be edited to be realistic:
    {satirical_headline} ->"
    (Uses " ->" separator, not "Realistic version:")

  Note: Paper uses 8-shot prompts with randomly sampled examples. This scenario uses
        zero-shot for simplicity. For few-shot evaluation, sample 8 examples from training data.

Evaluation:
  - Primary: Open-ended generation compared to human-created unfunned headlines
  - Metrics: BLEU, ROUGE (standard), plus optionally:
    - Edit distance (token-level similarity)
    - Humor classifier accuracy (requires trained classifier)
    - Human ratings (realness, funniness, grammaticality, coherence)

Fields used: funny_headline (input), unfunned_headline (reference), url (metadata)

Note: This benchmark tests the ability to understand and remove humor from text,
      which is an asymmetrical task to humor generation. The paper notes that LLMs
      excel at "unfunning" but underperform at generating novel jokes.

Data source: Original Unfun game data from https://github.com/epfl-dlab/unfun
            Processed version in this repo includes no-leakage splits.
"""

import csv
import os
from typing import List, Optional
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
    VALID_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class UnfunCorpusScenario(Scenario):
    """
    Unfun Corpus for evaluating humor understanding and manipulation.

    Models are tasked with editing satirical headlines from The Onion
    to create realistic, serious versions ("unfunning").
    """

    name = "unfun_corpus"
    description = "zacharyhorvitz/Getting-Serious-With-LLMs"  # GitHub repo
    tags = ["creativity", "humor", "text-editing", "language-understanding"]

    def __init__(self, prompt_style: str = "chat"):
        """
        Args:
            prompt_style: Style of prompt to use. Options: ["chat", "completion"]
                         "chat" = "You are a helpful assistant that edits humorous headlines..."
                         "completion" = "The following humorous headlines can be edited..."
        """
        super().__init__()
        if prompt_style not in ["chat", "completion"]:
            raise ValueError(f"Invalid prompt_style: {prompt_style}. Must be 'chat' or 'completion'")
        self.prompt_style = prompt_style

    def download_dataset(self, output_path: str) -> tuple:
        """Download the test and validation datasets."""
        base_url = "https://raw.githubusercontent.com/zacharyhorvitz/Getting-Serious-With-LLMs/main/datasets/unfun/unfun_processed/paired"

        test_url = f"{base_url}/test_unique_pairs_no_leakage.tsv"
        val_url = f"{base_url}/val_unique_pairs_no_leakage.tsv"

        test_path = os.path.join(output_path, "test_unique_pairs_no_leakage.tsv")
        val_path = os.path.join(output_path, "val_unique_pairs_no_leakage.tsv")

        ensure_file_downloaded(source_url=test_url, target_path=test_path)
        ensure_file_downloaded(source_url=val_url, target_path=val_path)

        return test_path, val_path

    def load_dataset(self, file_path: str) -> List[dict]:
        """Load and parse the TSV dataset."""
        examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 4:
                    examples.append({
                        'unfun_id': row[0],
                        'unfunned': row[1],
                        'funny_id': row[2],
                        'funny': row[3],
                        'url': row[4] if len(row) > 4 else None
                    })

        return examples

    def create_prompt(self, satirical_headline: str) -> str:
        """
        Create the prompt based on the selected style.

        Note: This uses zero-shot prompts. The paper uses 8-shot prompts with
        randomly sampled examples from high-quality human edits.

        Chat format: System message + user message (no explicit labels)
        Completion format: Preamble + input with " ->" separator
        """
        if self.prompt_style == "chat":
            # For chat models: System message followed by user message
            # In HELM, we simulate this as a single prompt with instruction + input
            return (
                "You are a helpful assistant that edits humorous headlines to make them realistic.\n\n"
                f"{satirical_headline}"
            )
        else:  # completion
            # For completion models: preamble + input + " ->" separator
            return (
                "The following humorous headlines can be edited to be realistic:\n"
                f"{satirical_headline} ->"
            )

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate instances for the Unfun Corpus.

        Creates instances from both test and validation splits.
        """
        # Download datasets
        test_path, val_path = self.download_dataset(output_path)

        # Load data
        test_examples = self.load_dataset(test_path)
        val_examples = self.load_dataset(val_path)

        instances = []

        # Process test split
        for example in test_examples:
            prompt = self.create_prompt(example['funny'])

            # Reference is the human-created unfunned headline
            references = [
                Reference(Output(text=example['unfunned']), tags=[CORRECT_TAG])
            ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"unfun_test_{example['funny_id']}"
                )
            )

        # Process validation split
        for example in val_examples:
            prompt = self.create_prompt(example['funny'])

            references = [
                Reference(Output(text=example['unfunned']), tags=[CORRECT_TAG])
            ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=VALID_SPLIT,
                    id=f"unfun_val_{example['funny_id']}"
                )
            )

        return instances
