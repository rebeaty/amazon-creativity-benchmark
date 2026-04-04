"""
HELM Scenario: Metaphor Generation from Literal Sentences

Paper: MERMAID: Metaphor Generation with Symbolism and Discriminative Decoding (NAACL 2021)
       https://aclanthology.org/2021.naacl-main.336/
       https://arxiv.org/abs/2103.06779
Code: https://github.com/tuhinjubcse/MetaphorGenNAACL2021

Task: Generate metaphorical sentences from literal sentences by replacing verbs
      with more creative, symbolic alternatives while maintaining semantic coherence.

Dataset: 156 human-curated test examples
- Input: Literal sentence (e.g., "I bet you have not done your homework")
- Output: Metaphorical sentence (e.g., "I fancy you have not done your homework")

The metaphorical sentences mark replaced words with <V> tags in the reference data,
but these tags are stripped for evaluation purposes.

Prompt format: Direct input (no special instruction prefix specified in paper)
  Input: {literal_sentence}

Evaluation: open_ended (BLEU, ROUGE, F1)
  Paper uses BLEU scoring (score.py) and human evaluation
  Human evaluation: Model metaphors preferred over 3 baselines 66% of the time

Fields used: human1test.txt (literal), human2test.txt (metaphorical)
Fields skipped: Training data (automatically constructed from Gutenberg Poetry, lower quality)
Note: Test set is human-curated and higher quality than training data
"""

import re
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output, Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)


class MetaphorGenerationScenario(Scenario):
    """Metaphor Generation Scenario

    Evaluates models' ability to transform literal language into metaphorical
    expressions by creatively replacing verbs while maintaining meaning.
    """

    name = "metaphor_generation"
    description = "tuhinjubcse/MetaphorGenNAACL2021"
    tags = ["creativity", "metaphor", "figurative_language"]

    # Raw GitHub URLs for test data
    LITERAL_URL = "https://raw.githubusercontent.com/tuhinjubcse/MetaphorGenNAACL2021/main/fairseq/human1test.txt"
    METAPHORICAL_URL = "https://raw.githubusercontent.com/tuhinjubcse/MetaphorGenNAACL2021/main/fairseq/human2test.txt"

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download literal sentences (inputs)
        with urllib.request.urlopen(self.LITERAL_URL) as response:
            literal_sentences = [line.strip() for line in response.read().decode("utf-8").splitlines()]

        # Download metaphorical sentences (references)
        with urllib.request.urlopen(self.METAPHORICAL_URL) as response:
            metaphorical_sentences = [
                line.strip() for line in response.read().decode("utf-8").splitlines()
            ]

        assert len(literal_sentences) == len(
            metaphorical_sentences
        ), f"Mismatch: {len(literal_sentences)} literal vs {len(metaphorical_sentences)} metaphorical"

        instances = []
        for literal, metaphorical in zip(literal_sentences, metaphorical_sentences):
            if literal and metaphorical:  # Skip empty lines
                instances.append(self._create_instance(literal, metaphorical))

        return instances

    def _create_instance(self, literal_sentence: str, metaphorical_sentence: str) -> Instance:
        """Create an instance from a literal-metaphorical sentence pair"""

        # Strip <V> tags from the metaphorical reference
        # The <V> tags mark which words were replaced, but aren't part of the actual text
        metaphorical_clean = re.sub(r"<V>\s*|\s*<V>", "", metaphorical_sentence)

        # Create prompt - paper uses direct input without special formatting
        prompt = literal_sentence

        # Create reference with the ground truth metaphorical sentence
        references = [Reference(output=Output(text=metaphorical_clean), tags=[CORRECT_TAG])]

        return Instance(input=Input(text=prompt), references=references, split=TEST_SPLIT)
