"""
HELM Scenario: Open-domain Dialogue Generation for Diversity

Paper: Improving Linguistic Diversity of Large Language Models with Possibility Exploration Fine-Tuning
       https://arxiv.org/abs/2412.03343
Code: https://github.com/mailong25/peft_diversity

Task: Generate multiple diverse dialogue responses to the same conversation context.
      Given a dialogue history, models must produce semantically distinct responses
      that maintain coherence and relevance.

Dataset: 299 test examples from dialogue conversations
- Each example contains:
  * Context: Multi-turn conversation history
  * Candidates: 10 reference diverse responses

Prompt format: For each possibility number k (typically 1-5 or 1-10):
  [INST] Given this conversation:

  Person A: [utterance]
  Person B: [utterance]
  ...

  Imagine you are person B and act as if you were a real individual. Think about all
  the possibilities in which person B might respond next and then provide the response
  that corresponds to possibility number #[k]. Keep the response short with no more
  than 25 words. [/INST] Person B:

Evaluation: open_ended + llm_judge + custom diversity metrics
  - Semantic Diversity: Cosine similarity between response embeddings (lower = better)
  - N-gram Diversity: Distinct unigrams and bigrams across responses
  - Coherence: LLM judge (Llama/GPT-4) rates 1-10 scale
  - Incoherence Rate: % of responses with coherence ≤ 5
  See annotator_notes.md for complete evaluation setup

Note: Models should generate multiple different responses (typically 5-10) for the same
      context by varying the possibility number parameter.
"""

import json
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output, Reference,
    TEST_SPLIT,
)


class DialogueDiversityScenario(Scenario):
    """Open-domain Dialogue Generation for Diversity

    Evaluates models' ability to generate multiple semantically diverse yet coherent
    responses to the same dialogue context.
    """

    name = "dialogue_diversity"
    description = "mailong25/peft_diversity"
    tags = ["creativity", "dialogue", "diversity", "open_ended"]

    # GitHub raw URL for test data
    BASE_URL = "https://raw.githubusercontent.com/mailong25/peft_diversity/main/data"

    def __init__(self, num_responses: int = 5):
        """
        Args:
            num_responses: Number of diverse responses to generate per context (default: 5)
        """
        super().__init__()
        if num_responses < 1:
            raise ValueError(f"num_responses must be >= 1, got {num_responses}")
        self.num_responses = num_responses

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download test data
        data_url = f"{self.BASE_URL}/test_dialogs.jsonl"

        instances = []

        with urllib.request.urlopen(data_url) as response:
            for line in response:
                item = json.loads(line.decode('utf-8'))
                # Create one instance per possibility number
                for k in range(1, self.num_responses + 1):
                    instances.append(self._create_instance(item, k))

        return instances

    def _create_instance(self, item: dict, possibility_num: int) -> Instance:
        """Create an instance for a specific possibility number"""

        context = item["context"]
        candidates = item.get("candidates", [])

        # Format context with Person A/Person B labels
        # Context is newline-separated, alternating A and B (B is last)
        context_lines = context.split('\n')
        formatted_lines = []
        for i, line in enumerate(reversed(context_lines)):
            if i % 2 == 0:
                formatted_lines.append(f"Person A: {line}")
            else:
                formatted_lines.append(f"Person B: {line}")
        formatted_context = '\n'.join(reversed(formatted_lines))

        # Build prompt following the EXACT format from prompt_generator.py (lines 12-26)
        # Using the format for option = int (possibility number specified)
        prompt = f"""[INST] Given this conversation:

{formatted_context}

Imagine you are person B and act as if you were a real individual. Think about all the possibilities in which person B might respond next and then provide the response that corresponds to possibility number #{possibility_num}. Keep the response short with no more than 25 words. [/INST] Person B:"""

        # Create references from candidate responses
        # Note: The candidates don't have explicit possibility numbers, so we treat them
        # as examples of diverse valid responses
        references = []
        for candidate in candidates:
            references.append(Reference(
                output=Output(text=candidate),
                tags=["diverse_response"]
            ))

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=TEST_SPLIT,
        )
