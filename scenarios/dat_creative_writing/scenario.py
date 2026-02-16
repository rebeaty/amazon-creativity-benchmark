"""
HELM Scenario: Divergent Creativity (DAT + Creative Writing)

Paper: "Divergent Creativity in Humans and Large Language Models"
       https://doi.org/10.1038/s41562-024-02046-9 (Nature Human Behaviour, 2024)
Code: https://github.com/AntoineBellemare/DAT_GPT

Two divergent creativity tasks:

  - dat: Divergent Association Task (Olson et al. 2021). Generate 10
    semantically distant words. Scored via GLoVe cosine distance.
  - dat_strategies: DAT with 5 cognitive strategies (thesaurus, etymology,
    opposites, randomness, child perspective). Tests strategy-guided creativity.
  - synopsis: Write a creative movie synopsis. Scored via DSI + Lziv.
  - flash_fiction: Write creative flash fiction in exactly 100 words.
  - haiku: Write a creative haiku (5-7-5 syllable pattern).

Each subset generates N instances with the same prompt (default 100).
With non-zero temperature, each instance produces different output,
enabling measurement of creative output diversity.

Prompt source: Exact prompts from scripts/api_call_dat_gpt4.py and
  scripts/api_call_gpt4_stories.py in the repo.

Fields used: None (prompts are fixed, not from a dataset)

Evaluation: Custom metrics. See metric_notes.md.
  - DAT: GLoVe 840B cosine distance (average pairwise distance × 100)
  - Writing: DSI (BERT-based semantic diversity), Lziv (Lempel-Ziv
    complexity), LLM-as-judge for quality
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

DAT_PROMPT = (
    "Please enter 10 words that are as different from each other as "
    "possible, in all meanings and uses of the words. Rules: Only single "
    "words in English. Only nouns (e.g., things, objects, concepts). No "
    "proper nouns (e.g., no specific people or places). No specialised "
    "vocabulary (e.g., no technical terms). Think of the words on your own "
    "(e.g., do not just look at objects in your surroundings). Make a list "
    "of these 10 words, a single word in each entry of the list."
)

DAT_STRATEGY_PROMPTS = {
    "thesaurus": (
        "Please enter 10 words that are as different from each other as "
        "possible, in all meanings and uses of the words, using a strategy "
        "that relies on using a thesaurus. Rules: Only single words in "
        "English. Only nouns (e.g., things, objects, concepts). No proper "
        "nouns (e.g., no specific people or places). No specialised "
        "vocabulary (e.g., no technical terms). Think of the words on your "
        "own (e.g., do not just look at objects in your surroundings). Make "
        "a list of these 10 words, a single word in each entry of the list."
    ),
    "etymology": (
        "Please enter 10 words that are as different from each other as "
        "possible, in all meanings and uses of the words, using a strategy "
        "that relies on varying etymology. Rules: Only single words in "
        "English. Only nouns (e.g., things, objects, concepts). No proper "
        "nouns (e.g., no specific people or places). No specialised "
        "vocabulary (e.g., no technical terms). Think of the words on your "
        "own (e.g., do not just look at objects in your surroundings). Make "
        "a list of these 10 words, a single word in each entry of the list."
    ),
    "opposites": (
        "Please enter 10 words that are as different from each other as "
        "possible, in all meanings and uses of the words, using a strategy "
        "that relies on meaning opposition. Rules: Only single words in "
        "English. Only nouns (e.g., things, objects, concepts). No proper "
        "nouns (e.g., no specific people or places). No specialised "
        "vocabulary (e.g., no technical terms). Think of the words on your "
        "own (e.g., do not just look at objects in your surroundings). Make "
        "a list of these 10 words, a single word in each entry of the list."
    ),
    "random": (
        "Please enter 10 words that are as different from each other as "
        "possible, in all meanings and uses of the words, using a strategy "
        "that relies on randomness. Rules: Only single words in English. "
        "Only nouns (e.g., things, objects, concepts). No proper nouns "
        "(e.g., no specific people or places). No specialised vocabulary "
        "(e.g., no technical terms). Think of the words on your own (e.g., "
        "do not just look at objects in your surroundings). Make a list of "
        "these 10 words, a single word in each entry of the list."
    ),
    "young": (
        "Please enter 10 words that are as different from each other as "
        "possible, in all meanings and uses of the words. Answer from the "
        "perspective of a child. Rules: Only single words in English. Only "
        "nouns (e.g., things, objects, concepts). No proper nouns (e.g., "
        "no specific people or places). No specialised vocabulary (e.g., "
        "no technical terms). Think of the words on your own (e.g., do not "
        "just look at objects in your surroundings). Make a list of these "
        "10 words, a single word in each entry of the list."
    ),
}

WRITING_PROMPTS = {
    "synopsis": "Write a creative and unique movie synopsis.",
    "flash_fiction": (
        "Write a creative flash fiction story in exactly 100 words."
    ),
    "haiku": "Write a creative haiku following the 5-7-5 syllable pattern.",
}


class DATCreativeWritingScenario(Scenario):
    name = "dat_creative_writing"
    description = "AntoineBellemare/DAT_GPT"
    tags = ["creativity", "divergent_thinking", "writing"]

    SUBSETS = [
        "dat",
        "dat_thesaurus",
        "dat_etymology",
        "dat_opposites",
        "dat_random",
        "dat_young",
        "synopsis",
        "flash_fiction",
        "haiku",
    ]

    def __init__(self, subset: str = "dat", num_instances: int = 100):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(
                f"Unknown subset '{subset}'. Choose from: {self.SUBSETS}"
            )
        self.subset = subset
        self.num_instances = num_instances

    def get_instances(self, output_path: str) -> List[Instance]:
        # Determine prompt
        if self.subset == "dat":
            prompt_text = DAT_PROMPT
        elif self.subset.startswith("dat_"):
            strategy = self.subset[4:]  # e.g. "thesaurus"
            prompt_text = DAT_STRATEGY_PROMPTS[strategy]
        else:
            prompt_text = WRITING_PROMPTS[self.subset]

        instances = []
        for i in range(self.num_instances):
            instances.append(
                Instance(
                    input=Input(text=prompt_text),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"{self.subset}_{i}",
                )
            )

        return instances
