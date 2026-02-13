"""
HELM Scenario: Cue-word-based Creative Story Generation

Paper: Evaluating Creative Short Story Generation in Humans and Large Language Models
       https://arxiv.org/abs/2411.02316
       ICCC 2025

Dataset: https://huggingface.co/datasets/mismayil/creative_story_generation_dataset
License: Not specified (academic research)
GitHub: https://github.com/mismayil/creative-story-gen

Task: Creative Story Generation with Cue Words
Generate a creative short story (maximum 5 sentences) that includes three given cue words.

Cue word sets (from paper):
- Low semantic distance: "stamp, letter, send" and "petrol, diesel, pump"
- High semantic distance: "gloom, payment, exist" and "organ, empire, comply"

Prompt format (from paper Section 3.1):
  Write a creative short story using a maximum of five sentences.
  The story must include the following three words: {cue_words}.

Dataset contains 479 pre-generated stories (236 human, 243 AI) with expert and non-expert
ratings on creativity, originality, surprise, and value. We use human-written stories
as references for evaluation.

Fields used: item_id (cue words), story (for human-authored stories as references)
Fields skipped: author (metadata), expert_*/non_expert_* (ratings for specific stories
in dataset, not applicable to new generations)

Evaluation: Open-ended generation (BLEU, ROUGE, F1 against human references)
Alternative: LLM-as-judge for creativity dimensions (see metric_notes.md)
"""

from typing import List
from collections import defaultdict
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)
from datasets import load_dataset


class CueWordStoryScenario(Scenario):
    """
    Cue-word-based Creative Story Generation

    Evaluates models' ability to generate creative short stories incorporating
    specific cue words with varying semantic distances.
    """

    name = "cue_word_story"
    description = "mismayil/creative_story_generation_dataset"
    tags = ["creativity", "story_generation", "creative_writing"]

    def __init__(self, semantic_distance: str = "all"):
        """
        Args:
            semantic_distance: "all", "low", or "high"
                - "all": All 4 cue word sets (default)
                - "low": Low semantic distance sets (stamp-letter-send, petrol-diesel-pump)
                - "high": High semantic distance sets (gloom-payment-exist, organ-empire-comply)
        """
        super().__init__()
        self.semantic_distance = semantic_distance.lower()

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load cue-word story generation dataset and create instances.

        Each instance contains:
        - Prompt: Instruction to write creative story with 3 cue words
        - References: Human-written stories for those cue words
        """
        # Load dataset from HuggingFace
        dataset = load_dataset("mismayil/creative_story_generation_dataset", split="train")

        # Cue word sets with semantic distance classification
        # From paper Section 3.1
        cue_word_sets = {
            "stamp-letter-send": {"words": ["stamp", "letter", "send"], "distance": "low"},
            "petrol-diesel-pump": {"words": ["petrol", "diesel", "pump"], "distance": "low"},
            "gloom-payment-exist": {"words": ["gloom", "payment", "exist"], "distance": "high"},
            "organ-empire-comply": {"words": ["organ", "empire", "comply"], "distance": "high"},
        }

        # Filter cue word sets based on semantic_distance parameter
        if self.semantic_distance == "low":
            selected_sets = {k: v for k, v in cue_word_sets.items() if v["distance"] == "low"}
        elif self.semantic_distance == "high":
            selected_sets = {k: v for k, v in cue_word_sets.items() if v["distance"] == "high"}
        else:  # "all"
            selected_sets = cue_word_sets

        # Group stories by item_id and filter for human-authored stories
        human_stories_by_item = defaultdict(list)
        for example in dataset:
            item_id = example["item_id"]
            # Only use human-authored stories as references
            if example["author"].lower() == "human":
                human_stories_by_item[item_id].append(example["story"])

        instances = []
        for item_id, cue_info in selected_sets.items():
            cue_words = cue_info["words"]
            cue_words_str = ", ".join(cue_words)

            # Build prompt based on paper instructions (Section 3.1)
            prompt = (
                f"Write a creative short story using a maximum of five sentences. "
                f"The story must include the following three words: {cue_words_str}."
            )

            # Get human-written reference stories
            reference_stories = human_stories_by_item.get(item_id, [])

            # Create references from human stories
            references = [
                Reference(Output(text=story), tags=[CORRECT_TAG])
                for story in reference_stories
            ]

            # Create instance
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"cue_word_story_{item_id}",
                )
            )

        return instances
