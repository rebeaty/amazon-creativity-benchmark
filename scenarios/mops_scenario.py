"""
HELM Scenario: MoPS Premise Evaluation

Paper: "MoPS: Modular Story Premise Synthesis for Evaluating Creative Writing" (arXiv:2406.05690)
Dataset: ManTle/mops on HuggingFace (https://huggingface.co/datasets/ManTle/mops)

Task: Given structured story components (theme, background, persona, event, ending, twist),
synthesize a coherent story premise sentence.

Prompt format (standard synthesis format; no explicit task prompt specified in paper):
  Given the following story components, write a concise story premise (1-3 sentences)
  that integrates all elements into a compelling narrative setup.

  Theme: {theme}
  Background: {background}
  Character: {persona}
  Event: {event}
  Ending: {ending}
  Twist: {twist}

  Story Premise:

Fields used: theme, background, persona, event, ending, twist (inputs); premise (reference)
Fields skipped: novel, script (long model-generated story outputs), id
Split: curated (100 highest-quality premises across 14 themes)

Eval: LLM-as-judge — fascination, completeness, originality (0-100 each, GPT-4-turbo);
      see annotator_notes.md. BLEU/ROUGE also applicable as proxy metrics.
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class MoPSPremiseScenario(Scenario):
    name = "mops_premise"
    description = "ManTle/mops"
    tags = ["creativity", "story_generation", "narrative"]

    @staticmethod
    def _strip_prefix(text, prefix):
        """Remove an embedded field label prefix if present (e.g. 'Twist: ...')."""
        stripped = text.strip()
        if stripped.lower().startswith(prefix.lower() + ":"):
            stripped = stripped[len(prefix) + 1:].strip()
        return stripped

    def get_instances(self, output_path):
        dataset = load_dataset("ManTle/mops", split="curated")

        instances = []
        for item in dataset:
            prompt = (
                "Given the following story components, write a concise story premise "
                "(1-3 sentences) that integrates all elements into a compelling narrative setup.\n\n"
                f"Theme: {item['theme']}\n"
                f"Background: {self._strip_prefix(item['background'], 'background')}\n"
                f"Character: {self._strip_prefix(item['persona'], 'persona')}\n"
                f"Event: {self._strip_prefix(item['event'], 'event')}\n"
                f"Ending: {self._strip_prefix(item['ending'], 'ending')}\n"
                f"Twist: {self._strip_prefix(item['twist'], 'twist')}\n\n"
                "Story Premise:"
            )

            instances.append(Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=item["premise"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            ))

        return instances
