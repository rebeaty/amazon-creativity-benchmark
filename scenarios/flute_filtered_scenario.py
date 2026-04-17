"""
HELM Scenario: FLUTE (Filtered) — Rhetorical Language

Paper: "Rhet2Pix: Generating Images from Rhetorical Figures of Speech"
       arXiv:2505.22792 (May 2025)
Original FLUTE dataset: ColumbiaNLP/FLUTE (EMNLP 2022)

FLUTE (Figurative Language Understanding through Textual Explanations) is a
natural language inference benchmark for figurative language. Given a
hypothesis containing figurative language and a literal premise, the model
must determine whether they Entail or Contradict each other, then explain
the figurative meaning.

The Rhet2Pix paper (arXiv:2505.22792) uses a filtered subset of FLUTE
containing only Metaphor and Simile examples (1,250 each = 2,500 total),
selected because they are "rhetorical devices with clear visual
interpretability". Sarcasm (no visual component), Idiom (literal imagery
misleading), and CreativeParaphrase (focus on wording) are excluded.

Task:
  Given a figurative hypothesis and a literal premise, predict whether
  the hypothesis entails or contradicts the premise, and explain the
  figurative meaning.

Modes:
  - "classification": predict label only (Entailment / Contradiction)
                      → eval with exact_match
  - "explanation": predict label + explain figurative meaning
                   → reference: "Label\\n\\nExplanation: {explanation}"
                   → eval with open_ended (BLEU/ROUGE/BERTScore)

Prompt source: No explicit prompt specified in either paper; standard NLI
  format used (noted in header). Original FLUTE paper uses no system prompt.

Fields used: hypothesis, premise, label, explanation, type
Fields skipped: id, idiom (always null for Metaphor/Simile), split

Split note: All 7,534 FLUTE examples are in the 'train' split only;
  no test/validation split exists. TEST_SPLIT tag is applied for HELM
  consistency but data is loaded from 'train'.

Type distribution after filtering:
  Metaphor: 1,250  |  Simile: 1,250  |  Total: 2,500
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

FILTERED_TYPES = {"Metaphor", "Simile"}


class FLUTEFilteredScenario(Scenario):
    name = "flute_filtered"
    description = "ColumbiaNLP/FLUTE"
    tags = ["creativity", "figurative_language", "natural_language_inference"]

    def __init__(self, mode: str = "classification"):
        """
        Args:
            mode: "classification" — predict Entailment or Contradiction only
                                     (exact_match eval)
                  "explanation"    — predict label and explain figurative meaning
                                     (open_ended eval)
        """
        super().__init__()
        assert mode in ("classification", "explanation"), (
            f"mode must be 'classification' or 'explanation', got '{mode}'"
        )
        self.mode = mode

    def get_instances(self, output_path):
        dataset = load_dataset("ColumbiaNLP/FLUTE", split="train")

        instances = []
        for item in dataset:
            if item["type"] not in FILTERED_TYPES:
                continue

            prompt = (
                f"Premise: {item['premise']}\n"
                f"Hypothesis: {item['hypothesis']}\n\n"
                f"Does the hypothesis entail or contradict the premise? "
                f"The hypothesis uses figurative language ({item['type'].lower()}). "
                f"Answer with 'Entailment' or 'Contradiction'."
            )

            if self.mode == "classification":
                references = [
                    Reference(Output(text=item["label"]), tags=[CORRECT_TAG]),
                ]
            else:
                # explanation mode: full reference = label + explanation
                reference_text = f"{item['label']}\n\nExplanation: {item['explanation']}"
                references = [
                    Reference(Output(text=reference_text), tags=[CORRECT_TAG])
                ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    extra_data={
                        "type": item["type"],
                        "id": item["id"],
                    },
                )
            )

        return instances
