"""
HELM Scenario: Fann or Flop — Arabic Poetry Understanding

Paper: "Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs"
       arXiv:2505.18152, EMNLP 2025
Code:  https://github.com/mbzuai-oryx/FannOrFlop

Task: Given an Arabic poem (with title, author, era, genre, and meter), generate a
verse-by-verse explanation in Arabic capturing literal meaning, thematic depth,
cultural context, and literary devices.

Dataset: HuggingFace: omkarthawakar/FannOrFlop (split: "train" — only split available)
         6,984 Arabic poem-explanation pairs spanning 12 historical eras and 21 genres.
         Dataset is publicly accessible (no HuggingFace token required).

Fields used:
  title       — poem title (Arabic)
  author      — poet name (Arabic)
  era         — historical literary era (Arabic, e.g., العصر العباسي)
  meter       — poetic meter (Arabic, e.g., الكامل)
  genre       — genre/theme (Arabic, e.g., مدح)
  poem_verses — full poem text, numbered verse by verse (Arabic)
  raw_explanation — full gold explanation as a paragraph (reference for BLEU/BERTScore)

Fields skipped:
  id, source  — metadata only
  tags        — redundant with era/meter/genre
  verse_count — computed from poem_verses
  explanation — structured verse-level explanation list; raw_explanation is the
                single-string gold reference used for BLEU/BERTScore scoring

Prompt source: No explicit model prompt specified in the paper or README.
  Standard Arabic instruction used (note: paper evaluation compared model outputs
  against gold using BLEU, chrF++, BERTScore, and LLM judge).

Evaluation: llm_judge (faithfulness 1-5, fluency 1-5, overall 1-5 via GPT-4o)
            + open_ended metrics (BLEU, chrF++, BERTScore vs raw_explanation)
            See annotator_notes.md for judge configuration.

Note: "not_suitable" flag for non-English is invalid — multilingual benchmarks are
fully supported in HELM. Arabic-language creative tasks are in scope.

Parameters:
  era:   filter by Arabic era string (e.g., "العصر العباسي") or "all" (default)
  genre: filter by Arabic genre string (e.g., "مدح") or "all" (default)
"""

from typing import List

from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    Scenario,
)

# Standard Arabic instruction (no explicit prompt in paper; standard generation format)
_INSTRUCTION = (
    "أنت خبير في الأدب العربي والشعر. سيُقدَّم إليك قصيدة عربية كاملة، "
    "والمطلوب منك تقديم شرح مفصّل لكل بيت من أبياتها. "
    "ينبغي أن يتضمّن شرحك: المعنى الحرفي، والعمق الموضوعي، والسياق الثقافي، "
    "والصور الأدبية، والأسلوب التعبيري."
)

_PROMPT_TEMPLATE = """{instruction}

عنوان القصيدة: {title}
الشاعر: {author}
الحقبة الأدبية: {era}
البحر: {meter}
النوع: {genre}

القصيدة:
{poem_verses}

اشرح كل بيت من أبيات القصيدة بيتاً بيتاً."""


class FannOrFlopScenario(Scenario):
    """
    Fann or Flop: Arabic poetry verse-explanation generation.

    6,984 poems spanning 12 historical eras and 21 genres. The model receives
    a full Arabic poem with metadata and must generate a verse-by-verse
    explanation in Arabic. Evaluated via LLM judge (faithfulness, fluency,
    overall 1-5) and BLEU/BERTScore against gold raw_explanation.

    Optional filters by era or genre (Arabic strings).
    """

    name = "fann_or_flop"
    description = "omkarthawakar/FannOrFlop (arXiv:2505.18152)"
    tags = ["creativity", "arabic", "poetry", "multilingual", "open_ended_generation"]

    def __init__(self, era: str = "all", genre: str = "all"):
        super().__init__()
        self.era = era
        self.genre = genre

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("omkarthawakar/FannOrFlop", split="train")

        instances = []
        for item in dataset:
            if self.era != "all" and item.get("era") != self.era:
                continue
            if self.genre != "all" and item.get("genre") != self.genre:
                continue

            prompt = _PROMPT_TEMPLATE.format(
                instruction=_INSTRUCTION,
                title=item["title"] or "",
                author=item["author"] or "",
                era=item["era"] or "",
                meter=item["meter"] or "",
                genre=item["genre"] or "",
                poem_verses=item["poem_verses"] or "",
            )

            # raw_explanation is the gold paragraph-form explanation (BLEU/BERTScore reference)
            references = [
                Reference(Output(text=item["raw_explanation"] or ""), tags=[CORRECT_TAG])
            ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances  # 6,984 total (all); subset when era/genre filter applied
