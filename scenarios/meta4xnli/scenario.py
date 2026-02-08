"""
HELM Scenario: Meta4XNLI

Paper: https://arxiv.org/abs/2404.07053 (LREC-COLING 2024)
Code: https://github.com/elisanchez-beep/meta4xnli

A crosslingual parallel corpus for metaphor detection and interpretation in
English and Spanish, built on XNLI and esXNLI datasets.

Subsets:

  "interpretation_en" / "interpretation_es" (Zero-shot):
    - NLI task on sentence pairs containing metaphorical language
    - 580 instances each (xnli_test_met), eval: exact_match
    - Labels: entailment, neutral, contradiction

  "interpretation_en_cot" / "interpretation_es_cot" (Chain-of-thought):
    - Same task with few-shot examples in prompt
    - 580 instances each, eval: exact_match

  "detection_en" (Metaphor Detection):
    - Token-level metaphor detection (sequence labeling)
    - 3630 instances (test split), eval: open_ended
    - Tags: 0=literal, 1=metaphor

Prompt formats (from Paper Appendix Table 29):

  Zero-shot:
    Say which is the inference relationship between these two sentences.
    Please, answer only with one word between 'entailment', 'neutral' or 'contradiction'.
    {Premise} -> {Hypothesis}:

  Chain-of-thought:
    You are an expert linguist and your task is to annotate sentences for
    the task of Natural Language Inference. This task consists in determining
    if a first sentence (premise) entails, contradicts or does not entail nor
    contradict the second sentence (hypothesis). Please, answer with one of
    the following labels: 'entailment', 'contradiction' or 'neutral'.

    Here you have a few examples:
    Premise: I am an open person.
    Hypothesis: I am friendly.
    Answer: entailment
    [... more examples ...]

    Premise: {premise}
    Hypothesis: {hypothesis}
    Answer:

Prompt source: Paper Appendix Table 29
Fields used: sentence1, sentence2, gold_label, language
Fields skipped: promptID, pairID, genre, source_dataset
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class Meta4XNLIScenario(Scenario):
    name = "meta4xnli"
    description = "HiTZ/meta4xnli"
    tags = ["creativity", "metaphor", "nli", "figurative_language"]

    SUBSETS = [
        "interpretation_en",
        "interpretation_es",
        "interpretation_en_cot",
        "interpretation_es_cot",
        "detection_en"
    ]

    # Chain-of-thought prompt from Table 29
    COT_PROMPT = """You are an expert linguist and your task is to annotate sentences for the task of Natural Language Inference. This task consists in determining if a first sentence (premise) entails, contradicts or does not entail nor contradict the second sentence (hypothesis). Please, answer with one of the following labels: 'entailment', 'contradiction' or 'neutral'.

Here you have a few examples:
Premise: I am an open person.
Hypothesis: I am friendly.
Answer: entailment

Premise: My heart is broken.
Hypothesis: I am happy.
Answer: contradiction

Premise: You are a close friend.
Hypothesis: I like your eyes.
Answer: neutral

Premise: {premise}
Hypothesis: {hypothesis}
Answer:"""

    def __init__(self, subset: str = "interpretation_en"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"subset must be one of {self.SUBSETS}, got '{subset}'")
        self.subset = subset

    def get_instances(self, output_path: str):
        if self.subset == "interpretation_en":
            return self._get_interpretation_instances(language="en", use_cot=False)
        elif self.subset == "interpretation_es":
            return self._get_interpretation_instances(language="es", use_cot=False)
        elif self.subset == "interpretation_en_cot":
            return self._get_interpretation_instances(language="en", use_cot=True)
        elif self.subset == "interpretation_es_cot":
            return self._get_interpretation_instances(language="es", use_cot=True)
        elif self.subset == "detection_en":
            return self._get_detection_instances()

    def _get_interpretation_instances(self, language: str, use_cot: bool = False):
        """NLI task on metaphorical sentence pairs."""
        dataset = load_dataset(
            "HiTZ/meta4xnli",
            "int_eval",
            split="xnli_test_met",
            trust_remote_code=True
        )

        instances = []
        for item in dataset:
            # Filter by language
            if item["language"] != language:
                continue

            if use_cot:
                # Chain-of-thought prompt (from Paper Appendix Table 29)
                prompt = self.COT_PROMPT.format(
                    premise=item['sentence1'],
                    hypothesis=item['sentence2']
                )
            else:
                # Zero-shot prompt (from Paper Appendix Table 29)
                prompt = (
                    "Say which is the inference relationship between these two sentences. "
                    "Please, answer only with one word between 'entailment', 'neutral' or 'contradiction'.\n"
                    f"{item['sentence1']} -> {item['sentence2']}:"
                )

            # Expected answer is the label word itself
            correct_label = item["gold_label"]

            # HELM pattern: all choices become References, correct one tagged
            references = []
            for label in ["entailment", "neutral", "contradiction"]:
                is_correct = (label == correct_label)
                tags = [CORRECT_TAG] if is_correct else []
                references.append(Reference(Output(text=label), tags=tags))

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances

    def _get_detection_instances(self):
        """Token-level metaphor detection task."""
        dataset = load_dataset(
            "HiTZ/meta4xnli",
            "det_en_finetune",
            split="test",
            trust_remote_code=True
        )

        instances = []
        for item in dataset:
            tokens = item["tokens"]
            tags = item["tags"]

            # Format: present sentence, ask which tokens are metaphorical
            sentence = " ".join(tokens)

            # Build expected output: tokens marked as metaphor (tag=1)
            metaphor_tokens = [t for t, tag in zip(tokens, tags) if tag == 1]

            prompt = (
                f"Identify the metaphorical words or phrases in the following sentence. "
                f"List only the metaphorical words, separated by commas. "
                f"If there are no metaphors, respond with \"None\".\n\n"
                f"Sentence: {sentence}"
            )

            # Reference is the list of metaphor tokens (or "None")
            if metaphor_tokens:
                answer = ", ".join(metaphor_tokens)
            else:
                answer = "None"

            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
