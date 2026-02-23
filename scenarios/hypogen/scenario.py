"""
HELM Scenario: HypoGen — Scientific Hypothesis Generation via Bit-Flip

Paper: Can LLMs Generate Novel Research Ideas?
       (bit-flip hypothesis framework)
       https://arxiv.org/abs/2409.04109 (HypoGen dataset)
Dataset: UniverseTBD/hypogen-dr1
         Test split: 50 examples

Task: Given a research paper abstract and a description of the conventional
approach or limitation (the "bit"), generate a novel hypothesis or approach
that addresses it (the "flip"). Tests scientific creativity — the ability
to recognize a limiting assumption and propose a genuinely new direction.

The "bit-flip" framework comes from the paper's conceptual model:
  bit  = the conventional wisdom or limiting belief about a problem
  flip = the novel insight the paper proposes, inverting or overcoming that limit

Prompt format (no explicit prompt published — standard instruction used):
  The following is an abstract from a research paper and a description of
  a conventional approach or limitation (the "bit").

  Abstract:
  {abstract}

  Conventional approach / limitation (bit):
  {bit}

  Propose a novel research hypothesis or approach that overcomes this
  limitation (the "flip"):

Prompt source: Standard instruction format (paper does not publish exact prompts)
Fields used: abstract + bit (input), flip (gold novel hypothesis)
Fields skipped: title, authors, venue, year, citation (metadata),
  spark (short label, not ground truth), chain_of_reasoning (model-generated
  reasoning, not ground truth), url, pdf_url, paper_id

Evaluation: open_ended (BLEU, ROUGE against author-written flip)
"""

from typing import List

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


class HypoGenScenario(Scenario):
    """
    Scientific hypothesis generation via the bit-flip framework.

    Given a paper abstract and a description of the conventional
    limitation (bit), generate the novel approach proposed in the
    paper (flip).
    """

    name = "hypogen"
    description = "UniverseTBD/hypogen-dr1"
    tags = ["creativity", "scientific_creativity", "hypothesis_generation", "open_ended"]

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("UniverseTBD/hypogen-dr1", split="test")

        instances = []
        for item in dataset:
            abstract = (item["abstract"] or "").strip()
            bit = (item["bit"] or "").strip()
            flip = (item["flip"] or "").strip()

            if not abstract or not bit or not flip:
                continue

            prompt = (
                "The following is an abstract from a research paper and a description "
                "of a conventional approach or limitation (the \"bit\").\n\n"
                f"Abstract:\n{abstract}\n\n"
                f"Conventional approach / limitation (bit):\n{bit}\n\n"
                "Propose a novel research hypothesis or approach that overcomes "
                "this limitation (the \"flip\"):"
            )

            instances.append(Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=flip), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            ))

        return instances
