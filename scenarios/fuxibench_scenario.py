"""
HELM Scenario: FuxiBench (Fùxì)

Paper: "Fuxi: A Benchmark for Evaluating Language Models on Ancient Chinese
       Text Understanding and Generation" (arXiv:2503.15837)
Code: https://github.com/cubenlp/FuxiBench

FuxiBench evaluates LLMs on classical Chinese text understanding and generation
across 21 tasks. This scenario onboards the 5 creative generation subtasks:

  - ci_gen: Compose ci poetry (词) following strict cipai (词牌) tonal patterns
            (300 instances, format accuracy via rule-based template matching)
  - couplet_gen: Generate matching second line for a couplet (对联) with
                 parallel structure (500 instances, format accuracy via
                 segment/character count matching)
  - poem_gen: Compose original poetry from thematic keywords
              (300 instances, LLM-as-judge correctness)
  - poem_nmt_inv: Reconstruct classical poems from modern Chinese translations
                  (300 instances, LLM-as-judge correctness)
  - poem_appre: Write literary appreciation/analysis of a given poem
                (109 instances, BLEU against reference)

Prompt source: Exact instruction text from each JSON data file's "instruction"
  field. These are the prompts used in the paper's evaluation.
Fields used: instruction, input, output, sample_id, cipai (ci_gen only),
  template (couplet_gen only)
Fields skipped: format_standard (ci_gen; verbose cipai rule descriptions,
  used internally by the format evaluator, not part of the model prompt)

Skipped tasks (not creativity-related):
  - ac_rc, idiom_rc, tcmsd_rc (reading comprehension / MC knowledge)
  - lc_qa, as_qa, book_author, book_dynasty, book_collection (knowledge QA)
  - poem_line_st, famous_quote_st, idiom_st (source tracing / knowledge recall)
  - poem_nmt, ac_nmt (translation)
  - idiom_exp (explanation)
  - tcm_qa, prescription_exp (traditional Chinese medicine)

Eval types by subset:
  - ci_gen: custom (rule-based cipai format checking — pacc)
  - couplet_gen: custom (rule-based parallel structure checking — cacc)
  - poem_gen, poem_nmt_inv: llm_judge (lacc via fine-tuned Qwen2-7B-Instruct,
    validated at 89.8% accuracy, Cohen's kappa=0.764 on 1K human annotations)
  - poem_appre: open_ended (BLEU against reference appreciation)
"""

import json
import os
import urllib.request
import zipfile

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)

# GitHub raw URL base for test data files
_RAW_BASE = "https://raw.githubusercontent.com/cubenlp/FuxiBench/main/test_data"

# Creative subtasks and their data filenames
_CREATIVE_TASKS = {
    "ci_gen": "ci_gen.json",
    "couplet_gen": "couplet_gen.json",
    "poem_gen": "poem_gen.json",
    "poem_nmt_inv": "poem_nmt_inv.json",
    "poem_appre": "poem_appre.json",
}


class FuxiBenchScenario(Scenario):
    name = "fuxibench"
    description = "cubenlp/FuxiBench"
    tags = ["creativity", "chinese", "poetry", "constrained_generation"]

    SUBSETS = list(_CREATIVE_TASKS.keys())

    def __init__(self, subset: str = "ci_gen"):
        """
        Args:
            subset: One of "ci_gen", "couplet_gen", "poem_gen",
                    "poem_nmt_inv", "poem_appre"
        """
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(
                f"subset must be one of {self.SUBSETS}, got '{subset}'"
            )
        self.subset = subset

    def _download_data(self, output_path: str) -> list:
        """Download the JSON data file for the selected subset."""
        filename = _CREATIVE_TASKS[self.subset]
        filepath = os.path.join(output_path, filename)
        if not os.path.exists(filepath):
            url = f"{_RAW_BASE}/{filename}"
            urllib.request.urlretrieve(url, filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_instances(self, output_path: str):
        data = self._download_data(output_path)

        instances = []
        for item in data:
            # Build prompt from instruction + input (exact paper format)
            instruction = item["instruction"]
            user_input = item["input"]
            prompt = f"{instruction}\n{user_input}"

            # Reference is the gold output
            gold = item.get("output", "")
            references = [
                Reference(Output(text=gold), tags=[CORRECT_TAG])
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=item.get("sample_id", ""),
            ))

        return instances
