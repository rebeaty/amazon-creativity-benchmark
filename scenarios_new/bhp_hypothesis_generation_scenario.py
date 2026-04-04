"""
HELM Scenario: BHP - Background and Hypothesis Pairs for Biomedical Hypothesis Generation

Paper: Large Language Models as Biomedical Hypothesis Generators: A Comprehensive Evaluation
        Biqing Qi, Kaiyan Zhang, Kai Tian, Haoxiang Li, Zhang-Ren Chen, Sihang Zeng,
        Ermo Hua, Hu Jinfang, Bowen Zhou
        COLM 2024
        https://arxiv.org/abs/2407.08940

Code: https://github.com/TsinghuaC3I/LLM4BioHypoGen

Dataset: Background-Hypothesis Pairs from biomedical literature
  - Test Seen: 190 examples (from earlier literature, likely in training data)
  - Test Unseen: 197 examples (from recent literature, less likely in training)
  - Total: 387 examples

Task: Generate scientific hypotheses from biomedical research backgrounds.
      Given numbered background points from literature, generate plausible hypotheses
      that are novel, coherent, and scientifically grounded.

Prompt format (with few-shot examples from repository):
  You are a renowned biomedical researcher. Based on the research background below,
  generate a scientifically grounded and novel hypothesis.

  [Few-shot examples included]

  Research Background:
  {background}

  Generate a hypothesis in the following format:
  (1) [First hypothesis point]
  (2) [Second hypothesis point]
  (3) [Third hypothesis point]

  Hypothesis:

Evaluation:
  - Primary: Open-ended generation evaluated with BLEU, ROUGE against ground truth
  - Optional: Paper proposes 4 novel metrics (LLM-based and human evaluation):
    * Novelty (is the hypothesis innovative?)
    * Coherence (is it logically consistent?)
    * Scientific validity (is it scientifically sound?)
    * Feasibility (can it be tested?)

Fields used: background (input), hypothesis (reference)

Note: The paper's multi-agent framework (Analyst, Scientist, Critic) is used for
      hypothesis generation in their experiments, but for benchmarking arbitrary LLMs,
      we use a simpler prompt format with few-shot examples.

Data splits: "seen" contains hypotheses from literature published before a cutoff date
            (likely in LLM training data), while "unseen" contains recent hypotheses
            (less likely contaminated). Both can be used for evaluation.
"""

import json
import os
from typing import List, Optional
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class BHPHypothesisGenerationScenario(Scenario):
    """
    BHP (Background and Hypothesis Pairs) for biomedical hypothesis generation.

    Models are tasked with generating scientific hypotheses from research backgrounds.
    """

    name = "bhp_hypothesis_generation"
    description = "TsinghuaC3I/LLM4BioHypoGen"  # GitHub repo
    tags = ["creativity", "hypothesis-generation", "scientific-reasoning", "biomedical", "open-ended"]

    # Few-shot examples from the repository (used in prompts_wo_tool.py)
    FEW_SHOT_EXAMPLES = [
        {
            "background": " (1) Neonatal intensive care is associated with long-term health problems in children such as cerebral palsy, mental retardation, deafness, blindness, learning disabilities, and behavioral problems. (2) Mothers of preterm infants experience more severe psychological distress compared to mothers of healthy full-term infants, but the impact of caregiving on parents of children discharged from NICUs is not well-researched. (3) Parents of NICU children show no difference in psychosocial health compared to parents of healthy full-term children.",
            "hypothesis": " (1) The mental health of parents of NICU children may improve over time due to adaptation and relief from initial fear and anxiety. (2) Child characteristics, such as health status, behavior problems, and birth-related risk factors, may influence parental psychosocial health. (3) Certain factors, such as caregiver strain, family function, and demographic variables, may predict parental psychosocial health."
        },
        {
            "background": " (1) Recruitment of tumor supporting stromal cells and tissue remodeling in the tumor microenvironment support cancer cell proliferation, invasion, metastasis, and drug resistance. (2) Mesenchymal stem cells (MSC) are recruited by cancer cells into the tumor site and play a role in modulating tumor progression. (3) Intratumoral heterogeneity exists in solid tumors, with cancer stem cells (CSCs) and clonal evolution contributing to the complexity of cancer.",
            "hypothesis": " (1) Transcriptional regulators are responsible for tumor-supporting stromal reprogramming, specifically in MSC in the tumor stroma. (2) Intercellular communication between cancer cells and recruited MSCs is mediated by cell-to-cell contact, paracrine interactions, and microvesicles. (3) Epithelial cancer cell plasticity is regulated by tumor stroma interaction signals, enabling non-CSCs to convert into CSCs."
        },
        {
            "background": " (1) Transitions in care are complex and require coordination and communication among different healthcare providers. (2) The experiences of two different patients during care transitions were significantly different. (3) Major gaps in care occur during client handoffs, leading to misunderstandings, errors, and negative outcomes.",
            "hypothesis": " (1) Differences in care transitions may be attributed to unique client needs. (2) Differences in the way healthcare providers respond to client needs may contribute to varied experiences. (3) Existing regulatory standards may not adequately address safety issues in care transitions."
        }
    ]

    def __init__(self, split: str = "both", model: str = "gpt-4"):
        """
        Args:
            split: Which test split to use. Options: ["seen", "unseen", "both"]
                  "seen" = 190 examples from earlier literature (likely in training data)
                  "unseen" = 197 examples from recent literature (less contamination)
                  "both" = All 387 examples
            model: Which model's extraction to use. Options: ["gpt-3.5", "gpt-4"]
                  Background-hypothesis pairs were extracted using GPT-3.5 or GPT-4.
                  Default: "gpt-4" (higher quality)
        """
        super().__init__()
        if split not in ["seen", "unseen", "both"]:
            raise ValueError(f"Invalid split: {split}. Must be 'seen', 'unseen', or 'both'")
        if model not in ["gpt-3.5", "gpt-4"]:
            raise ValueError(f"Invalid model: {model}. Must be 'gpt-3.5' or 'gpt-4'")

        self.split = split
        self.model = model

    def download_dataset(self, output_path: str) -> tuple:
        """Download the test_seen and test_unseen data files."""
        base_url = f"https://raw.githubusercontent.com/TsinghuaC3I/LLM4BioHypoGen/main/data/{self.model}"

        seen_url = f"{base_url}/test_seen.json"
        unseen_url = f"{base_url}/test_unseen.json"

        seen_path = os.path.join(output_path, f"test_seen_{self.model.replace('.', '_')}.json")
        unseen_path = os.path.join(output_path, f"test_unseen_{self.model.replace('.', '_')}.json")

        ensure_file_downloaded(source_url=seen_url, target_path=seen_path)
        ensure_file_downloaded(source_url=unseen_url, target_path=unseen_path)

        return seen_path, unseen_path

    def load_dataset(self, file_path: str) -> List[dict]:
        """Load background-hypothesis pairs from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def create_prompt(self, background: str) -> str:
        """
        Create the prompt for hypothesis generation with few-shot examples.
        """
        # Build few-shot examples
        few_shot_text = ""
        for i, example in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            few_shot_text += f"\nExample {i}:\n"
            few_shot_text += f"Background:{example['background']}\n"
            few_shot_text += f"Hypothesis:{example['hypothesis']}\n"

        prompt = (
            "You are a renowned biomedical researcher. Based on the research background below, "
            "generate a scientifically grounded and novel hypothesis.\n"
            f"{few_shot_text}\n"
            f"Research Background:{background}\n\n"
            "Generate a hypothesis in the following format:\n"
            "(1) [First hypothesis point]\n"
            "(2) [Second hypothesis point]\n"
            "(3) [Third hypothesis point]\n\n"
            "Hypothesis:"
        )

        return prompt

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate instances for biomedical hypothesis generation.

        Creates instances from test_seen and/or test_unseen based on split parameter.
        """
        # Download datasets
        seen_path, unseen_path = self.download_dataset(output_path)

        instances = []
        instance_id = 0

        # Process test_seen
        if self.split in ["seen", "both"]:
            seen_data = self.load_dataset(seen_path)

            for item in seen_data:
                prompt_text = self.create_prompt(item['background'])

                # Reference is the ground truth hypothesis from literature
                references = [
                    Reference(Output(text=item['hypothesis']), tags=[CORRECT_TAG])
                ]

                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"bhp_seen_{instance_id}"
                    )
                )
                instance_id += 1

        # Process test_unseen
        if self.split in ["unseen", "both"]:
            unseen_data = self.load_dataset(unseen_path)

            for item in unseen_data:
                prompt_text = self.create_prompt(item['background'])

                references = [
                    Reference(Output(text=item['hypothesis']), tags=[CORRECT_TAG])
                ]

                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"bhp_unseen_{instance_id}"
                    )
                )
                instance_id += 1

        return instances
