"""
HELM Scenario: LLM4BioHypoGen - Biomedical Hypothesis Generation

Paper: Large Language Models as Biomedical Hypothesis Generators: A Comprehensive Evaluation (COLM 2024)
       https://arxiv.org/abs/2407.08940
       Earlier version: https://arxiv.org/abs/2311.05965 (NeurIPS 2023 Workshop)
Code: https://github.com/TsinghuaC3I/LLM4BioHypoGen

Task: Generate scientific hypotheses from biomedical research backgrounds.
      Given background statements extracted from literature, generate novel,
      scientifically grounded hypotheses that extend current understanding.

Dataset: 787 background-hypothesis pairs extracted from biomedical literature
- GPT-3.5 generated: 400 examples (200 seen, 200 unseen)
- GPT-4 generated: 387 examples (190 seen, 197 unseen)
- Seen: Literature published before cutoff date (models may have seen during training)
- Unseen: Literature published after cutoff date (truly novel for evaluation)

Prompt format: Table 11 - Few-shot examples for hypothesis generation (5-shot)
  You are a renowned biomedical researcher. You can give novel hypothesis for the
  background based on your exist knowledge. Please follow the given examples and give
  the hypothesis in the SINGLE TURN.

  [5 Background/Hypothesis example pairs]

  Background: {background}
  Hypothesis:

Fields used: background, hypothesis
Note: Each background/hypothesis consists of multiple numbered statements

Evaluation: llm_judge + open_ended
  - BLEU, ROUGE-L for ground truth comparison
  - GPT-4 judge on 4 dimensions (0-3 scale):
    * Novelty: Does hypothesis introduce new information or perspectives?
    * Relevance: Is hypothesis aligned with the background?
    * Significance: Does hypothesis have potential scientific impact?
    * Verifiability: Can hypothesis be tested using existing methods/data?
  See annotator_notes.md for complete evaluation setup
"""

import json
import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output, Reference,
    TEST_SPLIT,
)


class LLM4BioHypoGenScenario(Scenario):
    """LLM4BioHypoGen: Biomedical Hypothesis Generation

    Evaluates models' ability to generate novel, scientifically grounded hypotheses
    from biomedical research backgrounds.
    """

    name = "llm4biohypogen"
    description = "TsinghuaC3I/LLM4BioHypoGen"
    tags = ["creativity", "scientific_reasoning", "hypothesis_generation", "biomedical"]

    # GitHub raw URLs for test data
    BASE_URL = "https://raw.githubusercontent.com/TsinghuaC3I/LLM4BioHypoGen/main/data"

    def __init__(self, model_version: str = "gpt-3.5", test_type: str = "seen"):
        """
        Args:
            model_version: Which model was used to generate the dataset ('gpt-3.5' or 'gpt-4')
            test_type: Test set type ('seen' or 'unseen')
        """
        super().__init__()
        if model_version not in ["gpt-3.5", "gpt-4"]:
            raise ValueError(f"Invalid model_version: {model_version}. Must be 'gpt-3.5' or 'gpt-4'")
        if test_type not in ["seen", "unseen"]:
            raise ValueError(f"Invalid test_type: {test_type}. Must be 'seen' or 'unseen'")

        self.model_version = model_version
        self.test_type = test_type

    def get_instances(self, output_path: str) -> List[Instance]:
        # Construct URL for the specific test file
        data_url = f"{self.BASE_URL}/{self.model_version}/test_{self.test_type}.json"

        # Download data
        with urllib.request.urlopen(data_url) as response:
            data = json.loads(response.read().decode("utf-8"))

        instances = []
        for item in data:
            instances.append(self._create_instance(item))

        return instances

    def _create_instance(self, item: dict) -> Instance:
        """Create an instance from a background-hypothesis pair"""

        background = item["background"]
        hypothesis = item["hypothesis"]

        # Build prompt following Table 11 format (exact prompt from paper)
        prompt = """You are a renowned biomedical researcher. You can give novel hypothesis for the background based on your exist knowledge. Please follow the given examples and give the hypothesis in the SINGLE TURN.
Background:
(1) Neonatal intensive care is associated with long-term health problems in children such as cerebral palsy, mental retardation, deafness, blindness, learning disabilities, and behavioral problems.
(2) Mothers of preterm infants experience more severe psychological distress compared to mothers of healthy full-term infants, but the impact of caregiving on parents of children discharged from NICUs is not well-researched.
(3) Parents of NICU children show no difference in psychosocial health compared to parents of healthy full-term children.
Hypothesis:
(1) The mental health of parents of NICU children may improve over time due to adaptation and relief from initial fear and anxiety.
(2) Child characteristics, such as health status, behavior problems, and birth-related risk factors, may influence parental psychosocial health.
(3) Certain factors, such as caregiver strain, family function, and demographic variables, may predict parental psychosocial health.
Background:
(1) Recruitment of tumor supporting stromal cells and tissue remodeling in the tumor microenvironment support cancer cell proliferation, invasion, metastasis, and drug resistance.
(2) Mesenchymal stem cells (MSC) are recruited by cancer cells into the tumor site and play a role in modulating tumor progression.
(3) Intratumoral heterogeneity exists in solid tumors, with cancer stem cells (CSCs) and clonal evolution contributing to the complexity of cancer.
Hypothesis:
(1) Transcriptional regulators are responsible for tumor-supporting stromal reprogramming, specifically in MSC in the tumor stroma.
(2) Intercellular communication between cancer cells and recruited MSCs is mediated by cell-to-cell contact, paracrine interactions, and microvesicles.
(3) Epithelial cancer cell plasticity is regulated by tumor stroma interaction signals, enabling non-CSCs to convert into CSCs.
Background:
(1) Transitions in care are complex and require coordination and communication among different healthcare providers.
(2) The experiences of two different patients during care transitions were significantly different.
(3) Major gaps in care occur during client handoffs, leading to misunderstandings, errors, and negative outcomes.
Hypothesis:
(1) Differences in care transitions may be attributed to unique client needs.
(2) Differences in the way healthcare providers respond to client needs may contribute to varied experiences.
(3) Existing regulatory standards may not adequately address safety issues in care transitions.
Background:
(1) Fatty acid species with a maximum chain length of 16-18 carbon atoms account for >90% of total fatty acids in most mammalian tissues.
(2) Very long chain fatty acids (VLCFA) consisting of 20 and more carbon atoms are found at high levels in the brain, skin, testis, and some glands.
(3) VLCFA are mainly esterified in various lipids, particularly ceramide in sphingolipids.
Hypothesis:
(1) Substrate specificity of fatty acyltransferases determines the distribution bias of VLCFA between sphingolipids and glycerolipids.
(2) Sphingolipids, containing VLCFA, have essential roles in cell proliferation, epidermal water barrier, myelin function, cell recognition, and adhesion.
(3) Multiple microsomal elongation systems might exist in cells with different saturation and chain length specificities for VLCFA synthesis.
Background:
(1) Research participants are interested in receiving study results.
(2) Research results are seldom communicated to participants, including those who participate in community-based participatory research (CBPR).
(3) Few studies have explored researchers' experiences, attitudes and barriers related to sharing study results with participants.
Hypothesis:
(1) Health researchers have varying opinions and experiences related to sharing research results with participants.
(2) Barriers to sharing results with participants include financial, ethical, logistical, methodological, and systems-related factors.
(3) Researchers express support for sharing scientific results with research participants but often do not currently have a plan for results sharing.
Background: {background}
Hypothesis:
"""

        prompt = prompt.format(background=background)

        # Create reference with ground truth hypothesis
        references = [Reference(output=Output(text=hypothesis), tags=[])]

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=TEST_SPLIT,
        )
