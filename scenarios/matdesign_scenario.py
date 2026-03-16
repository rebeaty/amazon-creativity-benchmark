"""
HELM Scenario: MATDESIGN - Materials Discovery and Design via Goal-Driven LLM Agents

Paper: Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents
       https://arxiv.org/abs/2501.13299
Code: https://github.com/shri071/Hypothesis-Generation-for-Materials-Discovery-and-Design-Using-Goal-Driven-and-Constraint-Guided-LLM

Task: Generate innovative materials and methods suggestions given a goal statement and constraints.
      Models must propose 20 suggestions, each with materials, methods, and reasoning that satisfy
      the goal and all constraints.

Dataset: 50 examples from materials science research papers published in 2024
- Each example contains:
  * Goal Statement: Materials science objective
  * Constraints: 4-6 numbered requirements
  * Reference Materials/Methods: Ground truth from the paper (for reference only)

Prompt format: Direct goal statement + constraint list + request for 20 JSON-formatted suggestions
  {goal_statement}

  Constraints:-
  {constraint_list}.

  Provide me 20 innovative suggestions that will help achieve the above goal while satisfying
  all of the above mentioned constraints strictly. Provide reason for each suggestion.

Evaluation: llm_judge (multi-agent)
  - Multi-agent critic system (GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Flash)
  - Each suggestion evaluated on: Meets_the_goal_statement_and_satisfies_all_constraints_strictly (YES/NO)
  - Detailed reasoning provided for each evaluation
  - Iterative refinement until all suggestions meet criteria
  - Final evaluation by OpenAI-o1-preview
  See annotator_notes.md for complete evaluation setup

Note: Reference materials/methods from papers are provided for context but models should
      generate NOVEL suggestions, not reproduce the paper's approach.
"""

import json
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    TEST_SPLIT,
)


class MatDesignScenario(Scenario):
    """MATDESIGN: Materials Discovery and Design via Goal-Driven LLM Agents

    Evaluates models' ability to generate innovative materials and methods suggestions
    that satisfy complex constraints for materials science applications.
    """

    name = "matdesign"
    description = "shri071/Hypothesis-Generation-Materials-Discovery"
    tags = ["creativity", "scientific_reasoning", "materials_science", "hypothesis_generation"]

    # GitHub raw URL for the dataset
    # Note: The original dataset is in Excel format. This scenario assumes a JSON conversion.
    # For production use, the Excel file should be converted to JSON and hosted.
    BASE_URL = "https://raw.githubusercontent.com/shri071/Hypothesis-Generation-for-Materials-Discovery-and-Design-Using-Goal-Driven-and-Constraint-Guided-LLM/main"

    def __init__(self):
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        # NOTE: The original dataset is in Excel format (Materials Discovery & Design Dataset.xlsx)
        # This scenario uses a JSON conversion of that file.
        # The JSON file should be hosted on GitHub or another stable URL for production use.

        # For now, we'll use a GitHub gist or raw URL approach
        # In the actual implementation, convert the Excel file to JSON and host it

        import os

        # Try to load from local converted file first (for development)
        local_json_path = "/tmp/matdesign_data.json"

        instances = []

        if os.path.exists(local_json_path):
            # Load from local converted JSON
            with open(local_json_path, 'r') as f:
                data = json.load(f)
        else:
            # Fallback to minimal hardcoded examples for testing
            data = [
                {
                    "Title": "Rapid Self-Healing Hydrogel with Ultralow Electrical Hysteresis for Wearable Sensing",
                    "Goal Statement": "A self-healing hydrogel that exhibits exceptionally rapid healing. The hydrogel should have an ideal balance between properties such as softness, deformability, ionic and electrical conductivity, self-adhesiveness, response and recovery times, durability, overshoot behavior, and resistance to nonaxial deformations such as twisting, bending, and pressing",
                    "Constraints": "1) The material must exhibit rapid self-healing, with a recovery time of less than 0.12 seconds, to ensure timely restoration of both mechanical and electrical functions.\n2) The hydrogel must possess ultralow electrical hysteresis (less than 0.64%) under cyclic strains up to 500%, ensuring minimal energy dissipation during repetitive movements.\n3) The material should be highly deformable, with the ability to stretch over 10,000%, while maintaining mechanical integrity in complex, nonaxial deformations such as twisting, bending, and pressing.\n4) The hydrogel must have high ionic and electrical conductivity (greater than 0.074 S m−1) and exhibit strong self-adhesiveness to human skin for effective use in wearable applications.\n5) The material must maintain durability and functionality over long-term use, suitable for monitoring physiological activities such as facial expressions, joint movements, and electrophysiological signals (ECG, EMG, EOG).",
                    "Materials": "Poly(3,4-ethylenedioxythiophene)/polystyrenesulfonate (PEDOT/PSS)\nPoly(vinyl alcohol) (PVA)\nBorax\nGlycerol",
                    "Methods": "Synthesis of the Self-Healing Hydrogel: The hydrogel was synthesized using a dynamic network of poly(3,4-ethylenedioxythiophene)/polystyrene sulfonate (PEDOT/PSS), poly(vinyl alcohol) (PVA), borax, and glycerol. This combination enhances both ionic and electrical conductivity, while the multiple hydroxyl groups in glycerol provide numerous healing sites, facilitating rapid self-repair."
                }
            ]

        for item in data:
            # Skip items without required fields
            if not item.get("Goal Statement") or not item.get("Constraints"):
                continue
            instances.append(self._create_instance(item))

        return instances

    def _create_instance(self, item: dict) -> Instance:
        """Create an instance from a materials discovery problem"""

        goal_statement = item["Goal Statement"]
        constraint_list = item["Constraints"]
        materials = item.get("Materials", "")
        methods = item.get("Methods", "")

        # Build prompt following the EXACT format from agent_framework_materials_discovery.py (lines 14-25)
        # Note: Trailing spaces after "strictly." and after colons are preserved from original
        prompt = f"""{goal_statement} \n\n Constraints:- \n{constraint_list}.\n
Provide me 20 innovative suggestions that will help achieve the above goal while satisfying all of the above mentioned constraints strictly. 
Provide reason for each suggestion. The suggestions must be in the below mentioned format in a JSON object. For example:\n
{{Suggestion_1: 
    Materials: 
    Methods_to_develop_the_materials_suggested: 
    Reasoning:
    ,
Suggestion_20: 
    Materials: 
    Methods_to_develop_the_materials_suggested: 
    Reasoning: }}"""

        # Create reference with ground truth (for context, not direct comparison)
        # Note: The ground truth represents ONE working solution from the paper,
        # but the model should generate NOVEL suggestions
        reference_text = f"Reference Materials: {materials}\n\nReference Methods: {methods}"
        references = [Reference(output={"text": reference_text}, tags=["reference_only"])]

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=TEST_SPLIT,
        )
