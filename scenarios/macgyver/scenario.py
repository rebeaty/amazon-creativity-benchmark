"""
HELM Scenario: MACGYVER

Paper: https://arxiv.org/abs/2311.09682 (NAACL 2024)
Code: https://github.com/allenai/MacGyver

MacGyver is a benchmark of 1,600+ real-world verbal problems designed to test
creative problem-solving and innovative usage of objects. Problems require
out-of-the-box thinking to solve using only the provided tools.

Prompt strategies (from paper Appendix D.3, Figures 15-16):

  1. "vanilla" (default): Single-round basic prompt
     Source: code/collect_solutions/collect_GPT4_solutions.py

  2. "divergent_convergent": Analyze tool affordances before solving
     Source: code/improving_LLM_via_prompting/improve_prompts.json (instruction_div_conv)
     Paper: Figure 15, Appendix D.3

  3. "reflection": Multi-round iterative verification of physical feasibility
     Source: code/improving_LLM_via_prompting/improve_prompts.json (instructions_reflect)
     Paper: Figure 16, Appendix D.3
     Note: This is a 2-turn conversation; Round 1 generates solution, Round 2 verifies

Prompt source: Paper Appendix D.3, Figures 15-16; code/improving_LLM_via_prompting/improve_prompts.json
Fields used: Problem, Solution, Solvable?, Unconventional?
Fields skipped: ID, Label (used for human annotation of model outputs)

Subsets:
  - "all": All 1,683 problems (default)
  - "solvable": Only solvable problems (1,306)
  - "unsolvable": Only unsolvable problems (377)
  - "unconventional": Solvable problems requiring unconventional tool use (956)

Evaluation: open_ended (BLEU/ROUGE) or llm_judge for correctness assessment.
Paper uses human annotation with categories: efficient, inefficient, infeasible,
correct_right_reason, correct_wrong_reason, wrong_solution.
"""

import os
import urllib.request
import pandas as pd
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class MacgyverScenario(Scenario):
    name = "macgyver"
    description = "allenai/MacGyver"
    tags = ["creativity", "problem_solving"]

    SUBSETS = ["all", "solvable", "unsolvable", "unconventional"]
    PROMPT_STRATEGIES = ["vanilla", "divergent_convergent", "reflection"]
    DATA_URL = "https://github.com/allenai/MacGyver/blob/main/data/MacGyver/problem_solution_pair.xlsx?raw=true"

    # Exact prompts from paper (Appendix D.3, Figures 15-16)
    # Source: code/collect_solutions/collect_GPT4_solutions.py
    INSTRUCTION_VANILLA = (
        "Give a valid (feasible and efficient) solution very concisely. "
        "Use step1, step2, etc, and mention the tools to achieve each step. "
        "Use as few steps as possible and the answer should ideally be less than 100 words. "
        "When there is not a feasible solution given the constraint and provided tools, "
        "just say that it is not possible and give a very short justification."
    )

    # Source: code/improving_LLM_via_prompting/improve_prompts.json (instruction_div_conv)
    # Paper: Figure 15, Appendix D.3 - Divergent-Convergent Thinking
    INSTRUCTION_DIVERGENT_CONVERGENT = (
        "Give a feasible solution very concisely. Note that some tools are not useful, "
        "so please analyze the affordance of each presented object, and rule out unnecessary ones first.\n\n"
        "Use the following format:\n"
        "1. List the affordance of presented items and whether they are useful\n"
        "2. Summary: list useful tools\n"
        "3. If the problem is solvable under all these constraints, write the solution. "
        "Use step1, step2, etc, and mention the tools to achieve each step. "
        "Use as few steps as possible and the answer should ideally be less than 100 words.\n"
        "   If you cannot find a feasible solution, just say that it is not possible "
        "and give a very short justification."
    )

    # Source: code/improving_LLM_via_prompting/improve_prompts.json (instructions_reflect)
    # Paper: Figure 16, Appendix D.3 - Iterative Step-Wise Reflection (Round 2)
    INSTRUCTION_REFLECTION_ROUND2 = (
        "Now, please verify if each step is physically feasible and afforded. "
        "After that, modify the solution if needed.\n"
        "Use the following format:\n"
        "Step 1: ...\n"
        "Step 2: ...\n"
        "...\n"
        "Conclusion 1: Whether the problem is indeed solvable given all the constraints\n"
        "Conclusion 2: (If still solvable) No modification needed/Modification needed.\n\n"
        "Modified solution:"
    )

    def __init__(self, subset: str = "all", prompt_strategy: str = "vanilla"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"subset must be one of {self.SUBSETS}, got '{subset}'")
        if prompt_strategy not in self.PROMPT_STRATEGIES:
            raise ValueError(f"prompt_strategy must be one of {self.PROMPT_STRATEGIES}, got '{prompt_strategy}'")
        self.subset = subset
        self.prompt_strategy = prompt_strategy

    def _download_data(self, output_path: str) -> str:
        """Download the dataset xlsx file if not cached."""
        filepath = os.path.join(output_path, "macgyver_problems.xlsx")
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(self.DATA_URL, filepath)
        return filepath

    def _build_prompt(self, problem: str) -> str:
        """Build the prompt based on the selected strategy."""
        if self.prompt_strategy == "vanilla":
            return f"{problem}\n\n{self.INSTRUCTION_VANILLA}"

        elif self.prompt_strategy == "divergent_convergent":
            return f"{problem}\n\n{self.INSTRUCTION_DIVERGENT_CONVERGENT}"

        elif self.prompt_strategy == "reflection":
            # Multi-round: Include both Round 1 and Round 2 instructions
            # Round 1: Generate initial solution
            # Round 2: Verify and modify
            return (
                f"{problem}\n\n"
                f"{self.INSTRUCTION_VANILLA}\n\n"
                f"[After generating your initial solution, verify it:]\n\n"
                f"{self.INSTRUCTION_REFLECTION_ROUND2}"
            )

        return f"{problem}\n\n{self.INSTRUCTION_VANILLA}"

    def get_instances(self, output_path: str):
        filepath = self._download_data(output_path)
        df = pd.read_excel(filepath)

        # Filter by subset
        if self.subset == "solvable":
            df = df[df["Solvable?"] == "Yes"]
        elif self.subset == "unsolvable":
            df = df[df["Solvable?"] == "No"]
        elif self.subset == "unconventional":
            df = df[(df["Solvable?"] == "Yes") & (df["Unconventional?"] == "unconventional")]

        instances = []
        for _, row in df.iterrows():
            prompt = self._build_prompt(row['Problem'])

            # Gold solution as reference (clean up <br> tags)
            solution = str(row["Solution"]).replace("<br>", "").replace("\n", " ").strip()

            references = [
                Reference(Output(text=solution), tags=[CORRECT_TAG])
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
