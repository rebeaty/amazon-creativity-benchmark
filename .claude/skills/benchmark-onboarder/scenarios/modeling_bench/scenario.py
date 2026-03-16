"""
HELM Scenario: ModelingBench

Paper: ModelingAgent: Reformulating Mathematical Modeling as a Code-Driven Agentic Competition
       arXiv:2505.15068 (May 2025)
Code:  https://github.com/qiancheng0/ModelingAgent
Dataset: https://github.com/qiancheng0/ModelingAgent/blob/main/data/modeling_data_final.json
License: Not specified

Task: Mathematical Modeling Report Generation
Evaluate LLMs on their ability to write comprehensive mathematical modeling reports
for competition-style problems from COMAP (HiMCM, MCM, ICM).

Dataset: 11 problems from 2001–2003 COMAP competitions across:
- HiMCM (High School Mathematical Contest in Modeling): 5 problems
- MCM (Mathematical Contest in Modeling): 5 problems
- ICM (Interdisciplinary Contest in Modeling): 1 problem
- Domains: public health, infrastructure, transportation, environmental science, sports

Prompt format: From src/ModelBase/baseline.py (vanilla baseline used in paper).
  System: Multi-criteria expert modeler instruction with mandatory output structure
          (Problem Restatement, Assumptions, Model Development, Solution Process,
           Results and Analysis, Recommendations)
  User:   "Please create a comprehensive mathematical modeling solution for the
           following problem:\n\n{question}\n\nDevelop a complete solution
           following the specified structure."

  Note: The baseline prompt uses only {question}. Requirements are NOT injected into
        the user prompt — they are used solely by the judge (via extra_data).

Fields used:    question (HTML img tags stripped)
Fields skipped: requirements (judge-only, stored in extra_data),
                link (relative local path, not a usable URL),
                decomposition (mirrors requirements, used internally by judge)

Evaluation: LLM-as-judge (ModelingJudge) with 6 dimensions across multiple expert roles.
            See annotator_notes.md for full judge configuration.
"""

import json
import os
import re
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    Input,
    TEST_SPLIT,
)
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class ModelingBenchScenario(Scenario):
    """ModelingBench: COMAP mathematical modeling competition benchmark.

    Models are given a competition-style problem (scenario description + graded
    requirements) and asked to produce a comprehensive mathematical modeling report.

    11 problems total. Evaluation uses ModelingJudge with 6 dimensions:
    scoring_decomposition, structural_coherency, modeling_groundedness,
    data_groundedness, analysis_groundedness, and innovativeness.
    See annotator_notes.md for judge setup.
    """

    name = "modeling_bench"
    description = "qiancheng0/ModelingAgent (data/modeling_data_final.json)"
    tags = ["creativity", "mathematical_modeling", "long_form_generation", "problem_solving"]

    DATASET_URL = (
        "https://raw.githubusercontent.com/qiancheng0/ModelingAgent"
        "/main/data/modeling_data_final.json"
    )

    # Verbatim from src/ModelBase/baseline.py (SYS_PROMPT + USER_PROMPT)
    SYS_PROMPT = (
        "You are an expert mathematical modeler tasked with creating comprehensive solutions "
        "to mathematical modeling problems. Your solutions must be of high quality and meet "
        "the following criteria:\n\n"
        "1. Structural Completeness:\n"
        "   - Clear problem restatement showing deep understanding\n"
        "   - Well-justified assumptions with rationale\n"
        "   - Detailed model implementation with mathematical rigor\n"
        "   - Clear solution process and results presentation\n"
        "   - Thorough analysis of results and limitations\n\n"
        "2. Problem Requirements:\n"
        "   - Address every requirement stated in the problem\n"
        "   - Ensure each component of the solution aligns with problem objectives\n"
        "   - Follow any specific format or deliverable requirements\n\n"
        "3. Modeling Quality:\n"
        "   - Use appropriate modeling approaches for the problem context\n"
        "   - Consider real-world factors and constraints\n"
        "   - Employ rigorous mathematical formalization\n"
        "   - Clearly state and justify model parameters\n"
        "   - Include validation methods\n\n"
        "4. Data Handling:\n"
        "   - Use authentic and reliable data sources\n"
        "   - Justify data selection and preprocessing\n"
        "   - Ensure sufficient data for meaningful analysis\n"
        "   - Include data validation and quality checks\n\n"
        "5. Analysis Depth:\n"
        "   - Base conclusions on mathematical/experimental evidence\n"
        "   - Provide insightful interpretation of results\n"
        "   - Include sensitivity analysis where appropriate\n"
        "   - Discuss limitations and uncertainties\n\n"
        "6. Innovation:\n"
        "   - Propose creative modeling approaches\n"
        "   - Consider novel combinations of methods\n"
        "   - Demonstrate potential real-world impact\n"
        "   - Suggest practical implementation strategies\n\n"
        "Your solution must follow this structure:\n\n"
        "### Problem Restatement\n"
        "[Clear restatement and interpretation of the problem]\n\n"
        "### Assumptions and Justification\n"
        "[List and justify key assumptions]\n\n"
        "### Model Development\n"
        "[Detailed mathematical model description]\n"
        "- Variables and Parameters\n"
        "- Equations and Relationships\n"
        "- Constraints and Conditions\n\n"
        "### Solution Process\n"
        "[Step-by-step solution implementation]\n"
        "- Data Collection and Processing\n"
        "- Model Implementation\n"
        "- Solution Methods\n\n"
        "### Results and Analysis\n"
        "[Comprehensive results presentation]\n"
        "- Key Findings\n"
        "- Sensitivity Analysis\n"
        "- Validation\n"
        "- Limitations\n\n"
        "### Recommendations\n"
        "[Practical implications and suggestions]\n\n"
        "Note: Ensure mathematical rigor while maintaining clarity. "
        "Include equations, diagrams, and data analysis as needed."
    )

    USER_PROMPT = (
        "Please create a comprehensive mathematical modeling solution for the following problem:\n\n"
        "{question}\n\n"
        "Develop a complete solution following the specified structure."
    )

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load ModelingBench and create HELM instances."""
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)

        data_path = os.path.join(data_dir, "modeling_data_final.json")
        ensure_file_downloaded(
            source_url=self.DATASET_URL,
            target_path=data_path,
        )

        with open(data_path, "r", encoding="utf-8") as f:
            problems = json.load(f)

        instances = []
        for problem_id, item in problems.items():
            # Strip <img>...</img> tags — images are local files not accessible via download
            question_text = re.sub(r"<img[^>]*>.*?</img>", "", item["question"], flags=re.DOTALL)
            question_text = question_text.strip()

            # User prompt verbatim from src/ModelBase/baseline.py
            # System prompt is stored as SYS_PROMPT class attribute; HELM passes it separately
            prompt = self.USER_PROMPT.format(question=question_text)

            # Open-ended generation — no gold reference report available
            references: List[Reference] = []

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    id=problem_id,
                    split=TEST_SPLIT,
                    extra_data={
                        "title": item["title"],
                        "year": item["year"],
                        "level": item["level"],
                        "source": item["source"],
                        "requirements": item["requirements"],
                        "eval_roles": item["eval_roles"],
                        "system_prompt": self.SYS_PROMPT,
                    },
                )
            )

        return instances


if __name__ == "__main__":
    scenario = ModelingBenchScenario()
    instances = scenario.get_instances("/tmp/modeling_bench_test")
    print(f"Loaded {len(instances)} instances")
    if instances:
        ex = instances[0]
        print(f"\nID: {ex.id}")
        print(f"Title: {ex.extra_data['title']} ({ex.extra_data['year']}, {ex.extra_data['source']})")
        print(f"Prompt preview:\n{ex.input.text[:400]}...")
        print(f"Requirements count: {len(ex.extra_data['requirements'])}")
        print(f"Eval roles count: {len(ex.extra_data['eval_roles'])}")
