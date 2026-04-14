"""
HELM Scenario: GAUSS (General Assessment of Underlying Structured Skills in Mathematics)

Paper: GAUSS: Benchmarking Structured Mathematical Skills for Large Language Models
       https://arxiv.org/abs/2509.18122

Website: https://gaussmath.ai/
Dataset: https://huggingface.co/datasets/GaussMath/GAUSS
License: Not specified

Task: Advanced Mathematical Problem Solving
Evaluate LLMs across 12 structured skill dimensions in mathematics, including creativity.

12 Skill Dimensions (from paper):
1. Memory of Math Knowledge
2. Understanding of Knowledge and Theories
3. Computational and Analytical Skills
4. Problem-Solving Framework
5. Logical Thinking and Reasoning
6. Writing and Presentation
7. Learning New Knowledge
8. Intuition
9. Meta Skills
10. Mathematical Modeling
11. Generalization
12. Creativity

Dataset contains 41 graduate/research-level mathematics problems with:
- Problem statements
- Standard solutions (expert-written)
- Rubrics (scoring criteria)
- Total scores

Creativity dimension (category 12) includes 3 problems:
- 12a: Massive SLE - Define and explore properties (open-ended)
- 12b: 1977 IMO Problem 2 - Find multiple solutions (creative problem-solving)
- 12c: Move one digit puzzle - Creative mathematical puzzle

Fields used: problem_name, problem_statement, category, standard_solution, rubric, total_score
Fields skipped: problem_attachment (mostly empty), model_name/model_response/model_score/evaluation
               (GPT-5-Thinking outputs, not ground truth), contributor_name/email (metadata)

Evaluation: Open-ended generation (BLEU, ROUGE, F1 against standard solutions)
Alternative: LLM-as-judge using rubrics (see metric_notes.md)
"""

from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)
from datasets import load_dataset


class GAUSSScenario(Scenario):
    """
    GAUSS: General Assessment of Underlying Structured Skills in Mathematics

    Evaluates mathematical problem-solving across 12 skill dimensions,
    with option to focus on creativity dimension.
    """

    name = "gauss"
    description = "GaussMath/GAUSS"
    tags = ["creativity", "mathematics", "problem_solving"]

    # Dimension names from paper
    DIMENSIONS = {
        "1": "Memory of Math Knowledge",
        "2": "Understanding of Knowledge and Theories",
        "3": "Computational and Analytical Skills",
        "4": "Problem-Solving Framework",
        "5": "Logical Thinking and Reasoning",
        "6": "Writing and Presentation",
        "7": "Learning New Knowledge",
        "8": "Intuition",
        "9": "Meta Skills",
        "10": "Mathematical Modeling",
        "11": "Generalization",
        "12": "Creativity"
    }

    def __init__(self, dimension: str = "12"):
        """
        Args:
            dimension: Which dimension(s) to include
                - "12" (default): Creativity only (3 problems)
                - "all": All 12 dimensions (41 problems)
                - "1"-"11": Specific dimension
        """
        super().__init__()
        self.dimension = dimension

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load GAUSS dataset and create instances.

        Each instance contains:
        - Problem statement
        - Reference: Standard solution with rubric
        """
        # Load dataset from HuggingFace
        dataset = load_dataset("GaussMath/GAUSS", split="train")

        instances = []
        for idx, problem in enumerate(dataset):
            # Extract dimension from category (first digit(s))
            category = problem["category"]
            dim = category.rstrip('abcd')

            # Filter by dimension if specified
            if self.dimension != "all" and dim != self.dimension:
                continue

            # Build prompt from problem statement
            problem_statement = problem["problem_statement"].strip()
            prompt = f"{problem_statement}"

            # Reference answer includes standard solution
            # The rubric provides scoring criteria (useful for LLM-as-judge)
            standard_solution = problem["standard_solution"].strip()
            rubric = problem["rubric"].strip()

            # Create reference with standard solution
            # Store rubric and scoring info in extra_data for potential judge use
            references = [
                Reference(
                    Output(text=standard_solution),
                    tags=[CORRECT_TAG]
                )
            ]

            # Create instance
            dimension_name = self.DIMENSIONS.get(dim, f"Dimension {dim}")
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"gauss_{category}_{problem['problem_name'].replace(' ', '_')}",
                    # Store rubric and scoring for potential LLM-as-judge evaluation
                    extra_data={
                        "dimension": dim,
                        "dimension_name": dimension_name,
                        "category": category,
                        "problem_name": problem["problem_name"],
                        "rubric": rubric,
                        "total_score": problem["total_score"],
                    }
                )
            )

        return instances
