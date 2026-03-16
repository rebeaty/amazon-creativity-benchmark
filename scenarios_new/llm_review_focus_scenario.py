"""
HELM Scenario: LLM Review Focus (Mind the Blind Spots)

Paper: Mind the Blind Spots: A Focus-Level Evaluation Framework for LLM Reviews
       https://arxiv.org/abs/2502.17086
       Shin, H., et al. (2025). EMNLP 2025.

Dataset: https://figshare.com/s/d5adf26c802527dd0f62
         676 papers from ICLR (2021-2024) with expert reviews
         3,657 annotated strengths/weaknesses from human experts
         43,042 strengths/weaknesses from 8 LLMs

Task: Generate peer reviews for academic papers (ICLR submissions).
      Reviews should identify strengths and weaknesses across multiple facets.

Prompt format: Exact prompt from paper Appendix A.1.2 (Figure 8)
  [[ Paper Content ]]
  {paper_content}
  [[ Instruction ]]
  Review the given paper for a top AI conference...
  [[ Your Response ]]
  # Summary of paper
  # Strengths
  # Weaknesses
  # Final Judgement

  Input: Paper content (title, abstract, full text combined)
  Output: Structured review with summary, strengths, weaknesses, and final judgement

Facets evaluated (Focus-Level Framework):

  Target Facets (7):
  - Problem: Research problem formulation
  - Prior Research: Related work and positioning
  - Method: Proposed approach/algorithm
  - Theory: Theoretical foundations
  - Experiment: Experimental setup and results
  - Conclusion: Findings and implications
  - Paper: Overall presentation and writing

  Aspect Facets (5):
  - Impact: Significance and potential contribution
  - Novelty: Originality and innovation
  - Clarity: Presentation and understandability
  - Validity: Technical correctness and soundness
  - Not-specific: General comments

Evaluation: Custom focus-level metrics (see metric_notes.md)
  - Focus distribution: KL divergence from human expert distribution
  - Facet coverage: F1 score for target/aspect identification
  - Content quality: ROUGE-L, BERTScore for text quality

  Key finding from paper: LLMs show "biased focus towards technical validity
  while significantly overlooking novelty assessment"

Data splits:
  - Train: 582 papers
  - Test: 98 papers
  - Total: 680 (676 after excluding 5 during tokenization)

Fields used: paper_id, title, abstract, content (or similar)
Fields used for reference: expert_review, strengths, weaknesses, target_labels, aspect_labels

Notes:
  - Dataset requires manual download from Figshare
  - Papers are rejected ICLR submissions (2021-2024)
  - Expert reviews extracted from OpenReview meta-reviews
  - 8 LLMs evaluated: GPT-3.5/4/4-turbo/4o, Llama-3-70B/3.1-70B, DeepSeek-V2.5/R1
  - Best model (GPT-4o) achieved F1=0.373 for facet matching
"""

import os
import json
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)


class LLMReviewFocusScenario(Scenario):
    """
    LLM Review Focus: Academic peer review generation for ICLR papers.

    Evaluates whether LLMs attend to the same critical facets that human expert
    reviewers prioritize when evaluating research papers.
    """

    name = "llm_review_focus"
    description = "Mind the Blind Spots - Focus-Level Evaluation for LLM Reviews (Figshare)"
    tags = ["review_generation", "academic_writing", "analytical_writing"]

    # Exact prompt template from paper Appendix A.1.2 (Figure 8)
    REVIEW_PROMPT_TEMPLATE = """[[ Paper Content ]]
{paper_content}

[[ Instruction ]]
Review the given paper for a top AI conference. Please be critical, focused, and constructive so that the authors find the review convincing and improve their manuscript accordingly. Please write a review that includes:
1. Summary of paper
2. Strengths
3. Weaknesses
4. Final Judgement

[[ Your Response ]]
# Summary of paper
# Strengths
 - **Strength header**:
 - **Strength header**:
 - **Strength header**:
 ...
# Weaknesses
 - **Weakness header**:
 - **Weakness header**:
 - **Weakness header**:
 ...
# Final Judgement
 - **Rationale of recommendation**:
 - **Recommendation**: (either "Accept" or "Reject")"""

    def __init__(self, data_dir: str = "data/llm_review_focus"):
        """
        Initialize the scenario.

        Args:
            data_dir: Directory containing the downloaded Figshare dataset
        """
        self.data_dir = data_dir

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate instances for review generation evaluation.

        Note: This scenario requires manual dataset download from:
        https://figshare.com/s/d5adf26c802527dd0f62

        Expected data format (JSON or similar):
        {
            "paper_id": str,
            "title": str,
            "abstract": str,
            "content": str (optional full paper text),
            "expert_review": str,
            "strengths": List[str],
            "weaknesses": List[str],
            "target_labels": List[str],  # facet annotations
            "aspect_labels": List[str],   # facet annotations
            "split": "train" or "test"
        }

        Returns:
            List of Instance objects for review generation.
        """

        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir}. "
                f"Please download from https://figshare.com/s/d5adf26c802527dd0f62"
            )

        instances = []

        # Load dataset (format TBD based on actual Figshare data structure)
        # This is a placeholder implementation
        data_file = os.path.join(self.data_dir, "reviews.json")

        if not os.path.exists(data_file):
            # Provide helpful error message
            raise FileNotFoundError(
                f"Expected data file not found: {data_file}\n"
                f"Please download the dataset from Figshare and place it in {self.data_dir}/"
            )

        with open(data_file, 'r') as f:
            data = json.load(f)

        for item in data:
            # Extract paper information
            paper_id = item.get("paper_id", "")
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            full_paper = item.get("content", "")  # May be empty if not included

            # Determine split
            split = TRAIN_SPLIT if item.get("split") == "train" else TEST_SPLIT

            # Construct paper content (combining title, abstract, and full paper)
            paper_content_parts = []
            if title:
                paper_content_parts.append(f"Title: {title}")
            if abstract:
                paper_content_parts.append(f"\nAbstract:\n{abstract}")
            if full_paper:
                paper_content_parts.append(f"\n\nFull Paper:\n{full_paper}")

            paper_content = "\n".join(paper_content_parts)

            # Create prompt using exact format from paper
            prompt = self.REVIEW_PROMPT_TEMPLATE.format(paper_content=paper_content)

            # Expert review as reference
            expert_review = item.get("expert_review", "")

            # Create references with expert review
            references = []
            if expert_review:
                references.append(
                    Reference(
                        output=Output(text=expert_review),
                        tags=[CORRECT_TAG]
                    )
                )

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=split,
                )
            )

        return instances
