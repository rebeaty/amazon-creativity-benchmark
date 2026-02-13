"""
HELM Scenario: IDRBench (Interactive Deep Research Benchmark) - Adapted

Paper: https://arxiv.org/abs/2601.06676
Code: https://github.com/ServiceNow/drbench
Published: January 2025

ADAPTATION NOTE: This is a simplified single-turn version of IDRBench,
which was originally designed for interactive multi-turn research.

Task: Generate comprehensive research reports answering business questions
by synthesizing information from multiple supporting documents.

This tests analytical reasoning, multi-document synthesis, and research
capabilities rather than pure creativity.

Prompt format:
  You are {persona} at {company}. Review the following documents and
  answer the research question:

  Research Question: {dr_question}

  Context:
  - Industry: {industry}
  - Domain: {domain}

  Documents:
  [Document {doc_id}]: {question}
  Answer: {answer}

  Generate a comprehensive research report that:
  1. Directly addresses the research question
  2. Synthesizes information from the provided documents
  3. Provides actionable insights
  4. Cites specific documents to support claims

  Research Report:

Evaluation: LLM-as-judge (GPT-4) evaluating:
  - Accuracy: Factual correctness based on provided documents
  - Completeness: Addresses all aspects of the question
  - Coherence: Logical organization and flow
  - Citations: Proper attribution of sources

See annotator_notes.md for detailed evaluation setup.

Fields used: dr_question, persona, company, industry, domain, supporting facts
Fields skipped: distractor facts (only use supporting facts), url, url_date

Dataset: 15 deep research questions
  - Industries: retail (5), healthcare (5), automobiles (5)
  - Domains: compliance, sales, CRM, marketing, cybersecurity, etc.
  - Supporting facts: 3-16 per question (avg 7.6)
"""

import os
import pandas as pd
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
)


class IDRBenchScenario(Scenario):
    """
    IDRBench: Deep Research Benchmark (Adapted for Single-Turn Generation)

    Tests analytical reasoning and multi-document synthesis through
    business research report generation.
    """

    name = "idrbench"
    description = "ServiceNow/drbench"
    tags = ["reasoning", "research", "synthesis", "analytical"]

    def __init__(self, include_distractors: bool = False):
        """
        Args:
            include_distractors: If True, include distractor facts in prompts.
                Default False (only supporting facts for clearer evaluation).
        """
        super().__init__()
        self.include_distractors = include_distractors

    def _download_data(self, output_path: str) -> str:
        """
        Download IDRBench dataset from GitHub if not already present.

        Returns:
            Path to the data directory
        """
        data_dir = os.path.join(output_path, "idrbench_data")

        # Check if data already exists
        if os.path.exists(data_dir) and os.path.exists(
            os.path.join(data_dir, "drbench", "data", "summary", "dr_questions.csv")
        ):
            print(f"Data already exists at {data_dir}")
            return data_dir

        # Clone the repository
        import subprocess
        print("Downloading IDRBench dataset from GitHub...")
        os.makedirs(output_path, exist_ok=True)

        repo_url = "https://github.com/ServiceNow/drbench.git"
        subprocess.run(
            ["git", "clone", repo_url, data_dir],
            check=True,
            capture_output=True
        )
        print("Download complete")

        return data_dir

    def _load_supporting_facts(self, data_dir: str, task_id: str) -> List[dict]:
        """
        Load supporting facts for a given task.

        Args:
            data_dir: Path to data directory
            task_id: Task ID (e.g., "DR0001")

        Returns:
            List of fact dictionaries
        """
        facts_path = os.path.join(
            data_dir, "drbench", "data", "summary", "facts", f"{task_id}_facts.csv"
        )

        facts_df = pd.read_csv(facts_path)

        # Filter for supporting facts (or include distractors if specified)
        if self.include_distractors:
            filtered_facts = facts_df
        else:
            filtered_facts = facts_df[facts_df['type'] == 'supporting']

        return filtered_facts.to_dict('records')

    def _format_prompt(
        self,
        dr_question: str,
        persona: str,
        company: str,
        industry: str,
        domain: str,
        facts: List[dict]
    ) -> str:
        """
        Format the research report generation prompt.

        Args:
            dr_question: Research question
            persona: Role/persona (e.g., "Emily Patel")
            company: Company name
            industry: Industry sector
            domain: Business domain
            facts: List of supporting facts

        Returns:
            Formatted prompt string
        """
        prompt = (
            f"You are {persona} at {company}. Review the following documents and "
            f"answer the research question:\n\n"
            f"Research Question: {dr_question}\n\n"
            f"Context:\n"
            f"- Industry: {industry}\n"
            f"- Domain: {domain}\n\n"
            f"Documents:\n"
        )

        # Add each fact as a document
        for fact in facts:
            prompt += f"\n[Document {fact['fact_id']}]:\n"
            prompt += f"Q: {fact['question']}\n"
            prompt += f"A: {fact['answer']}\n"

        prompt += (
            f"\nGenerate a comprehensive research report that:\n"
            f"1. Directly addresses the research question\n"
            f"2. Synthesizes information from the provided documents\n"
            f"3. Provides actionable insights\n"
            f"4. Cites specific documents to support claims\n\n"
            f"Research Report:"
        )

        return prompt

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate IDRBench instances.

        Each instance contains:
        - Input: Research question with supporting documents
        - References: Empty (will use LLM-as-judge evaluation)
        """
        # Download data if needed
        data_dir = self._download_data(output_path)

        # Load questions
        questions_path = os.path.join(
            data_dir, "drbench", "data", "summary", "dr_questions.csv"
        )
        questions_df = pd.read_csv(questions_path)

        instances = []
        for _, row in questions_df.iterrows():
            task_id = row['task_id']

            # Load supporting facts for this task
            facts = self._load_supporting_facts(data_dir, task_id)

            # Skip if no supporting facts
            if len(facts) == 0:
                print(f"Warning: No supporting facts for {task_id}, skipping")
                continue

            # Build prompt
            prompt = self._format_prompt(
                dr_question=row['dr_question'],
                persona=row['persona'],
                company=row['company'],
                industry=row['industry'],
                domain=row['domain'],
                facts=facts
            )

            # Create instance with empty references (LLM-as-judge evaluation)
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],  # No ground truth - use LLM judge
                    split=TEST_SPLIT,
                    id=task_id
                )
            )

        return instances
