"""
HELM Scenario: FutureGen

Paper: FutureGen: A RAG-based Approach to Generate the Future Work of Scientific Article
       https://arxiv.org/abs/2503.16561
Code: https://github.com/IbrahimAlAzhar/FutureWorkGeneration
Dataset: https://huggingface.co/datasets/iaadlab/FutureGen

Task: Generate future work sections for scientific papers based on full paper content

Prompt formats (from code notebooks):

  Primary (RAG-based, from 10_(FutureGen)_RAG_LLM_neurips_data.ipynb):
    Long-form prompt requesting substantial, long-term research goals that extend
    current work meaningfully. Emphasizes ambitious, grounded directions avoiding
    trivial tasks. Output format specifies bulleted list with explanations.

  Alternative (Non-RAG, from 5.(FutureGen)_GPT_3_(all_sections).ipynb):
    "You are an AI trained to analyze scientific research and suggest future
    directions based on the content of a paper. Below, you will find sections
    from a scientific article including the 'Abstract', 'Introduction',
    'Conclusion', 'Limitation', 'Experiment and Results', 'Related Work',
    'Methodology' of a scientific paper. Based on these details, please generate
    comprehensive and plausible future work suggestions that could extend the
    research findings, address limitations, and propose new avenues for
    exploration. Generate a future work based on these texts. Future work
    should be within 100 words."

Evaluation: Open-ended generation (ROUGE, BLEU, BERTScore) or LLM-as-judge
Ground truth: Combined author-mentioned future work + OpenReview peer suggestions

Dataset: Uses NeurIPS subset (278 papers from 2021-2022) with OpenReview feedback.
         ACL files excluded due to missing ground truth columns.

Fields used: df_Concatenated Text (input), future_work_combined (reference)
Fields available but not used: Future_Work_extraction, LLM_extracted_review_future_work
  (these are components that were merged into future_work_combined)
"""

import json
import pandas as pd
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class FuturegenScenario(Scenario):
    name = "futuregen"
    description = "iaadlab/FutureGen"
    tags = ["creativity", "scientific_writing", "future_work"]

    # Exact prompt from 10_(FutureGen)_RAG_LLM_neurips_data.ipynb
    PROMPT_RAG = """I want to generate future work directions for my research paper based on its entire content (all sections, including abstract, introduction, background, methodology, results, discussion, etc.). Please analyze the paper and propose substantial, long-term research goals that extend the current work in a meaningful way, advancing the field or addressing significant open challenges. Ensure the suggested future work directions are ambitious, grounded in the paper's content, and avoid trivial or short-term tasks (e.g., minor experiments, parameter tuning, or small-scale tests). Each direction should be clearly linked to specific aspects of the paper (e.g., limitations, findings, or discussed challenges) and propose innovative, impactful research objectives. If no suitable long-term future work can be derived, clearly state: "No long-term future work directions could be derived from the paper." Provide the generated future work directions in a concise, bulleted list, with each direction accompanied by a brief explanation of how it connects to the paper's content.

Input Text (Paper Content): {paper_text}

Output Format: Future Work Directions (Long-Term Goals)

[Future work direction]: [Brief explanation of how this direction connects to the paper's content and why it is a substantial, long-term goal.]

[Additional future work directions and explanations, if applicable.]

OR

No long-term future work directions could be derived from the paper."""

    # Exact prompt from 5.(FutureGen)_GPT_3_(all_sections).ipynb
    PROMPT_SHORT = """You are an AI trained to analyze scientific research and suggest future directions based on the content of a paper. Below, you will find sections from a scientific article including the 'Abstract', 'Introduction', 'Conclusion','Limitation','Experiment and Results','Related Work','Methodology' of a scientific paper. Based on these details, please generate comprehensive and plausible future work suggestions that could extend the research findings, address limitations, and propose new avenues for exploration. Generate a future work based on these texts. Future work should be within 100 words.

{paper_text}"""

    def __init__(self, prompt_style: str = "rag"):
        """
        Args:
            prompt_style: Which prompt to use. Options: ["rag", "short"]
                         "rag" = Detailed long-form prompt (default, from RAG notebook)
                         "short" = Concise 100-word prompt (from GPT-3 notebook)
        """
        super().__init__()
        if prompt_style not in ["rag", "short"]:
            raise ValueError(f"Invalid prompt_style: {prompt_style}. Must be 'rag' or 'short'")
        self.prompt_style = prompt_style

    def get_instances(self, output_path: str) -> list[Instance]:
        # Download the NeurIPS dataset file
        data_url = "https://huggingface.co/datasets/iaadlab/FutureGen/resolve/main/df_neurips_future_work_dataset.csv"
        data_path = ensure_file_downloaded(
            source_url=data_url,
            target_path="df_neurips_future_work_dataset.csv",
            unpack=False,
        )

        # Load the dataset
        df = pd.read_csv(data_path)

        instances = []
        for idx, row in df.iterrows():
            # Get the input (full paper text)
            paper_text = str(row["df_Concatenated Text"])

            # Get the ground truth (combined future work)
            future_work = str(row["future_work_combined"])

            # Parse the future_work if it's a JSON string (it appears to be a list)
            try:
                future_work_parsed = json.loads(future_work)
                if isinstance(future_work_parsed, list):
                    # Join list elements into single text
                    future_work = "\n\n".join(future_work_parsed)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, use as-is
                pass

            # Build the prompt using selected style
            if self.prompt_style == "rag":
                prompt = self.PROMPT_RAG.format(paper_text=paper_text)
            else:  # short
                prompt = self.PROMPT_SHORT.format(paper_text=paper_text)

            # Create reference
            references = [Reference(Output(text=future_work), tags=[])]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
