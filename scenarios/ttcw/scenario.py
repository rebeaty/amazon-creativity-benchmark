"""
HELM Scenario: Torrance Test of Creative Writing (TTCW)

Paper: https://arxiv.org/abs/2510.05135 (Curiosity-Driven LLM-as-a-judge for Personalized Creative Judgment)
Original TTCW: Chakrabarty et al. (2024) - https://arxiv.org/abs/2309.14556
Dataset: https://huggingface.co/datasets/Salesforce/ttcw_creativity_eval

Prompt format:
  Story: {plot_summary}

  Question: {ttcw_question}

  Based on this story, answer Yes or No to the question above.
  Answer:

Prompt source: Adapted from paper (Section 2.2 describes input format as Qd+S)
Fields used: story_metadata (plot_summary), test_metadata (ttcw_question, ttcw_category),
             annotations (binary_verdict)
Fields skipped: content (external URLs to full stories), expert_idx, explanation

Task: Binary classification predicting expert judgments on 14 TTCW dimensions
      (Fluency, Flexibility, Originality, Elaboration categories)
Evaluation: Models predict Yes/No for whether a story meets creative criteria

Note: Dataset contains 48 stories × 14 dimensions × 3 experts = 2,016 instances
      Stories include both professional works and GPT-generated content
      Full story texts not included (only plot summaries); see story_metadata.content for URLs
"""

import ast
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)

class TTCWScenario(Scenario):
    name = "ttcw"
    description = "Salesforce/ttcw_creativity_eval"
    tags = ["creativity", "creative_writing", "binary_classification", "llm_as_judge"]

    def get_instances(self, output_path):
        dataset = load_dataset("Salesforce/ttcw_creativity_eval", split="train")

        instances = []

        # Each row is one story with 14 dimension columns
        for story_row in dataset:
            # Iterate through each TTCW dimension column
            for dimension_col in dataset.column_names:
                # Parse the JSON/string data
                dimension_data = ast.literal_eval(story_row[dimension_col])

                # Extract components
                story_metadata = dimension_data[0]['story_metadata']
                test_metadata = dimension_data[1]['test_metadata']
                annotations = dimension_data[2]['annotations']

                plot_summary = story_metadata['plot_summary']
                question = test_metadata['ttcw_question']
                category = test_metadata['ttcw_category']

                # Create one instance per expert annotation (3 experts per dimension)
                for annotation in annotations:
                    expert_verdict = annotation['binary_verdict']  # "Yes" or "No"

                    # Format prompt
                    prompt = f"Story: {plot_summary}\n\n"
                    prompt += f"Question: {question}\n\n"
                    prompt += "Based on this story, answer Yes or No to the question above.\n"
                    prompt += "Answer:"

                    # Create references for binary classification
                    references = []
                    for answer in ["Yes", "No"]:
                        is_correct = (answer == expert_verdict)
                        tags = [CORRECT_TAG] if is_correct else []
                        references.append(Reference(Output(text=answer), tags=tags))

                    instances.append(Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=TEST_SPLIT
                    ))

        return instances
