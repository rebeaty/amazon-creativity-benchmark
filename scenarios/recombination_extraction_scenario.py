"""
HELM Scenario: Recombination Extraction

Paper: https://arxiv.org/abs/2505.20779 (2025)
Title: CHIMERA: A Knowledge Base of Scientific Idea Recombinations for Research Analysis and Ideation
Code: https://github.com/noy-sternlicht/CHIMERA-KB
Data: https://huggingface.co/datasets/noystl/Recombination-Extraction

Task: Extract recombination of ideas from scientific abstracts in JSON format.
Recombination types:
  - Combination: combining two or more ideas/methods (entities: comb-element)
  - Inspiration: drawing from one domain to another (entities: analogy-src, analogy-target)

Prompt format (PROMPT_E2E from source code):
  You are an AI assistant tasked with analyzing scientific abstracts for idea recombination.
  Your goal is to identify the most salient recombination in the given abstract and format
  it as a JSON string. Follow these instructions carefully:

  1. First, familiarize yourself with the possible entity types for recombinations:

  <entity_types>
  - comb-element: An idea, method, model, technique, or approach combined with other elements.
  - inspiration-src: A concept, idea, problem, approach, or domain the authors drew inspiration from.
  - inspiration-target: A concept, idea, problem, approach, or domain in which the authors
    utilize the inspiration they drew from the inspiration source.
  </entity_types>

  2. Now, carefully read the following scientific abstract:

  <abstract>
  {TEXT}
  </abstract>

  3. Your task is to extract the most salient recombination from this abstract.
     A recombination can be either:
     a) Combination: The authors combine two or more ideas, methods, models, techniques,
        or approaches to obtain a certain goal.
     b) Inspiration: The authors draw inspiration or similarities from one concept, idea,
        problem, approach, or domain and implement it in another.

  4. After identifying the recombination, format it as a JSON string:

     <recombination>
     {recombination_type: {entity_type_1: [ent_1, ent_2], entity_type_2: [ent_3],...}}
     </recombination>

     If you don't think the text discusses a recombination, or that the recombination is
     not a central part of the work, return an empty JSON object: {}.

  5. Before providing your final answer, use the following scratchpad to think through:

     <scratchpad>
     1. Identify the main ideas, methods, or approaches discussed in the abstract.
     2. Determine if there is a clear combination of ideas or if one idea inspired
        the application in another domain.
     3. Identify the specific entities involved in the recombination.
     4. Classify the entities according to the provided entity types.
     5. Determine the recombination type (combination or inspiration).
     </scratchpad>

  6. Now, provide your final output in the specified JSON format. Ensure that the output
     is a valid JSON string. If the output is empty, return {}. Place your answer within
     <answer> tags.

  Remember to carefully analyze the abstract and only identify a recombination if it is
  clearly present and central to the work described.

Prompt source: PROMPT_E2E from src/util.py in GitHub repository
Fields used: text, document_class (for evaluation ground truth)
Fields skipped: paper_id, entities, relations, readable_entities, readable_relations

Evaluation notes:
  - Binary classification: Empty JSON {} = irrelevant, non-empty JSON = relevant
  - This scenario implements Level 1 (classification) via JSON output parsing
  - Full benchmark includes entity extraction (Level 2) and relation extraction (Level 3)
  - See scenarios/recombination_extraction/metric_notes.md for full evaluation details
  - Models must output valid JSON; evaluation requires parsing to check if empty or not

Test set: 216 examples (100 relevant, 116 irrelevant)
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


# Entity type descriptions (from NER_ENTITY_TYPES_ATTRIBUTES in source)
ENTITY_TYPES_DESC = """- comb-element: An idea, method, model, technique, or approach combined in the text with other elements.
- inspiration-src: A concept, idea, problem, approach, or domain the authors drew inspiration from.
- inspiration-target: A concept, idea, problem, approach, or domain in which the authors utilize the inspiration they drew from the inspiration source."""


class RecombinationExtractionScenario(Scenario):
    name = "recombination_extraction"
    description = "noystl/Recombination-Extraction"
    tags = ["creativity", "scientific_ideation", "information_extraction", "structured_output"]

    def get_instances(self, output_path):
        # Load test split (216 examples: 100 relevant, 116 irrelevant)
        dataset = load_dataset("noystl/Recombination-Extraction", split="test")

        instances = []
        for item in dataset:
            # Build prompt using PROMPT_E2E format from source
            prompt = f"""You are an AI assistant tasked with analyzing scientific abstracts for idea recombination. Your goal is to identify the most salient recombination in the given abstract and format it as a JSON string. Follow these instructions carefully:

1. First, familiarize yourself with the possible entity types for recombinations:

<entity_types>
{ENTITY_TYPES_DESC}
</entity_types>

2. Now, carefully read the following scientific abstract:

<abstract>
{item['text']}
</abstract>

3. Your task is to extract the most salient recombination from this abstract. A recombination can be either:
   a) Combination: The authors combine two or more ideas, methods, models, techniques, or approaches to obtain a certain goal.
   b) Inspiration: The authors draw inspiration or similarities from one concept, idea, problem, approach, or domain and implement it in another.

4. After identifying the recombination, you will format it as a JSON string in the following structure:

   <recombination>
   {{recombination_type: {{entity_type_1: [ent_1, ent_2], entity_type_2: [ent_3],...}}}}
   </recombination>

   If you don't think the text discusses a recombination, or that the recombination is not a central part of the work, return an empty JSON object: {{}}.

5. Before providing your final answer, use the following scratchpad to think through the process:

   <scratchpad>
   1. Identify the main ideas, methods, or approaches discussed in the abstract.
   2. Determine if there is a clear combination of ideas or if one idea inspired the application in another domain.
   3. Identify the specific entities involved in the recombination.
   4. Classify the entities according to the provided entity types.
   5. Determine the recombination type (combination or inspiration).
   </scratchpad>

6. Now, provide your final output in the specified JSON format. Ensure that the output is a valid JSON string. If the output is empty, return {{}}. Place your answer within <answer> tags.

Remember to carefully analyze the abstract and only identify a recombination if it is clearly present and central to the work described."""

            # Create references based on expected output format
            # For classification: {} = irrelevant, non-empty JSON = relevant
            correct_class = item['document_class']  # 'relevant' or 'irrelevant'

            # Reference outputs: empty JSON for irrelevant, example JSON for relevant
            references = [
                Reference(
                    output=Output(text="{}"),
                    tags=[CORRECT_TAG] if correct_class == "irrelevant" else []
                ),
                Reference(
                    output=Output(text='{"combination": {"comb-element": ["element1", "element2"]}}'),
                    tags=[CORRECT_TAG] if correct_class == "relevant" else []
                )
            ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT
                )
            )

        return instances
