"""
HELM Scenario: Recombination Extraction (CHIMERA-KB)

Paper: CHIMERA: A Knowledge Base of Scientific Recombinations
       Sternlicht et al., 2025. arXiv:2505.20779
Code:  https://github.com/noy-sternlicht/CHIMERA-KB
Data:  https://huggingface.co/datasets/noystl/Recombination-Extraction

Task: Given a scientific abstract, extract the most salient recombination as structured JSON.
Recombination = creating original ideas by integrating elements of existing concepts:
  - Combination: authors combine two or more ideas/methods/techniques
  - Inspiration: authors draw from one domain and apply to another (analogy)

Evaluated at 3 levels (single unified prompt, PROMPT_E2E from src/util.py):
  Level 1 - Classification:  Is recombination present? (JSON empty vs. non-empty)
  Level 2 - Entity extraction: What entities are involved?
  Level 3 - Relation extraction: What is the relation type linking the entities?

216 test / 362 train instances from human-annotated scientific abstracts.

Prompt format: Verbatim PROMPT_E2E from src/util.py (paper Appendix / GitHub src/util.py).
  Entity type descriptions filled from NER_ENTITY_TYPES_ATTRIBUTES (src/util.py).
  Model should output JSON within <answer> tags.

Expected output format:
  Relevant (combination): {"combination": {"comb-element": ["entity1", "entity2"]}}
  Relevant (inspiration): {"inspiration": {"inspiration-src": ["src"], "inspiration-target": ["tgt"]}}
  Irrelevant:             {}

Note: The dataset stores "analogy"/"analogy-src"/"analogy-target" keys, but the prompt
instructs models to use "inspiration"/"inspiration-src"/"inspiration-target". The reference
uses the dataset's "analogy" terminology; the custom metric must handle this mapping.

Fields used:   text (input abstract), readable_relations (gold JSON reference),
               document_class (classification label for Level 1 eval)
Fields skipped: paper_id (metadata), entities/relations (character-span format),
                readable_entities (redundant with readable_relations)

Evaluation: custom — JSON parsing + soft entity matching with LLM judge (GPT-4o-mini).
See metric_notes.md for full evaluation methodology including HDR and soft matching.
"""

import json
from typing import List

from datasets import load_dataset

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

# Verbatim from src/util.py (GitHub: noy-sternlicht/CHIMERA-KB)
_PROMPT_E2E = """You are an AI assistant tasked with analyzing scientific abstracts for idea recombination. Your goal is to identify the most salient recombination in the given abstract and format it as a JSON string. Follow these instructions carefully:

1. First, familiarize yourself with the possible entity types for recombinations:

<entity_types>
{ENTITY_TYPE_DESCRIPTIONS}
</entity_types>

2. Now, carefully read the following scientific abstract:

<abstract>
{TEXT}
</abstract>

3. Your task is to extract the most salient recombination from this abstract. A recombination can be either:
   a) Combination: The authors combine two or more ideas, methods, models, techniques, or approaches to obtain a certain goal.
   b) Inspiration: The authors draw inspiration or similarities from one concept, idea, problem, approach, or domain and implement it in another.

4. After identifying the recombination, you will format it as a JSON string in the following structure:

   <recombination>
   {recombination_type: {entity_type_1: [ent_1, ent_2], entity_type_2: [ent_3],...}}
   </recombination>

   If you don't think the text discusses a recombination, or that the recombination is not a central part of the work, return an empty JSON object: {}.

5. Before providing your final answer, use the following scratchpad to think through the process:

   <scratchpad>
   1. Identify the main ideas, methods, or approaches discussed in the abstract.
   2. Determine if there is a clear combination of ideas or if one idea inspired the application in another domain.
   3. Identify the specific entities involved in the recombination.
   4. Classify the entities according to the provided entity types.
   5. Determine the recombination type (combination or inspiration).
   </scratchpad>

6. Now, provide your final output in the specified JSON format. Ensure that the output is a valid JSON string. If the output is empty, return {}. Place your answer within <answer> tags.

Remember to carefully analyze the abstract and only identify a recombination if it is clearly present and central to the work described."""

# Entity type descriptions built from NER_ENTITY_TYPES_ATTRIBUTES (src/util.py)
# prompt_type_name overrides entity_type for display in prompt
_ENTITY_TYPE_DESCRIPTIONS = (
    "1. comb-element: An idea, method, model, technique, or approach combined in the text "
    "with other elements.\n"
    "2. inspiration-src: A concept, idea, problem, approach, or domain the authors drew "
    "inspiration from.\n"
    "3. inspiration-target: A concept, idea, problem, approach, or domain in which the authors "
    "utilize the inspiration they drew from the inspiration source."
)


class RecombinationExtractionScenario(Scenario):
    name = "recombination_extraction"
    description = "noystl/Recombination-Extraction"
    tags = ["creativity", "scientific_creativity", "information_extraction", "recombination"]

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("noystl/Recombination-Extraction")

        instances = []
        split_map = {"train": TRAIN_SPLIT, "test": TEST_SPLIT}

        for split_name, helm_split in split_map.items():
            for item in dataset[split_name]:
                # Use str.replace instead of .format() because the prompt contains
                # literal curly braces (e.g., {recombination_type: ...}) in the
                # example output format that would confuse Python's format().
                prompt = (
                    _PROMPT_E2E
                    .replace("{ENTITY_TYPE_DESCRIPTIONS}", _ENTITY_TYPE_DESCRIPTIONS)
                    .replace("{TEXT}", item["text"])
                )

                # Gold reference: structured JSON from readable_relations
                # {} for irrelevant, relation dict for relevant examples
                rel = item["readable_relations"]
                if isinstance(rel, str):
                    rel = json.loads(rel)
                gold_json = json.dumps(rel)

                references = [
                    Reference(Output(text=gold_json), tags=[CORRECT_TAG])
                ]

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=helm_split,
                    )
                )

        return instances
