"""
HELM Scenario: CPers (Creativity in Persian)

Paper: Evaluating the Creativity of LLMs in Persian Literary Text Generation
       https://arxiv.org/abs/2509.18401
       Tourajmehr, A., & Modarres, M. R. (2025). arXiv preprint.

Dataset: https://huggingface.co/datasets/teias-ai/CPers
         4,371 Persian literary texts across 20 cultural/emotional topics
         200 human-annotated subset with creativity scores

Task: Generate single-sentence Persian literary texts on given topics.
      Evaluates creativity in Persian text generation using culturally-grounded
      Torrance Tests of Creative Thinking (TTCT) framework.

Prompt format: From paper Section 3.1
  "Write a literary text in one sentence about {Topic}" (in Persian)
  درباره {موضوع} یک متن ادبی در یک جمله بنویس

Topics (20 culturally diverse themes):
  - عشق (Love) - 500 examples
  - دلتنگی (Longing) - 670 examples
  - دلشکستگی، غم، اندوه (Heartbreak/Sorrow) - 500 examples
  - مادر (Mother) - 499 examples
  - پدر (Father) - 330 examples
  - رفاقت (Friendship) - 266 examples
  - Plus 14 additional topics (seasons, celebrations, life stages, emotions)

Evaluation: LLM-as-judge using Claude 3.7 Sonnet (see annotator_notes.md)
  Four TTCT dimensions (1-5 scale, 3 questions each):
  - Originality: Creativity, avoidance of clichés, literary devices
  - Fluency: Grammar, naturalness, appropriateness
  - Flexibility: Multiple perspectives, stylistic variety
  - Elaboration: Vocabulary richness, imagery, emotional conveyance

  Plus rhetorical device analysis:
  - Simile (تشبیه)
  - Metaphor (استعاره)
  - Hyperbole (اغراق)
  - Antithesis (تضاد)

Fields used: Topic (as input), Text (as reference)
Fields skipped: ID (metadata)

Notes:
  - Paper used temperature=1 for generation
  - Study evaluated GPT-3.5, GPT-4.1, DeepSeek-V3, DeepSeek-R1, Qwen2.5, Gemma
  - Key finding: Models rely on learned patterns vs genuine creativity
  - Rhetorical device usage skews toward simile/metaphor
"""

from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)


class CPersScenario(Scenario):
    """
    CPers: Creativity in Persian literary text generation.

    Evaluates LLM creativity in generating single-sentence Persian literary texts
    across 20 culturally and emotionally diverse topics using adapted Torrance Tests
    of Creative Thinking (TTCT) framework.
    """

    name = "cpers"
    description = "teias-ai/CPers"
    tags = ["creativity", "generation", "multilingual", "persian"]

    # Prompt template from paper (Section 3.1)
    # English: "Write a literary text in one sentence about {Topic}"
    PROMPT_TEMPLATE_FA = "درباره {topic} یک متن ادبی در یک جمله بنویس"
    PROMPT_TEMPLATE_EN = "Write a literary text in one sentence about {topic}"

    def get_instances(self, output_path: str) -> list[Instance]:
        """
        Generate instances for CPers evaluation.

        Each instance:
        - Input: Prompt with topic in Persian
        - Reference: Human-written literary text for that topic (with CORRECT_TAG)

        Returns:
            List of 4,371 Instance objects, one per human-written text in dataset.
        """
        # Load dataset
        dataset = load_dataset("teias-ai/CPers", split="test")

        instances = []
        for item in dataset:
            topic = item["Topic"]
            text = item["Text"]

            # Create prompt using Persian template
            prompt = self.PROMPT_TEMPLATE_FA.format(topic=topic)

            # Add human-written text as reference
            # Note: In LLM-as-judge evaluation, generated text is compared to reference
            # for context, but primarily evaluated on TTCT dimensions
            reference = Reference(output=Output(text=text), tags=[CORRECT_TAG])

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[reference],
                    split=TEST_SPLIT,
                )
            )

        return instances
