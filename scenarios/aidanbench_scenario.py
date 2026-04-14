"""
HELM Scenario: AidanBench — Open-Ended Novel Idea Generation

Paper: "AidanBench: Evaluating Novel Idea Generation on Open-Ended Questions"
  McLaughlin, Campbell, Uppuluri — NeurIPS 2024 Workshop on Language Gamification
  OpenReview: https://openreview.net/forum?id=fz969ahcvJ

Code: https://github.com/aidanmclaughlin/AidanBench

Note: This benchmark is listed as "Codeforces" in benchmarks.json but is AidanBench —
  a creativity benchmark with 66 open-ended questions (no Codeforces content).

Task: Given an open-ended question, generate multiple distinct, creative responses.
  The original benchmark is multi-turn: a model keeps generating new answers to the
  same question until it fails a coherence check (≤15/100) or a novelty check
  (embedding similarity ≤0.15 vs prior responses). The score is the count of valid
  responses before stopping.

  This scenario adapts AidanBench to HELM's single-turn architecture using batch
  prompting: the model generates N=30 responses in one shot. This faithfully captures
  the diversity/novelty objective while being compatible with HELM.
  See metric_notes.md for evaluation details.

Dataset: 66 open-ended questions spanning economics, science, creative design,
  philosophy, mathematics, and everyday creativity. Hardcoded from:
  https://raw.githubusercontent.com/aidanmclaughlin/AidanBench/main/benchmark/question_list.py

Prompt format (adapted from AidanBench GitHub, single-turn batch variant):
  Generate {n} distinct and creative responses to the following question.
  Each response should present a substantially different idea, perspective,
  or approach — do not repeat or rephrase earlier responses.

  Question: {question}

  Responses:
  1.

  No official single-turn prompt specified; original benchmark uses iterative prompting
  with prior responses appended. This batch format follows the same spirit.

Fields used:   question (hardcoded from question_list.py)
Fields skipped: none (no external dataset file needed)

Evaluation: custom metrics — see metric_notes.md
  Self-BLEU for within-response diversity; LLM-as-judge for coherence per item;
  AidanBench Score = count of responses passing both coherence and novelty thresholds.

Split used: TEST_SPLIT (all 66 questions)
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Scenario,
)

# 66 questions from https://github.com/aidanmclaughlin/AidanBench/blob/main/benchmark/question_list.py
QUESTIONS: List[str] = [
    "Provide an explanation for Japan's Lost Decades.",
    "What is a cause of World War 1?",
    "Why might the United States government nationalize ASI development?",
    "How might you use a brick and a blanket?",
    "What architectural design features should be included in a tasteful home?",
    "What's one way to use oregano?",
    "How might we enable LLMs to spend more output tokens to get predictably better results?",
    "Propose a solution to Los Angeles traffic.",
    "What activities might I include at a party for firefighters?",
    "Why did Rome fall?",
    "How could we redesign the American education system to better prepare students for the 22nd century?",
    "What might be an unexpected consequence of achieving nuclear fusion?",
    "Describe a plausible alien life form that doesn't rely on carbon-based biology.",
    "How could we modify the rules of chess to make it more exciting for spectators?",
    "What would be the implications of a universal basic income on American society?",
    "Propose an alternative to democracy for successfully and fairly governing a country.",
    "How might we terraform Venus instead of Mars, and why?",
    "Design an original sport that combines elements of three existing sports.",
    "What could be a novel use for blockchain technology outside of cryptocurrency?",
    "How might human evolution be affected by long term space colonization?",
    "Invent a new musical instrument and describe how it would be played.",
    "What might be an unexpected solution to reducing plastic waste in oceans?",
    "How might we design a city that functions entirely underwater?",
    "What societal changes might occur if humans could communicate with animals?",
    "I have a fleet of 100 drones, how can I use them?",
    "Describe a sustainable farming method that could be used in a floating city.",
    "If all industrial buildings were required to be bioluminescent, what effects might this have?",
    "Invent a device that translates human dreams into tangible visualizations.",
    "How might daily life change if humans had the ability to breathe underwater?",
    "Create a recipe for a smoothie to have first thing in the morning that will give me energy.",
    "What new environmental challenges might arise if all vehicles were self-driving?",
    "Design a fashion line that incorporates smart clothing technology.",
    "Imagine a world where books are replaced by holographic storytelling; what impacts might this have?",
    "What might be the implications of having robots as therapists?",
    "Propose a system for energy-harvesting from natural disasters.",
    "How might the education system be revolutionized by virtual reality classrooms?",
    "What unique challenges might arise in a society where everyone lives to be 150 years old?",
    "Describe a mobile app that encourages acts of kindness.",
    "Give me a diet that a human should eat to best prepare them for a hypothetical hibernation.",
    "Imagine a competition where contestants build habitats for animals; what might be included?",
    "What might be the benefits of reintroducing dinosaurs into modern ecosystems?",
    "Propose a mechanism for reducing food waste through technological innovation.",
    "Design a city where all modes of transportation are vertically oriented.",
    "What is a useless ingredient for a baker to have in their kitchen?",
    "Imagine a civilization based entirely on underwater agriculture; what technology might be required?",
    "How might public health improve if all houses had healing gardens?",
    "Describe how to build a time travel machine assuming I can procure any required material.",
    "Create a concept for a museum that showcases possible futures.",
    "What would be the impact of a government mandating weekly mental health days?",
    "Invent a game that teaches players about sustainable living.",
    "How could we design a school that encourages lifelong learning from adults as well as children?",
    "Describe a new form of professional sports that focuses on non-physical competition.",
    "Devise a farming technique to harvest dinoflagellates and retain their bioluminescence.",
    "How can a perfumer increase the sillage of their scent?",
    "What can an artist who enjoys Basquiat's art take inspiration from?",
    "Provide a proof for the Pythagorean theorem.",
    "A perfumer is creating a unique, unisex scent with benzoin and vanilla middle notes, what base and top notes should they add?",
    "What is a non poisonous recipe nobody has prepared before?",
    "Design an earring that would complement someone with a round face and small ears.",
    "Devise a strategy for me to always find gems when I mine in Webkinz World's mines.",
    "What is a human value to align a large language model on?",
    "Give me a proof to Euclid's theorem of the infinitude of primes.",
    "Give a proof for the Cauchy-Schwarz Inequality.",
]

# Number of responses to request per question (matches typical AidanBench run depth)
DEFAULT_NUM_RESPONSES = 30

PROMPT_TEMPLATE = (
    "Generate {n} distinct and creative responses to the following question. "
    "Each response should present a substantially different idea, perspective, or "
    "approach — do not repeat or rephrase earlier responses.\n\n"
    "Question: {question}\n\n"
    "Responses:\n1."
)


class AidanBenchScenario(Scenario):
    """
    AidanBench: Open-ended novel idea generation across 66 diverse questions.

    Each instance asks the model to generate NUM_RESPONSES distinct creative
    answers to one question. Evaluation uses Self-BLEU for diversity and
    LLM-as-judge for coherence — see metric_notes.md.
    """

    name = "aidanbench"
    description = "github.com/aidanmclaughlin/AidanBench"
    tags = ["creativity", "open_ended", "idea_generation", "diversity"]

    def __init__(self, num_responses: int = DEFAULT_NUM_RESPONSES):
        """
        Args:
            num_responses: Number of distinct responses to request per question.
                           Defaults to 30 (matches typical AidanBench run depth).
        """
        super().__init__()
        self.num_responses = num_responses

    def get_instances(self, output_path: str) -> List[Instance]:
        instances = []
        for question in QUESTIONS:
            prompt = PROMPT_TEMPLATE.format(
                n=self.num_responses,
                question=question,
            )
            # No reference answers — evaluated by diversity and coherence metrics
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                )
            )
        return instances
