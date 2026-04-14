"""
HELM Scenario: MixAssist (Music Mixing Dialogue Assistant)

Data:  https://huggingface.co/datasets/mclemcrew/MixAssist
Paper: No associated paper found; dataset-only release by Michael Clemens.

MixAssist contains 640 multi-turn conversation turns (340 train / 50 val /
250 test) transcribed from real audio mixing sessions between a human audio
engineer and an AI assistant. The sessions cover six mixing topics:
  drums, overall_mix, guitars, vocals, bass, keys.

Each row is one conversational turn. The `has_content` flag distinguishes
substantive dialogue turns (62% of test rows) from filler turns — short
acknowledgments ("Mm-hmm.") or the templated trigger "Please analyze this
audio segment." The scenario filters to `has_content=True` only (156 test
instances), where the exchanges are meaningful discussions about instrument
identification, mix balance, effects, and creative mixing decisions.

Task: given the conversation history (text transcriptions) and the current
user utterance, generate the assistant's next response.

Prompt format (no paper-specified prompt; standard multi-turn dialogue format):

  The following is a conversation between an audio engineer and an AI
  assistant during a music mixing session. The topic is: {topic}.

  User: {prior user turn}
  Assistant: {prior assistant turn}
  ...
  User: {current user utterance}
  Assistant:

Fields used:   has_content (filter), input_history (context), user (current
               turn), assistant (reference response), topic, conversation_id,
               turn_id
Fields skipped: audio_file (audio paths; not needed for text-only task)

Dataset quirks:
  - input_history is a native Python list — no json.loads() needed.
  - Each history entry has keys: role, content, audio_file.
  - has_content=False rows include templated "Please analyze this audio
    segment." user turns and thin acknowledgment exchanges; skip these.
  - No train/val splits needed — only the test split is used for evaluation.

Evaluation: open_ended (ROUGE-L, BLEU)
"""

from typing import List

from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)


class MixAssistScenario(Scenario):
    """
    MixAssist: predict the assistant's next turn in a multi-turn music
    mixing session dialogue, given the full conversation history.

    Filters to has_content=True rows (156 test instances) where exchanges
    contain substantive audio-engineering discussion.
    """

    name = "mixassist"
    description = "mclemcrew/MixAssist"
    tags = ["creativity", "music", "dialogue", "audio_engineering", "open_ended"]

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("mclemcrew/MixAssist", split="test")

        instances = []
        for item in dataset:
            if not item["has_content"]:
                continue

            topic = item["topic"].replace("_", " ")

            # Build conversation history from prior turns (text only)
            history_lines = []
            for turn in item["input_history"]:
                content = (turn.get("content") or "").strip()
                if content:
                    role = turn["role"].capitalize()  # "User" or "Assistant"
                    history_lines.append(f"{role}: {content}")

            user_turn = item["user"].strip()
            history_text = "\n".join(history_lines)

            if history_text:
                prompt = (
                    f"The following is a conversation between an audio engineer "
                    f"and an AI assistant during a music mixing session. "
                    f"The topic is: {topic}.\n\n"
                    f"{history_text}\n"
                    f"User: {user_turn}\n"
                    f"Assistant:"
                )
            else:
                prompt = (
                    f"The following is a conversation between an audio engineer "
                    f"and an AI assistant during a music mixing session. "
                    f"The topic is: {topic}.\n\n"
                    f"User: {user_turn}\n"
                    f"Assistant:"
                )

            instances.append(Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=item["assistant"].strip()), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
                extra_data={
                    "conversation_id": item["conversation_id"],
                    "turn_id": str(item["turn_id"]),
                    "topic": item["topic"],
                },
            ))

        return instances
