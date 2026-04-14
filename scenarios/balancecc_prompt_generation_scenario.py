"""
HELM Scenario: BalanceCC Creative Prompt Generation

Paper: https://arxiv.org/abs/2309.16496 (CCEdit: Creative and Controllable Video Editing via Diffusion Models)
Code: https://github.com/RuoyuFeng/CCEdit
Dataset: https://huggingface.co/datasets/RuoyuFeng/BalanceCC

Task: Creative prompt transformation for video editing
      Given an original video description and editing type, generate a target prompt
      that describes the edited video following the editing specifications.

Prompt: Adapted from Appendix B.1 (GPT-4V dataset creation prompt)
        Includes editing type definitions, examples, and fantasy level guidelines.

Fields used: Original Prompt, Editing Type, Target Prompt, Fantasy Level,
             Video Type, Camera Motion, Object Motion, Scene Complexity
Fields skipped: Video Name (used as ID only)

Note: Original benchmark evaluates video editing systems through user studies.
      This text-based scenario tests creative prompt engineering abilities.

Dataset:
  - 100 videos × 4 editing types = 400 examples
  - Editing types: Style Change, Object Change, Background Change, Multiple Change
  - Fantasy levels: 1 (realistic), 2 (moderate), 3 (highly fantastical)
  - Video categories: Animal, Human, Object, Landscape
"""

import json
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class BalanceCCPromptGenerationScenario(Scenario):
    name = "balancecc_prompt_generation"
    description = "RuoyuFeng/BalanceCC"
    tags = ["creativity", "prompt_engineering", "video_editing", "generation"]

    DATA_URL = "https://huggingface.co/datasets/RuoyuFeng/BalanceCC/raw/main/BalanceCC.json"

    # Exact prompt from Appendix B.1 (adapted for single target prompt generation)
    SYSTEM_PROMPT = """Now I'm trying to build a benchmark for video editing. I need you to assist me in doing that. I will provide the description of a video for you. About the video description, I hope you provide the following information to me:

Generate a text prompt to edit this video. Be creative and imaginative. Offer me the corresponding "Target Prompt" for the given "Editing Type" and "Fantasy Level".

The "Editing Type" should be one of "Style Change, Object Change, Background Change, and Compound Change." About "Style Change", some examples are "old movies", "impressionist style", "Van Gogh style", "neon lights style", "cyberpunk style", "sepia-toned photo", "grayscale", "claymation style", "origami style", "oil painting style". About "Object Change", just change the object into other ones, like "dog to cat", "cat to tiger", "human to bear", "human to teddy bear", and even some specific identities like "Ironman". About "Background Change", just change the background, here are some examples, "in the Mars", "in the moon", "in the forest", "in the ocean", "in the castle". You can pick one of the examples I provided, and I hope you can also consider other ones that you think are interesting or suit this video. About "Compound Change", just combine what mentioned above.

Please remember, be creative and imaginative, and don't be too outrageous.

The form of "Target Prompt" should be just like a description of a video, don't say something like "Transform the background into moon." Here is an example, the original prompt is "A majestic black swan gracefully floats on calm waters, with its reflection visible.", the "Target Prompt" can be "An elegant flamingo swan gracefully floats on calm waters, with its reflection visible, set against a backdrop of a mystical enchanted forest.".

As for the "Fantasy Level" for each "Target Prompt", it indicates the degree of imagination. For example, if you change the cat to a tiger or change the background from autumn to winter, it can be seen as a relatively low degree of imagination. Transforming a cat into pixel tiger or tiger made of origami is relative high degree of imagination. Here is also 1-3 in total 3 levels. And similar to the description, be brief, concise, and precise."""

    def get_instances(self, output_path: str):
        # Download dataset
        data_path = os.path.join(output_path, "BalanceCC.json")
        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(self.DATA_URL, data_path)

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instances = []
        for video in data:
            video_type = video['Video Type']
            original_prompt = video['Original Prompt']
            camera_motion = video['Camera Motion']
            object_motion = video['Object Motion']
            scene_complexity = video['Scene Complexity']

            # Create one instance per editing variation
            for editing in video['Editing']:
                editing_type = editing['Editing Type']
                target_prompt = editing['Target Prompt']
                fantasy_level = editing['Fantasy Level']

                # Build prompt using structure from Appendix B.1
                prompt = self.SYSTEM_PROMPT + "\n\n"
                prompt += "---\n\n"
                prompt += f"Original Video Description: {original_prompt}\n\n"
                prompt += f"Video Category: {video_type}\n"
                prompt += f"Scene Complexity: {scene_complexity}/3 (1=simple, 2=moderate, 3=complex)\n"
                prompt += f"Camera Motion: {camera_motion}/3\n"
                prompt += f"Object Motion: {object_motion}/3\n\n"
                prompt += f"Editing Type: {editing_type}\n"
                prompt += f"Fantasy Level: {fantasy_level}/3 (1=low imagination, 2=moderate, 3=high imagination)\n\n"
                prompt += "Generate a brief, concise target prompt that describes the video after editing. "
                prompt += "Write it as a description, not instructions.\n\n"
                prompt += "Target Prompt:"

                # Reference is the human-annotated target prompt
                references = [Reference(Output(text=target_prompt), tags=[CORRECT_TAG])]

                instances.append(Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT
                ))

        return instances
