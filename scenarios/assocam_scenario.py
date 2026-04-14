"""
HELM Scenario: AssoCiAm

Paper: https://arxiv.org/abs/2509.14171 (EMNLP 2025)
Code: https://github.com/lyf15/AssoCiAm
Data: https://drive.google.com/drive/folders/1pvpfOMA6HLq2x3EEXnVhXFo_FY6VdNGo

AssoCiAm evaluates visual associative thinking in MLLMs. Given an image of a
natural object (cloud, beach, waterfall, etc.) whose shape resembles a common
object, the model must identify what the shape most resembles from MC options.

Dataset: 225 images x 3 questions each = 675 questions per subtask
Subtasks: 4T1 (4 options), 7T1 (7 options), 10T1 (10 options)
Total: 2,025 questions across all subtasks

Prompt source: prompt_template/ in Google Drive download (question_4.txt,
  question_7.txt, question_10.txt). Two-shot prompt with worked examples.
  Exact templates used below.

Fields used: QUESTION, OPTIONS (from question.json), gt answer (from gt.json),
  images (from pic/ directory)
Evaluation: exact_match (Top-1 accuracy)
  Paper also uses weighted average: (4*4T1 + 7*7T1 + 10*10T1) / 21
"""

import json
import os
import subprocess
import zipfile
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class AssoCiAmScenario(Scenario):
    name = "assocam"
    description = "chandl2/AssoCiAm"
    tags = ["creativity", "multimodal", "vision", "associative_thinking"]

    SUBTASKS = ["4T1", "7T1", "10T1"]
    DRIVE_FILE_ID = "1t30OAoYvz3rubPbLw18SCRpY0f19C9M0"

    # Exact prompt templates from prompt_template/ files (two-shot format).
    # The preamble and examples are identical across subtasks; only the
    # answer letters and example option counts differ.
    PREAMBLE = (
        "You are a test subject participating in an associative ability test. "
        "Your task is as follows: Given an image and a question, you need to "
        "fully utilize your associative abilities and choose the best option "
        "from the given choices to answer the question. The given image and "
        "multiple-choice question are in the INPUT, specifically:\n"
        "1.IMAGE: A provided image\n"
        "2.QUESTION: A given question related to the image, requiring "
        "association to answer\n"
        "3.OPTION: The given options, from which you need to select the best "
        "option based on your associative thinking to answer the question\n"
        "Your OUTPUT should include ANSWER, formatted as follows:\n"
        "{answer_letters}\n"
        "One of the letters {answer_letters_desc}, representing the best "
        "option for answering the question.\n"
        "Please strictly follow this format to output the answer for the "
        "question posed in the INPUT\n\n"
        "Associative thinking helps people understand things from different "
        "perspectives, enabling them to quickly adapt to new situations and "
        "improve problem-solving abilities. In the field of LLM, associative "
        "ability is also a very important metric. Now, we will test your "
        "associative thinking. Based on the given image, fully utilize your "
        "associative abilities to think about the given question, and select "
        "the best option from the given choices as your answer. Then, output "
        "your answer in the specified format.\n\n"
        "Below are some examples. Please strictly follow the format."
    )

    # Two-shot examples (same across all subtasks, option count varies)
    EXAMPLES = {
        "4T1": (
            '\n\nExample 1:\n'
            'INPUT: {\n'
            '\t"IMAGE": ,\n'
            '\t"QUESTION": "Observe the mountain in the image and use your '
            'associative thinking to consider what it resembles. Please choose '
            'the best option from the following choices to answer the question.",\n'
            '\t"OPTION": "(A) Fish tank\n(B) Tissue\n(C) Eye\n(D) Water bottle"\n'
            '}\nOUTPUT: C\n\n'
            'Example 2:\n'
            'INPUT: {\n'
            '\t"IMAGE": ,\n'
            '\t"QUESTION": Carefully observe the shape of the lake in the image. '
            'What does it remind you of using your associative thinking? Please '
            'choose the best option from the following choices to answer the question.",\n'
            '\t"OPTION": "(A) Cat\n(B) Phone\n(C) Rock\n(D) Alarm clock"\n'
            '}\nOUTPUT: A'
        ),
        "7T1": (
            '\n\nExample 1:\n'
            'INPUT: {\n'
            '\t"IMAGE": ,\n'
            '\t"QUESTION": "Observe the mountain in the image and use your '
            'associative thinking to consider what it resembles. Please choose '
            'the best option from the following choices to answer the question.",\n'
            '\t"OPTION": "(A) Fish tank\n(B) Tissue\n(C) Eye\n(D) Water bottle\n'
            '(E) Fan\n(F) Glasses\n(G) Book"\n'
            '}\nOUTPUT: C\n\n'
            'Example 2:\n'
            'INPUT: {\n'
            '\t"IMAGE": ,\n'
            '\t"QUESTION": Carefully observe the shape of the lake in the image. '
            'What does it remind you of using your associative thinking? Please '
            'choose the best option from the following choices to answer the question.",\n'
            '\t"OPTION": "(A) Cat\n(B) Phone\n(C) Rock\n(D) Alarm clock\n'
            '(E) Shoes\n(F) Bicycle\n(G) Pillow"\n'
            '}\nOUTPUT: A'
        ),
        "10T1": (
            '\n\nExample 1:\n'
            'INPUT: {\n'
            '\t"IMAGE": ,\n'
            '\t"QUESTION": "Observe the mountain in the image and use your '
            'associative thinking to consider what it resembles. Please choose '
            'the best option from the following choices to answer the question.",\n'
            '\t"OPTION": "(A) Fish tank\n(B) Tissue\n(C) Eye\n(D) Water bottle\n'
            '(E) Fan\n(F) Glasses\n(G) Book\n(H) Bag\n(I) Fork\n(J) Stone"\n'
            '}\nOUTPUT: C\n\n'
            'Example 2:\n'
            'INPUT: {\n'
            '\t"IMAGE": ,\n'
            '\t"QUESTION": Carefully observe the shape of the lake in the image. '
            'What does it remind you of using your associative thinking? Please '
            'choose the best option from the following choices to answer the question.",\n'
            '\t"OPTION": "(A) Cat\n(B) Phone\n(C) Rock\n(D) Alarm clock\n'
            '(E) Shoes\n(F) Bicycle\n(G) Pillow\n(H) Door\n(I) Vase\n(J) Shell"\n'
            '}\nOUTPUT: A'
        ),
    }

    SUFFIX = (
        "\n\nReferencing the above examples, based on the following latest "
        "INPUT information, use associative thinking to analyze the given "
        "image, and select the best option from the provided choices to "
        "answer the question. Strictly output the result in the same format "
        "as shown in the examples."
    )

    ANSWER_META = {
        "4T1": {"letters": "A/B/C/D", "desc": "A, B, C, or D"},
        "7T1": {"letters": "A/B/C/D/E/F/G", "desc": "A, B, C, D, E, F, or G"},
        "10T1": {
            "letters": "A/B/C/D/E/F/G/H/I/J",
            "desc": "A, B, C, D, E, F, G, H, I, or J",
        },
    }

    def __init__(self, subtask: str = "4T1"):
        super().__init__()
        if subtask not in self.SUBTASKS:
            raise ValueError(
                f"subtask must be one of {self.SUBTASKS}, got '{subtask}'"
            )
        self.subtask = subtask

    def _download_data(self, output_path: str) -> str:
        """Download and extract benchmark.zip from Google Drive."""
        data_dir = os.path.join(output_path, "assocam_data")
        benchmark_dir = os.path.join(data_dir, "benchmark")

        if not os.path.exists(benchmark_dir):
            os.makedirs(data_dir, exist_ok=True)
            zip_path = os.path.join(data_dir, "benchmark.zip")

            subprocess.run(
                ["pip", "install", "-q", "gdown"],
                check=True, capture_output=True
            )
            subprocess.run(
                [
                    "gdown",
                    f"https://drive.google.com/uc?id={self.DRIVE_FILE_ID}",
                    "-O", zip_path,
                ],
                check=True, capture_output=True
            )
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)

        return benchmark_dir

    def get_instances(self, output_path: str):
        benchmark_dir = self._download_data(output_path)
        subtask_dir = os.path.join(
            benchmark_dir, f"{self.subtask}subtask"
        )

        with open(os.path.join(subtask_dir, "question.json")) as f:
            questions = json.load(f)
        with open(os.path.join(subtask_dir, "gt.json")) as f:
            ground_truth = json.load(f)

        # Build prompt preamble for this subtask
        meta = self.ANSWER_META[self.subtask]
        preamble = self.PREAMBLE.format(
            answer_letters=meta["letters"],
            answer_letters_desc=meta["desc"],
        )
        examples = self.EXAMPLES[self.subtask]
        prompt_prefix = preamble + examples + self.SUFFIX

        num_choices = int(self.subtask.replace("T1", ""))

        instances = []
        for q_id in sorted(questions.keys(), key=int):
            question_text = questions[q_id]["QUESTION"]
            options_text = questions[q_id]["OPTIONS"]
            correct_letter = ground_truth[q_id]
            image_path = os.path.join(subtask_dir, "pic", f"{q_id}.png")

            # Build multimodal content: prompt text + image + question/options
            input_block = (
                f'\n\nINPUT: {{\n'
                f'\t"IMAGE": ,\n'
                f'\t"QUESTION": "{question_text}",\n'
                f'\t"OPTION": "{options_text}"\n'
                f'}}\nOUTPUT: '
            )

            multimedia_content = MultimediaObject([
                MediaObject(content_type="text/plain", text=prompt_prefix),
                MediaObject(
                    content_type="image/png",
                    location=image_path,
                ),
                MediaObject(content_type="text/plain", text=input_block),
            ])

            # All choices as references; correct one tagged
            references = []
            for i in range(num_choices):
                letter = chr(65 + i)
                tags = [CORRECT_TAG] if letter == correct_letter else []
                references.append(Reference(Output(text=letter), tags=tags))

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT,
                id=f"{self.subtask}_{q_id}",
            ))

        return instances
