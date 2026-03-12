"""
HELM Scenario: FunQA - Towards Surprising Video Comprehension

Paper: https://arxiv.org/abs/2306.14899 (FunQA: Towards Surprising Video Comprehension, ECCV 2024)
Code: https://github.com/Jingkang50/FunQA
Dataset: https://huggingface.co/datasets/fesvhtr/FunQA

Task: Video question-answering on surprising, creative, and funny videos
      Models answer questions about counter-intuitive moments in videos

Prompt format:
  <VIDEO: {video_file}>

  {instruction}

  Answer:

Fields used: instruction, visual_input, output, task
Fields skipped: None

Task Types:
  H1-H4: HumorQA - timestamp localization, description, reasoning, title
  C1-C5: CreativeQA - timestamp localization, description, reasoning, title, creativity scoring
  M1-M3: MagicQA - timestamp localization, description, magic method explanation

Dataset:
  - 30,170 test examples from 424 unique videos
  - Three subsets: HumorQA (45%), CreativeQA (23%), MagicQA (32%)
  - Average video length: 19 seconds
  - Average answer length: 34.2 words
  - Evaluation: BLEU-4, ROUGE-L, CIDEr, BLEURT, GPT-4-based scoring

Note: Videos are referenced by filenames. The dataset provides YouTube IDs for downloading.
      This is a multimodal benchmark using HELM's MediaObject for video input.
"""

import json
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class FunQAScenario(Scenario):
    name = "funqa"
    description = "fesvhtr/FunQA"
    tags = ["creativity", "multimodal", "video", "humor", "magic", "creative_performance", "video_qa"]

    DATA_URL = "https://huggingface.co/datasets/fesvhtr/FunQA/raw/main/FunQA_test.json"

    # Video base path - assumes videos are downloaded to output_path/videos/
    # Users need to download videos using YouTube IDs provided by the dataset
    VIDEOS_DIR = "videos"

    def get_instances(self, output_path: str):
        # Download annotation file
        anno_path = os.path.join(output_path, "FunQA_test.json")
        if not os.path.exists(anno_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(self.DATA_URL, anno_path)

        # Load annotations
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instances = []
        videos_dir = os.path.join(output_path, self.VIDEOS_DIR)

        for item in data:
            instruction = item['instruction']
            video_file = item['visual_input']
            answer = item['output']
            task_type = item['task']

            # Construct video path
            video_path = os.path.join(videos_dir, video_file)

            # Create multimodal input with video + text question
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="video/mp4",
                    location=video_path
                ),
                MediaObject(
                    content_type="text/plain",
                    text=f"\n{instruction}\n\nAnswer:"
                )
            ])

            # Reference is the ground-truth answer
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
