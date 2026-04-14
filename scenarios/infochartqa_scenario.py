"""
HELM Scenario: InfoChartQA

Paper: InfoChartQA: A Dataset for Visual Question Answering on Infographic Charts
GitHub: https://github.com/CoolDawnAnt/InfoChartQA
Dataset: https://huggingface.co/datasets/Jietson/InfoChartQA

Task: Multimodal question answering on infographic charts with pictorial elements
(pictograms, icons, visual metaphors). Tests both content understanding and
visual design comprehension.

Dataset: 58,857 questions across 5,948 chart pairs
- text: 50,920 text-based QA on charts
- visual_metaphor: 462 visual metaphor understanding questions
- visual_basic: 7,475 basic visual element understanding questions

Evaluation: Exact match accuracy

Prompt format (from HuggingFace dataset card):
  {question}{instructions}

Instructions are task-specific and provide:
- Output format constraints (e.g., "Your response should only contain...")
- Calculation methods
- Value formatting requirements

Chart types: 54 types including bar, line, pie, treemap, funnel, radar, etc.
Question types: 26 types including value extraction, trend analysis, categorization,
                aggregation, ranking, extreme values, etc.

Fields used: question, instructions, url (image), answer, extra_input_figure_bboxes
Fields available: question_id, question_type_id, question_type_name, figure_id,
                  data_fact, difficulty, chart_type

Note: visual_basic questions may include cropped image sections (extra_input_figure_bboxes)
to focus on specific visual elements.
"""

from typing import List
from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject


class InfoChartQAScenario(Scenario):
    """
    InfoChartQA: Multimodal question answering on infographic charts.

    Evaluates models' ability to understand charts with creative visual elements
    like pictograms, icons, and metaphors.
    """

    name = "infochartqa"
    description = "Jietson/InfoChartQA"
    tags = ["creativity", "visual_reasoning", "chart_qa", "multimodal"]

    def __init__(self, subset: str = "all"):
        """
        Args:
            subset: Which split to use - "text", "visual_metaphor", "visual_basic", or "all"
        """
        super().__init__()
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        # Load dataset from HuggingFace
        dataset = load_dataset("Jietson/InfoChartQA")

        instances = []

        # Determine which splits to include
        if self.subset == "all":
            splits_to_use = ["text", "visual_metaphor", "visual_basic"]
        else:
            splits_to_use = [self.subset]

        for split_name in splits_to_use:
            split_data = dataset[split_name]

            for item in split_data:
                # Build question text with instructions
                question_text = item['question']
                if item['instructions']:
                    question_text += "\n\n" + item['instructions']

                # Create MediaObject for the chart image
                # Fix malformed URLs in the dataset
                url = item['url']
                # Strip repeated leading "h"s (e.g. "hhttps://" → "https://")
                while url.startswith("hhttp"):
                    url = url[1:]
                # Add scheme if missing entirely (e.g. "preview.redd.it/...")
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url

                chart_image = MediaObject(
                    content_type="image/png",
                    location=url
                )

                # For visual_basic split, some questions include cropped sections
                # These are indicated by extra_input_figure_bboxes
                # Note: HELM doesn't natively support bbox cropping, so we include
                # the full image. Alternative: pre-crop and host separately.
                media_objects = [chart_image]

                # Create multimedia input
                multimedia_content = MultimediaObject(media_objects)

                # Create reference with the ground truth answer
                references = [
                    Reference(
                        output=Output(text=item['answer']),
                        tags=[CORRECT_TAG]
                    )
                ]

                instances.append(
                    Instance(
                        input=Input(
                            text=question_text,
                            multimedia_content=multimedia_content
                        ),
                        references=references,
                        split=TEST_SPLIT,
                    )
                )

        return instances
