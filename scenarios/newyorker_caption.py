"""
HELM Scenario: New Yorker Caption Contest

Paper: Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning
       https://arxiv.org/abs/2406.10522
       NeurIPS 2024
Code: https://github.com/yguooo/cartoon-caption-generation
Dataset: yguooo/newyorker_caption_ranking (Hugging Face)

Task: Multiple-choice caption selection for New Yorker cartoons
      Models view a cartoon and select the funniest caption from 4 options

Prompt format:
  [CARTOON IMAGE]

  Which caption is funniest?

  A) [Caption 1]
  B) [Caption 2]
  C) [Caption 3]
  D) [Caption 4]

  Answer:

Dataset composition:
  - 250M+ human ratings on 2.2M+ captions from 370+ contests (contests 530-900+)
  - Test split: 47 contests, ~6,000 captions per contest
  - Rating scale: 1-3 (not funny, somewhat funny, funny)
  - Selection: Top-ranked caption (rank 0) + 3 random lower-ranked captions as distractors

Fields used: contest_number (from ratings), rank, caption (from ratings), image (from cartoons)
Fields skipped: mean, precision, votes, not_funny, somewhat_funny, funny (human rating details)

Note: Paper does not specify exact prompt wording. Using standard multiple-choice format.
      Cartoons are 571x600 average size JPEG images stored as bytes in parquet files.
      Index of cartoon in cartoons dataset corresponds to contest_number - 530 (first contest).
"""

from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image
import io
import os
import random


class NewYorkerCaptionScenario(Scenario):
    """
    New Yorker Caption Contest: Multiple-choice humor caption selection.

    Models view New Yorker cartoons and select the funniest caption from
    4 choices: the top-rated human caption + 3 lower-ranked distractors.
    Tests AI's ability to understand visual humor and caption quality.
    """

    name = "newyorker_caption"
    description = "yguooo/newyorker_caption_ranking"
    tags = ["creativity", "multimodal", "vision", "humor"]

    def __init__(self, split: str = "test", num_distractors: int = 3, seed: int = 42):
        """
        Args:
            split: Dataset split to use (train/validation/test). Default: test
            num_distractors: Number of incorrect captions to include (default: 3 for 4-way MC)
            seed: Random seed for selecting distractors
        """
        super().__init__()
        self.split = split
        self.num_distractors = num_distractors
        self.seed = seed

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load New Yorker Caption Contest dataset and create multimodal instances.

        Each instance contains:
        - Cartoon image
        - Multiple caption choices (top-ranked + distractors)
        - Correct answer is the top-ranked (rank 0) caption
        """
        # Load datasets
        print("Loading New Yorker Caption Ranking dataset...")

        # Load ratings (captions with rankings)
        ds_rating = load_dataset("yguooo/newyorker_caption_ranking", "1_rating", split=self.split)

        # Load cartoon images
        parquet_path = hf_hub_download(
            repo_id="yguooo/newyorker_caption_ranking",
            filename=f"cartoons/{self.split}-00000-of-00001.parquet",
            repo_type="dataset"
        )
        df_cartoons = pd.read_parquet(parquet_path)

        # Load descriptions to get contest numbers
        ds_desc = load_dataset("yguooo/newyorker_caption_ranking", "2_gpt4o_description", split=self.split)

        # Group captions by contest
        from collections import defaultdict
        contest_captions = defaultdict(list)
        for item in ds_rating:
            contest_captions[item['contest_number']].append({
                'rank': item['rank'],
                'caption': item['caption'],
                'mean': item['mean']
            })

        # Sort captions by rank within each contest
        for contest_num in contest_captions:
            contest_captions[contest_num].sort(key=lambda x: x['rank'])

        print(f"Found {len(contest_captions)} contests with captions")
        print(f"Found {len(df_cartoons)} cartoon images")
        print(f"Found {len(ds_desc)} contest descriptions")

        # Create temp directory for images
        temp_dir = os.path.join(output_path, "temp_images")
        os.makedirs(temp_dir, exist_ok=True)

        instances = []
        random.seed(self.seed)

        # Create one instance per contest
        for idx in range(len(ds_desc)):
            contest_number = ds_desc[idx]['contest_number']

            # Check if we have captions for this contest
            if contest_number not in contest_captions:
                continue

            captions_list = contest_captions[contest_number]

            # Need at least num_distractors + 1 captions
            if len(captions_list) <= self.num_distractors:
                continue

            # Get top caption (rank 0)
            top_caption = captions_list[0]['caption']

            # Sample distractors from lower-ranked captions
            # Use captions ranked 10-50 to ensure they're not too bad but clearly worse than top
            distractor_pool = [c['caption'] for c in captions_list[10:min(50, len(captions_list))]]
            if len(distractor_pool) < self.num_distractors:
                # If not enough in range, use all lower-ranked captions
                distractor_pool = [c['caption'] for c in captions_list[1:]]

            distractors = random.sample(distractor_pool, min(self.num_distractors, len(distractor_pool)))

            # Combine and shuffle choices
            all_choices = [top_caption] + distractors
            correct_index = 0  # Top caption is initially at index 0

            # Shuffle choices and track correct answer
            shuffled_indices = list(range(len(all_choices)))
            random.shuffle(shuffled_indices)
            shuffled_choices = [all_choices[i] for i in shuffled_indices]
            correct_index = shuffled_indices.index(0)  # Find where top caption ended up

            # Save cartoon image to temporary file
            image_data = df_cartoons.iloc[idx]['image']
            img = Image.open(io.BytesIO(image_data['bytes']))
            temp_image_path = os.path.join(temp_dir, f"contest_{contest_number}.jpg")
            img.save(temp_image_path, format='JPEG')

            # Create multimedia content: cartoon + question + choices
            choices_text = "\n".join(
                f"{chr(65+i)}) {choice}"
                for i, choice in enumerate(shuffled_choices)
            )

            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="image/jpeg",
                    location=temp_image_path
                ),
                MediaObject(
                    content_type="text/plain",
                    text=f"\nWhich caption is funniest?\n\n{choices_text}\n\nAnswer:"
                )
            ])

            # Build references: all choices, correct one tagged
            references = []
            for i in range(len(shuffled_choices)):
                is_correct = (i == correct_index)
                references.append(
                    Reference(
                        Output(text=chr(65 + i)),  # A, B, C, D
                        tags=[CORRECT_TAG] if is_correct else []
                    )
                )

            # Create instance
            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"newyorker_{self.split}_contest{contest_number}"
                )
            )

        print(f"Created {len(instances)} instances")
        return instances
