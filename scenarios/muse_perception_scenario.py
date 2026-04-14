"""
HELM Scenario: MuSe-Perception (Social Perception Sub-Challenge)

Paper: https://arxiv.org/abs/2406.07753
       "The MuSe 2024 Multimodal Sentiment Analysis Challenge:
        Social Perception and Humor Recognition"
Code: https://github.com/amirip/MuSe-2024
Data: LMU-ELP dataset (CEO interviews) + Passau-SFCH (football coaches)
      Access via EULA at https://www.muse-challenge.org/

MuSe 2024 has two creativity-relevant sub-challenges:

  Subset "perception" (MuSe-Perception):
    - Predict 16 social perception attributes (assertiveness, charisma,
      likability, etc.) from CEO pitch videos on 1-7 Likert scale
    - LMU-ELP dataset: 177 subjects (train: 59, dev: 58, test: 60)
    - Audio-visual recordings with face-detected frames
    - Eval: mean Pearson's correlation across 16 dimensions
    - Baseline: 0.3573 mean rho

  Subset "humor" (MuSe-Humor):
    - Binary humor detection in football coach press conferences
    - Passau-SFCH dataset: 16 coaches, cross-cultural (German train,
      English test)
    - Eval: AUC (baseline: 0.8682)

Prompt source: Standard classification/rating prompts; no specific
  prompts in paper (challenge uses pre-extracted features).

Fields used:
  Perception: subject video frames + labels.csv (35 social attributes)
  Humor: segment transcripts + humor labels per segment

Note: Dataset requires EULA signed by a professor. Data must be
      pre-downloaded to local paths. This scenario expects the MuSe 2024
      data package structure:
        c1_muse_perception/
          feature_segments/
          labels.csv
          metadata/partition.csv
        c2_muse_humor/
          feature_segments/
          label_segments/
          metadata/partition.csv
"""

import os
import glob as glob_module
import numpy as np
import pandas as pd
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT, VALID_SPLIT, TRAIN_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


# 16 challenge perception dimensions (from perception_full.sh in baseline).
# Note: config.py lists all 35 LMU-ELP attributes; the challenge uses only these 16.
PERCEPTION_LABELS = [
    "aggressive", "arrogant", "dominant", "enthusiastic", "friendly",
    "leader_like", "likeable", "assertiv", "confident", "independent",
    "risk", "sincere", "collaborative", "kind", "warm", "good_natured",
]


class MuSePerceptionScenario(Scenario):
    name = "muse_perception"
    description = "MuSe-2024/LMU-ELP"
    tags = ["creativity", "multimodal", "social_perception"]

    SUBSETS = ["perception", "humor"]

    def __init__(self, subset: str = "perception", data_path: str = ""):
        """
        Args:
            subset: "perception" for social attribute prediction,
                    "humor" for humor detection
            data_path: Path to the MuSe 2024 data package root directory.
                       Must contain c1_muse_perception/ and/or c2_muse_humor/
        """
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"subset must be one of {self.SUBSETS}, got '{subset}'")
        self.subset = subset
        self.data_path = data_path

    def _get_perception_instances(self):
        """Social perception: rate CEO on 16 attributes from video."""
        perception_path = os.path.join(self.data_path, "c1_muse_perception")
        labels_path = os.path.join(perception_path, "labels.csv")
        partition_path = os.path.join(perception_path, "metadata", "partition.csv")

        labels_df = pd.read_csv(labels_path)
        partition_df = pd.read_csv(partition_path)
        subj2part = dict(zip(
            partition_df.iloc[:, 0].astype(str),
            partition_df.iloc[:, -1]
        ))

        split_map = {"train": TRAIN_SPLIT, "devel": VALID_SPLIT, "test": TEST_SPLIT}

        instances = []
        for _, row in labels_df.iterrows():
            subj_id = str(int(row["subj_id"]))
            partition = subj2part.get(subj_id, "test")
            helm_split = split_map.get(partition, TEST_SPLIT)

            # Try to find video frames for this subject
            frames_dir = os.path.join(
                perception_path, "feature_segments", "facenet512", subj_id
            )

            media_objects = []

            # If video frames exist as extracted features, reference them
            frame_path = os.path.join(
                perception_path, "raw_videos", f"{subj_id}.mp4"
            )
            if os.path.exists(frame_path):
                media_objects.append(
                    MediaObject(content_type="video/mp4", location=frame_path)
                )

            prompt_text = (
                "Watch the video of a CEO presenting their firm to investors.\n\n"
                "Rate this person on the following social perception attributes "
                "using a scale from 1 (low) to 7 (high):\n"
                "aggressive, arrogant, dominant, enthusiastic, friendly, "
                "leader-like, likeable, assertive, confident, independent, "
                "risk-taking, sincere, collaborative, kind, warm, good-natured\n\n"
                "Provide your ratings as: attribute1: X, attribute2: Y, ..."
            )

            media_objects.append(
                MediaObject(content_type="text/plain", text=prompt_text)
            )

            multimedia_content = MultimediaObject(media_objects)

            # Build reference: all 16 attributes
            rating_parts = []
            for label in PERCEPTION_LABELS:
                if label in row:
                    rating_parts.append(f"{label}: {row[label]:.1f}")
            reference_text = ", ".join(rating_parts)

            references = [
                Reference(Output(text=reference_text), tags=[CORRECT_TAG])
            ]

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=helm_split,
                id=f"muse_perception_{subj_id}",
            ))

        return instances

    def _get_humor_instances(self):
        """Humor detection: binary humor classification from coach speech."""
        humor_path = os.path.join(self.data_path, "c2_muse_humor")
        label_dir = os.path.join(humor_path, "label_segments")
        partition_path = os.path.join(humor_path, "metadata", "partition.csv")

        partition_df = pd.read_csv(partition_path)
        subj2part = dict(zip(
            partition_df.iloc[:, 0].astype(str),
            partition_df.iloc[:, -1]
        ))

        split_map = {"train": TRAIN_SPLIT, "devel": VALID_SPLIT, "test": TEST_SPLIT}

        instances = []
        for subject_id, partition in subj2part.items():
            helm_split = split_map.get(partition, TEST_SPLIT)

            # Load label segments for this coach
            label_files = sorted(
                glob_module.glob(os.path.join(label_dir, subject_id, "*.csv"))
            )
            if not label_files:
                continue

            for label_file in label_files:
                label_df = pd.read_csv(label_file)

                for _, row in label_df.iterrows():
                    segment_id = row.get("segment_id", "")
                    humor_label = int(row.iloc[-1])

                    # Try to find transcript for this segment
                    transcript_path = os.path.join(
                        humor_path, "transcriptions", subject_id,
                        f"{segment_id}.txt"
                    )
                    transcript = ""
                    if os.path.exists(transcript_path):
                        with open(transcript_path) as f:
                            transcript = f.read().strip()

                    media_objects = []

                    # Check for video segment
                    video_path = os.path.join(
                        humor_path, "raw_videos", subject_id,
                        f"{segment_id}.mp4"
                    )
                    if os.path.exists(video_path):
                        media_objects.append(
                            MediaObject(
                                content_type="video/mp4",
                                location=video_path,
                            )
                        )

                    context = f'\nTranscript: "{transcript}"\n' if transcript else "\n"
                    prompt_text = (
                        f"A football coach is speaking at a press conference."
                        f"{context}\n"
                        f"Is this segment humorous? Answer: Yes or No."
                    )
                    media_objects.append(
                        MediaObject(content_type="text/plain", text=prompt_text)
                    )

                    multimedia_content = MultimediaObject(media_objects)

                    correct = "Yes" if humor_label == 1 else "No"
                    references = [
                        Reference(
                            Output(text="Yes"),
                            tags=[CORRECT_TAG] if humor_label == 1 else [],
                        ),
                        Reference(
                            Output(text="No"),
                            tags=[CORRECT_TAG] if humor_label == 0 else [],
                        ),
                    ]

                    instances.append(Instance(
                        input=Input(multimedia_content=multimedia_content),
                        references=references,
                        split=helm_split,
                        id=f"muse_humor_{subject_id}_{segment_id}",
                    ))

        return instances

    def get_instances(self, output_path: str):
        if not self.data_path:
            raise ValueError(
                "data_path is required. Download MuSe 2024 data via EULA at "
                "https://www.muse-challenge.org/ and provide the path to the "
                "data package root directory."
            )

        if self.subset == "perception":
            return self._get_perception_instances()
        else:
            return self._get_humor_instances()
