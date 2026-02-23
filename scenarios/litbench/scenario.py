"""
HELM Scenario: LitBench

Paper: "LitBench: A Benchmark and Dataset for Reliable Evaluation of
       Creative Writing" https://arxiv.org/abs/2507.00769
Code: https://github.com/drfein/LitBench
Data: https://huggingface.co/collections/SAA-Lab/litbench-68267b5da3aafe58f9e43461

LitBench evaluates pairwise creative writing preference. Given a Reddit
r/WritingPrompts prompt and two stories, the model predicts which story
would receive more upvotes. The test set contains 2,381 debiased,
human-labeled story comparisons.

The test set on HuggingFace stores only Reddit comment IDs (for copyright
compliance). This scenario rehydrates them via the PullPush API
(api.pullpush.io), which requires no credentials.

Prompt format: Based on the paper's evaluation prompt (src/dataloader.py).

  Writing prompt: {prompt}

  Story A:
  {story_a}

  Story B:
  {story_b}

  Which story is better written and more engaging? Consider originality,
  imagery, emotional impact, coherence, and technical skill.

  A) Story A
  B) Story B
  Answer:

Fields used: chosen_comment_id, rejected_comment_id (from LitBench-Test)
  → rehydrated to: body (story text), link_id → submission title (prompt)

Position randomization: chosen/rejected are randomly assigned to A/B
  positions with a fixed seed per pair for reproducibility.
"""

import hashlib
import json
import os
import random
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

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

PULLPUSH_COMMENT_URL = "https://api.pullpush.io/reddit/comment/search"
PULLPUSH_SUBMISSION_URL = "https://api.pullpush.io/reddit/submission/search"
BATCH_SIZE = 80
REQUEST_DELAY = 1.0


def _fetch_json(url: str, retries: int = 3) -> dict:
    """Fetch JSON from URL with retries and backoff."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "HELM-LitBench/1.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def _fetch_comments(comment_ids: List[str]) -> Dict[str, dict]:
    """Fetch comments from PullPush API in batches."""
    results = {}
    for i in range(0, len(comment_ids), BATCH_SIZE):
        batch = comment_ids[i : i + BATCH_SIZE]
        ids_str = ",".join(batch)
        url = f"{PULLPUSH_COMMENT_URL}?ids={ids_str}"
        data = _fetch_json(url)
        for comment in data.get("data", []):
            results[comment["id"]] = comment
        if i + BATCH_SIZE < len(comment_ids):
            time.sleep(REQUEST_DELAY)
    return results


def _fetch_submissions(submission_ids: List[str]) -> Dict[str, dict]:
    """Fetch submissions from PullPush API in batches."""
    results = {}
    for i in range(0, len(submission_ids), BATCH_SIZE):
        batch = submission_ids[i : i + BATCH_SIZE]
        ids_str = ",".join(batch)
        url = f"{PULLPUSH_SUBMISSION_URL}?ids={ids_str}"
        data = _fetch_json(url)
        for sub in data.get("data", []):
            results[sub["id"]] = sub
        if i + BATCH_SIZE < len(submission_ids):
            time.sleep(REQUEST_DELAY)
    return results


class LitBenchScenario(Scenario):
    name = "litbench"
    description = "SAA-Lab/LitBench-Test"
    tags = ["creativity", "writing", "preference"]

    PROMPT_TEMPLATE = (
        "Writing prompt: {prompt}\n\n"
        "Story A:\n{story_a}\n\n"
        "Story B:\n{story_b}\n\n"
        "Which story is better written and more engaging? Consider "
        "originality, imagery, emotional impact, coherence, and "
        "technical skill.\n\n"
        "A) Story A\n"
        "B) Story B\n"
        "Answer:"
    )

    def _rehydrate(self, output_path: str) -> List[dict]:
        """Rehydrate test set from Reddit comment IDs via PullPush API.

        Returns list of dicts with keys: prompt, chosen_story, rejected_story.
        Caches result to disk so rehydration only runs once.
        """
        cache_path = os.path.join(output_path, "litbench_rehydrated.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

        dataset = load_dataset("SAA-Lab/LitBench-Test", split="train")

        # Collect all unique comment IDs
        all_comment_ids = set()
        for item in dataset:
            all_comment_ids.add(item["chosen_comment_id"])
            all_comment_ids.add(item["rejected_comment_id"])

        # Fetch all comments
        comments = _fetch_comments(list(all_comment_ids))

        # Collect unique submission IDs from comments
        submission_ids = set()
        for c in comments.values():
            link_id = c.get("link_id", "")
            if link_id.startswith("t3_"):
                submission_ids.add(link_id[3:])

        # Fetch all submissions (for writing prompts)
        submissions = _fetch_submissions(list(submission_ids))

        # Assemble pairs
        pairs = []
        for item in dataset:
            chosen_id = item["chosen_comment_id"]
            rejected_id = item["rejected_comment_id"]

            chosen = comments.get(chosen_id)
            rejected = comments.get(rejected_id)
            if not chosen or not rejected:
                continue

            # Get writing prompt from parent submission
            link_id = chosen.get("link_id", "")
            sub_id = link_id[3:] if link_id.startswith("t3_") else ""
            submission = submissions.get(sub_id, {})
            prompt_text = submission.get("title", "")

            pairs.append(
                {
                    "prompt": prompt_text,
                    "chosen_story": chosen["body"],
                    "rejected_story": rejected["body"],
                    "chosen_id": chosen_id,
                    "rejected_id": rejected_id,
                }
            )

        os.makedirs(output_path, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(pairs, f)

        return pairs

    def get_instances(self, output_path: str) -> List[Instance]:
        pairs = self._rehydrate(output_path)

        instances = []
        for idx, pair in enumerate(pairs):
            # Deterministic position randomization based on pair IDs
            seed = int(
                hashlib.md5(
                    f"{pair['chosen_id']}_{pair['rejected_id']}".encode()
                ).hexdigest()[:8],
                16,
            )
            rng = random.Random(seed)
            chosen_is_a = rng.random() < 0.5

            if chosen_is_a:
                story_a = pair["chosen_story"]
                story_b = pair["rejected_story"]
                correct_letter = "A"
            else:
                story_a = pair["rejected_story"]
                story_b = pair["chosen_story"]
                correct_letter = "B"

            prompt_text = self.PROMPT_TEMPLATE.format(
                prompt=pair["prompt"],
                story_a=story_a,
                story_b=story_b,
            )

            references = [
                Reference(
                    Output(text="A"),
                    tags=[CORRECT_TAG] if correct_letter == "A" else [],
                ),
                Reference(
                    Output(text="B"),
                    tags=[CORRECT_TAG] if correct_letter == "B" else [],
                ),
            ]

            instances.append(
                Instance(
                    input=Input(text=prompt_text),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"litbench_{idx}",
                )
            )

        return instances
