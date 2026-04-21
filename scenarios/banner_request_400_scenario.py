"""
HELM Scenario: BannerRequest400

Paper: "BannerAgency: Advertising Banner Design with Multimodal LLM Agents"
       https://arxiv.org/abs/2503.11060 (EMNLP 2025 Main)
Code: https://github.com/sony/BannerAgency

BannerRequest400 evaluates creative banner ad design capability. Given an
advertiser's logo image and a detailed banner request (target audience,
purpose, design specifications), models must generate a JSON foreground
design blueprint specifying element positions, typography, colors, and
copywriting for a 300x250 banner.

Prompt source: Adapted from prompts/foreground_designer_prompt.py in the
repository. The prompt asks the model to produce a structured JSON layout
blueprint. The original system uses this as the Foreground Designer role
in a multi-agent pipeline.

Fields used: logo_name, logo_description, concrete_request_300x250,
             target_audience, primary_purpose, advertiser, banner_request
Fields skipped: concrete_request_{other sizes} (12 other IAB dimensions),
                logos_svg/ (SVG versions of logos)

Evaluation: LLM-as-judge across 6 dimensions (TAA, LPS, AQS, CTAE, CPYQ,
BIS) on a 1-5 scale. See annotator_notes.md for full rubrics from eval.py.
"""

import json
import os
import urllib.request
import zipfile
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class BannerRequest400Scenario(Scenario):
    name = "banner_request_400"
    description = "sony/BannerAgency"
    tags = ["creativity", "design", "layout", "multimodal"]

    REPO_URL = "https://github.com/sony/BannerAgency/archive/refs/heads/main.zip"

    TASK_PROMPT = (
        "You are an expert banner ad designer. Given the advertiser's logo "
        "and a banner design request, create a foreground design blueprint "
        "for a 300x250 banner ad.\n\n"
        "Your blueprint must be a JSON object specifying all foreground "
        "elements (headline, subheadline, CTA button, logo placement, and "
        "any decorative elements). For each element, specify:\n"
        '- "type": element type (e.g., "headline", "subheadline", "cta_button", "logo")\n'
        '- "text": the copywriting text (for text elements)\n'
        '- "position": {{"x": pixels_from_left, "y": pixels_from_top}}\n'
        '- "size": {{"width": pixels, "height": pixels}}\n'
        '- "font": font name and size (for text elements)\n'
        '- "color": hex color code\n'
        '- "background_color": hex color (for buttons)\n\n'
        "Design guidelines:\n"
        "- Canvas: 300x250 pixels\n"
        "- Logo: 15-20% of banner width, minimum 16px clear space\n"
        "- Typography: Headline 24-32px, Subheadline 18-24px, CTA 16-18px\n"
        "- Aim for 70-90% spatial utilization\n"
        "- Maintain 4.5:1 contrast ratio for text readability\n"
        "- Reserve at least 10px from edges\n\n"
        "Advertiser: {advertiser}\n"
        "Logo description: {logo_description}\n\n"
        "Banner request: {banner_request}\n\n"
        "Target audience: {target_audience}\n"
        "Primary purpose: {primary_purpose}\n\n"
        "Output your design blueprint as a JSON object with an "
        '"elements" array.'
    )

    def _download_data(self, output_path: str) -> str:
        """Download and extract the BannerAgency repository."""
        data_dir = os.path.join(output_path, "BannerAgency-main", "BannerRequest400")
        if os.path.exists(data_dir):
            return data_dir

        # Also check a shared cache so multiple runs don't re-download the 16MB zip.
        shared_cache = os.path.join("benchmark_output", "scenarios", "banner_request_400")
        shared_data = os.path.join(shared_cache, "BannerAgency-main", "BannerRequest400")
        if os.path.exists(shared_data):
            return shared_data

        zip_path = os.path.join(output_path, "BannerAgency.zip")
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(self.REPO_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)

        return data_dir

    def get_instances(self, output_path: str):
        data_dir = self._download_data(output_path)

        # Load concrete requests
        with open(os.path.join(data_dir, "concrete_5k.json")) as f:
            advertisers = json.load(f)

        logo_dir = os.path.join(data_dir, "logos_png")

        instances = []
        for adv in advertisers:
            logo_path = os.path.join(logo_dir, adv["logo_name"])

            for pair_key in ["pair_1", "pair_2", "pair_3", "pair_4"]:
                pair = adv["advertising_variations"][pair_key]

                prompt_text = self.TASK_PROMPT.format(
                    advertiser=adv["advertiser"],
                    logo_description=adv["logo_description"],
                    banner_request=pair["concrete_request_300x250"],
                    target_audience=pair["target_audience"],
                    primary_purpose=pair["primary_purpose"],
                )

                # Multimodal: logo image + text prompt
                multimedia_content = MultimediaObject([
                    MediaObject(
                        content_type="text/plain",
                        text=prompt_text,
                    ),
                    MediaObject(
                        content_type="image/png",
                        location=logo_path,
                    ),
                ])

                instance_id = f"{adv['id']:03d}_{pair_key}"

                # Open-ended generation: no gold reference
                instances.append(Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=[],
                    split=TEST_SPLIT,
                    id=instance_id,
                ))

        return instances
