"""
HELM Scenario: SONNET_OR_NOT_BOT (Poetry Form Recognition)

Paper: Sonnet or Not, Bot? Poetry Evaluation for Large Models and Datasets
       Walsh, Preus, Antoniak. EMNLP Findings 2024
       https://arxiv.org/abs/2406.18906

Code: https://github.com/maria-antoniak/poetry-eval

Task: Given a poem text, identify its poetic form from a list of possible forms within
      the same form group. This tests the model's understanding of poetry structure,
      meter, rhyme schemes, and other formal poetic devices.

Prompt format (from paper's evaluation scripts):
  Read the following poem and then choose the form of the poem from one of these
  possible {form_group}: {possible_forms}. All of the poems have been tagged by
  experts as one of these forms. You must pick one of these options.

  Poem Text (in full): {poem_text}

  Pick ONE of these possible {form_group}: {possible_forms}

Dataset: 1,453 public domain poems with expert form annotations
         - 585 sonnets, 302 couplets, 73 blank verse, 69 common measure, etc.
         - 4 form groups: verse forms (678), stanza forms (315), types/modes (261), meters (199)
         - Sources: Poetry Foundation, Academy of American Poets

Fields used: poem_text, form, form_group
Evaluation: Classification accuracy (exact match on form name)
"""

import os
import urllib.request
from typing import List

import pandas as pd
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Reference,
    Scenario,
)


class SonnetOrNotBotScenario(Scenario):
    """
    Poetry form recognition benchmark from "Sonnet or Not, Bot?"

    Models must identify the correct poetic form (e.g., sonnet, ballad, haiku) from a
    list of possible forms in the same category (form_group).
    """

    name = "sonnet_or_not_bot"
    description = "maria-antoniak/poetry-eval (GitHub)"
    tags = ["creativity", "poetry", "literary_analysis", "classification"]

    # Form groupings from the paper's evaluation methodology
    FORM_GROUPS = {
        "verse forms": [
            "sonnet",
            "blank verse",
            "free verse",
            "prose poem",
            "villanelle",
            "sestina",
            "ghazal",
            "pantoum",
        ],
        "stanza forms": [
            "couplet",
            "quatrain",
            "terza rima",
            "ottava rima",
            "ballad stanza",
        ],
        "types/modes": [
            "elegy",
            "ballad",
            "dramatic monologue",
            "ode",
            "pastoral",
            "ekphrasis",
            "ars poetica",
            "aubade",
            "epistle",
            "epithalamium",
        ],
        "meters": [
            "common measure",
            "long measure",
            "short measure",
            "heroic couplet",
        ],
    }

    def __init__(self, form_group: str = "all"):
        """
        Args:
            form_group: Which form group to evaluate on. Options:
                       "all" (default), "verse forms", "stanza forms", "types/modes", "meters"
        """
        super().__init__()
        if form_group != "all" and form_group not in self.FORM_GROUPS:
            raise ValueError(
                f"Invalid form_group: {form_group}. "
                f"Must be 'all' or one of {list(self.FORM_GROUPS.keys())}"
            )
        self.form_group = form_group

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load poetry dataset and create classification instances."""

        # Download dataset from GitHub if not already present
        data_dir = os.path.join(output_path, "data")
        os.makedirs(data_dir, exist_ok=True)

        csv_path = os.path.join(data_dir, "poetry-evaluation_public-domain-poems.csv")

        if not os.path.exists(csv_path):
            url = "https://raw.githubusercontent.com/maria-antoniak/poetry-eval/main/data/poetry-evaluation_public-domain-poems.csv"
            print(f"Downloading dataset from {url}...")
            urllib.request.urlretrieve(url, csv_path)

        # Load dataset
        df = pd.read_csv(csv_path)

        # Build form group mapping from actual data
        form_to_group = {}
        for form_group_name, forms in self.FORM_GROUPS.items():
            for form in forms:
                form_to_group[form] = form_group_name

        # Also map forms from actual data to their annotated form_group
        for _, row in df.iterrows():
            if pd.notna(row["form"]) and pd.notna(row["form_group"]):
                form_to_group[row["form"]] = row["form_group"]

        # Filter by form_group if specified
        if self.form_group != "all":
            # Get all forms in this group from actual data
            forms_in_group = df[df["form_group"] == self.form_group]["form"].unique()
            df = df[df["form"].isin(forms_in_group)]

        instances = []

        # Group poems by their form_group to create sensible multiple-choice options
        for group_name, group_df in df.groupby("form_group"):
            # Get unique forms in this group
            possible_forms = sorted(group_df["form"].unique().tolist())

            # Need at least 2 forms for meaningful MC task
            if len(possible_forms) < 2:
                continue

            for _, row in group_df.iterrows():
                # Skip if essential fields are missing
                if pd.isna(row["poem_text"]) or pd.isna(row["form"]):
                    continue

                # Truncate very long poems (paper uses 5000 char limit)
                poem_text = str(row["poem_text"]).strip()[:5000]

                # Build prompt following paper's format
                prompt = (
                    f"Read the following poem and then choose the form of the poem from one of these "
                    f"possible {group_name}: {possible_forms}. All of the poems have been tagged by "
                    f"experts as one of these forms. You must pick one of these options.\n\n"
                    f"Poem Text (in full): {poem_text}\n\n"
                    f"Pick ONE of these possible {group_name}: {possible_forms}"
                )

                # Create references for all possible forms (MC format)
                references = []
                for form in possible_forms:
                    tags = [CORRECT_TAG] if form == row["form"] else []
                    references.append(Reference(Output(text=form), tags=tags))

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=TEST_SPLIT,
                    )
                )

        return instances
