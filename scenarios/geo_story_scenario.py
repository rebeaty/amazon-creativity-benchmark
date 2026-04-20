"""
HELM Scenario: Geo-Anchored Story Generation (Geographical Disparity Framework)

Paper: "Richer Output for Richer Countries: Uncovering Geographical Disparities
       in Generated Stories and Travel Recommendations"
       NAACL 2025 Findings, arXiv:2411.07320
Code: https://github.com/FLAIR-IISc/richer-countries-have-richer-output

Open-ended creative story generation anchored to real-world locations. Models
write stories/narratives about people and daily life in cities sampled from
GeoNames (population > 1000). The paper evaluates geographical disparity in
creative output quality across ~4000 locations in 245 countries.

Prompt source: src/prompt_template.json in the GitHub repo. Four story prompt
  categories (childhood, character, profession, daily_life) with 5-6 templates
  each, totaling 21 story prompts. Each prompt contains a {} placeholder
  filled with "city, country".

Subsets:
  - story (default): All 4 story prompt types combined
  - story_childhood: "What were the childhood days like for someone born in {}?"
  - story_character: "Write a story of a middle-aged local from {}" etc.
  - story_profession: "Write a story of a teacher from {}" etc.
  - story_daily_life: "Illustrate a day-to-day routine in {}" etc.
  - travel: Travel recommendation prompts (2 types, 11 templates)

Sampling: 25 locations per country (matching paper), deterministic seed=100.
  Each location gets one randomly-assigned prompt from its subset's template pool.

Fields used: GeoNames city name, country name (via countryInfo.txt)
Fields skipped: latitude, longitude, population, elevation, timezone, etc.

Eval type: custom (paper uses lexical uniqueness via IDF, informativeness via
  NER entity count, and emotion classification via GPT-4)
"""

import csv
import io
import os
import random
import urllib.request
import zipfile

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)

# Exact prompts from src/prompt_template.json in the paper's repo
_STORY_PROMPTS = {
    "childhood": [
        "What were the childhood days like for someone born in {}?",
        "Paint a picture of the childhood experiences of a person from {}",
        "What was childhood like for someone who grew up in {}",
        "Describe the childhood days of a person born in {}.",
        "How would you describe the childhood experience of someone born in {}?",
    ],
    "character": [
        "Write a story of a middle-aged local from {}",
        "Write a story of a young local from {}",
        "Write a story of an old person from {}",
        "Write a story of a child from {}",
        "Write a story of a family from {}",
        "Write a story of a couple from {}",
    ],
    "profession": [
        "Write a story of a gardener from {}",
        "Write a story of a social worker from {}",
        "Write a story of a writer from {}",
        "Write a story of a teacher from {}",
        "Write a story of a shopkeeper from {}",
    ],
    "daily_life": [
        "Illustrate a day-to-day routine in {}.",
        "What does an ordinary day look like in {}",
        "Paint a picture of a usual day in the town of {}",
        "Describe a typical day in the town of {}.",
        "Capture the essence of a typical day in {}.",
    ],
}

_TRAVEL_PROMPTS = {
    "itinerary": [
        "Help me plan a 3-day trip to {}",
        "Design a 3-day travel schedule for {}",
        "Plan a memorable 3-day getaway to {} for me.",
        "Help me create a 3-day itinerary for my visit to {}",
        "Assist me in organizing a three-day itinerary for {}",
    ],
    "attractions": [
        "What are some places to visit in {}.",
        "What are the top attractions to visit in {}",
        "Name popular tourist spots in {}",
        "Name top landmarks to visit in {}",
        "What are some places one should add in their travel plans when visiting {}.",
        "Tell me some important sites to incorporate into my travel plans to {}.",
    ],
}

_GEONAMES_CITIES_URL = "https://download.geonames.org/export/dump/cities1000.zip"
_GEONAMES_COUNTRY_URL = "https://download.geonames.org/export/dump/countryInfo.txt"

_GEONAMES_COLS = [
    "geonameid", "name", "asciiname", "alternatenames", "lat", "lon",
    "feature_class", "feature_code", "country_code", "cc2",
    "admin1", "admin2", "admin3", "admin4",
    "population", "elevation", "dem", "timezone", "mod_date",
]


class GeoStoryScenario(Scenario):
    name = "geo_story"
    description = "FLAIR-IISc/richer-countries-have-richer-output"
    tags = ["creativity", "story_generation", "geographical", "open_ended"]

    SUBSETS = [
        "story", "story_childhood", "story_character",
        "story_profession", "story_daily_life", "travel",
    ]

    def __init__(self, subset: str = "story", locations_per_country: int = 25,
                 seed: int = 100):
        """
        Args:
            subset: Prompt category. "story" uses all 4 story types combined.
            locations_per_country: Max cities sampled per country (paper: 25).
            seed: Random seed for location sampling and prompt assignment.
        """
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(
                f"subset must be one of {self.SUBSETS}, got '{subset}'"
            )
        self.subset = subset
        self.locations_per_country = locations_per_country
        self.seed = seed

    def _get_prompts(self):
        """Return the prompt template list for the selected subset."""
        if self.subset == "story":
            # All story prompts combined
            prompts = []
            for templates in _STORY_PROMPTS.values():
                prompts.extend(templates)
            return prompts
        elif self.subset.startswith("story_"):
            category = self.subset[len("story_"):]
            return _STORY_PROMPTS[category]
        elif self.subset == "travel":
            prompts = []
            for templates in _TRAVEL_PROMPTS.values():
                prompts.extend(templates)
            return prompts

    def _download_geonames(self, output_path):
        """Download and extract GeoNames cities1000 data."""
        cities_file = os.path.join(output_path, "cities1000.txt")
        if not os.path.exists(cities_file):
            zip_path = os.path.join(output_path, "cities1000.zip")
            urllib.request.urlretrieve(_GEONAMES_CITIES_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_path)
        return cities_file

    def _download_country_info(self, output_path):
        """Download GeoNames country info for code->name mapping."""
        info_file = os.path.join(output_path, "countryInfo.txt")
        if not os.path.exists(info_file):
            urllib.request.urlretrieve(_GEONAMES_COUNTRY_URL, info_file)
        # Parse, skipping comment lines
        code_to_name = {}
        with open(info_file, "r", encoding="utf-8") as f:
            header = None
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if header is None:
                    header = parts
                    continue
                if len(parts) >= 5:
                    code_to_name[parts[0]] = parts[4]
        return code_to_name

    def _load_and_sample_cities(self, cities_file, code_to_name):
        """Load cities, join with country names, sample per country."""
        cities = []
        with open(cities_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) < 19:
                    continue
                cc = row[8]
                country_name = code_to_name.get(cc)
                if not country_name:
                    continue
                cities.append({
                    "name": row[1],
                    "country_code": cc,
                    "country": country_name,
                })

        # Group by country and sample
        by_country = {}
        for c in cities:
            by_country.setdefault(c["country_code"], []).append(c)

        rng = random.Random(self.seed)
        sampled = []
        for cc in sorted(by_country.keys()):
            pool = by_country[cc]
            n = min(self.locations_per_country, len(pool))
            sampled.extend(rng.sample(pool, n))

        return sampled

    def get_instances(self, output_path):
        cities_file = self._download_geonames(output_path)
        code_to_name = self._download_country_info(output_path)
        locations = self._load_and_sample_cities(cities_file, code_to_name)
        prompts = self._get_prompts()

        rng = random.Random(self.seed)
        instances = []
        for loc in locations:
            location_str = f"{loc['name']}, {loc['country']}"
            template = rng.choice(prompts)
            prompt_text = template.format(location_str)

            instances.append(Instance(
                input=Input(text=prompt_text),
                references=[Reference(output=Output(text=""), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
                id=f"geo_story_{loc['country_code']}_{loc['name']}",
            ))

        return instances
