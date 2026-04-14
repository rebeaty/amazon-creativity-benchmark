"""
HELM Scenario: MineAnyBuild

Paper: "MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents"
       https://arxiv.org/abs/2505.20148 (NeurIPS 2025 Datasets & Benchmarks)
Code: https://github.com/MineAnyBuild/MineAnyBuild

MineAnyBuild evaluates spatial creativity and planning in Minecraft
architecture generation. Given text instructions (and optionally reference
images), models must produce 3D blueprint matrices specifying block types
and positions for constructing Minecraft buildings.

5 task types (subset parameter):
  - creativity: Text-only. Free-form creative building from short instruction
    + available block list. 1,419 instances (473 architectures x 3 verb
    variants). Creativity weighted 80% in evaluation.
  - spatial_planning: Multimodal. Build from instruction + reference image.
    946 instances (473 concrete + 473 simple).
  - spatial_understanding: Multimodal. Parse layer-by-layer coordinate text
    + reference image into blueprint. 473 instances.
  - spatial_reasoning: Multimodal VQA. Mental rotation of 3D Minecraft
    structures. 1,728 instances (exact match A/B/C/D/True/False).
  - spatial_commonsense: Multimodal QA. Spatial commonsense about room
    environments. 50 instances.

Prompt source: Exact prompts from mineanybuild/prompter.py in the official
repository. System prompts and user prompts reproduced verbatim.

Output format (blueprint tasks): JSON 3D array where dimensions are
height(y) x depth(z) x width(x), -1 = air, positive integers map to
block materials via provided dictionary.

Fields used: instruction, block_materials, image (for multimodal tasks),
             blueprint (ground truth from architectures.json),
             question/answer (spatial_reasoning),
             question/ref_ans/images/image_desp (spatial_commonsense)
Fields skipped: image_urls, metadata (source URLs), biome, type (categories)

Evaluation: See metric_notes.md (blueprint comparison) and
annotator_notes.md (LLM-as-judge scoring) for details.
"""

import json
import os
from huggingface_hub import hf_hub_download, snapshot_download
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


SYSTEM_PROMPT_BUILDER = (
    "You are an expert Minecraft builder in a flat Minecraft Java 1.20.4 "
    "server and Python coding. Your goal is to produce a Minecraft "
    "architecture, considering aspects such as architecture structure, "
    "block variety, symmetry and asymmetry, overall aesthetics, and most "
    "importantly, adherence to the platonic ideal of the requested creation."
)

SYSTEM_PROMPT_REASONING = (
    "You are an expert Minecraft builder and player in a flat Minecraft "
    "Java 1.20.4 server."
)

SYSTEM_PROMPT_COMMONSENSE = (
    "You are an expert in Minecraft and interior design, familiar with "
    "real-life common sense. "
)

BLUEPRINT_EXAMPLE = (
    'Block materials: {"oak_planks": 1}\n'
    "Instruction: build a 3*3*4 (width, length, height) wooden house "
    "layer by layer from bottom to top. "
    "Layer 1: oak_planks: [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), "
    "(2,0), (2,1), (2,2)]. "
    "Layer 2: oak_planks: [(0,0), (0,2), (1,0), (1,2), (2,0), (2,1), "
    "(2,2)]. "
    "Layer 3: oak_planks: [(0,0), (0,2), (1,0), (1,2), (2,0), (2,1), "
    "(2,2)]. "
    "Layer 4: oak_planks: [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), "
    "(2,0), (2,1), (2,2)].\n"
    "Output: '''json [[[1,1,1],[1,1,1],[1,1,1]],[[1,-1,1],[1,-1,1],"
    "[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]'''"
)

BLUEPRINT_CONSTRAINTS = (
    "IMPORTANT: You must only use blocks from the given block materials "
    "dictionary.\n"
    "IMPORTANT: You must output ONLY ONE BLUEPRINT following the example "
    "format (A 3-DIM MATRIX).\n"
    'IMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix '
    "if no block is placed in the corresponding position."
)

CREATIVITY_PROMPT = (
    "Build the architecture based on the instruction and let your "
    "imagination run wild and use your creativity to build your best "
    "architecture. Before providing your final blueprint matrix, plan "
    "your solution considering the following spatial planning "
    "perspectives:\n"
    "1. How to best interpret and implement the build specification.\n"
    "- List key elements from the build specification\n"
    "- Brainstorm block combinations for different parts of the structure\n"
    "- Outline a rough structure layout\n"
    "2. Creative ways to use the available blocks to achieve the desired "
    "aesthetic.\n"
    "3. How to ensure the mapping correctness in your blueprint matrix.\n"
    "4. Ways to maximize creativity and the dynamic range of possible "
    "builds.\n"
    "5. Consider potential challenges and solutions\n"
    "The blueprint is a three-dimension list, and the dimensions of the "
    "blueprint matrix are in the order of height(positive y), "
    "length(positive z) and width(positive x). Fill in \"-1\" into the "
    'blueprint to denote as "air". The elements of the list matrix are '
    "integers filled according to the given mapping table.\n"
    "Here is an output example.\n"
    'Block materials = ["oak_planks", "cobblestone", "red_wool"]\n'
    "Instruction: build a 3*3*4 (width, length, height) wooden house. \n"
    "Output:\n"
    "Planning Reasons: Let's build a simple wooden house. I use "
    "cobblestone as the material for the floor and oak_planks for the "
    "wall and roof. \n"
    'Selected_block_materials = {"oak_planks": 1, "cobblestone": 2}\n'
    "Blueprint: '''json [[[2,2,2],[2,2,2],[2,2,2]],[[1,-1,1],[1,-1,1],"
    "[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]'''\n"
    "IMPORTANT: You must output ONLY ONE BLUEPRINT following the example "
    "format (A 3-DIM MATRIX).\n"
    'IMPORTANT: Fill in -1(interger) as "air" into the blueprint matrix '
    "if no block is placed in the corresponding position."
)

SPATIAL_PLAN_CONCRETE_PROMPT = (
    "Build the architecture based on the instruction and reference image. "
    "The instruction divides the architecture in the reference image by "
    "structure and demands. Please analyze and plan how to build the "
    "corresponding sub-structures according to the divided structure and "
    "demands, and give the ONLY ONE OVERALL blueprint matrix of the total "
    "architecture. The blueprint is a three-dimension list, and the "
    "dimensions of the blueprint matrix are in the order of "
    "height(positive y), length(positive z) and width(positive x). "
    'Fill in "-1" into the blueprint to denote as "air". The elements '
    "of the list matrix are integers filled according to the given "
    "mapping table.\n"
    "Here is an example.\n"
    'Block materials: {"oak_planks": 1}\n'
    "Instruction: build a 3*3*4 (width, length, height) wooden house. "
    "We want it to be a simple boxy house. The roof and floor should be "
    "solid and there is some space that player can go inside the house.\n"
    "output: \n"
    "Planning: The floor and roof of this wooden house can be made of "
    "3*3 oak_planks as a square. Make the house hollow with air in the "
    "layer 2 and 3 and leave the space for entrance towards west "
    "(negative x). The two-layer walls are also made of 7*2=14 "
    "oak_planks. I can build it layer by layer so that I can truly "
    "understand the spatial structure of this house. Then, here is the "
    "overall blueprint matrix of the whole architeture. I'm sure that "
    'the 3-dim list has correct template and fill in -1 as "air" to '
    "simulate no block here.\n"
    "Blueprint:'''json [[[1,1,1],[1,1,1],[1,1,1]],[[1,-1,1],[1,-1,1],"
    "[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]'''\n"
    + BLUEPRINT_CONSTRAINTS
)

SPATIAL_PLAN_SIMPLE_PROMPT = (
    "Build the architecture based on the instruction and reference image. "
    "You should understand the structure of the reference image and output "
    "your planning(reasoning) and overall blueprint matrix. The blueprint "
    "is a three-dimension list, and the dimensions of the blueprint "
    "matrix are in the order of height(positive y), length(positive z) "
    'and width(positive x). Fill in "-1" into the blueprint to denote '
    'as "air". The elements of the list matrix are integers filled '
    "according to the given mapping table.\n"
    "Here is an example.\n"
    'Block materials: {"oak_planks": 1}\n'
    "Instruction: build a 3*3*4 (width, length, height) wooden house.\n"
    "Output: Planning: We want it to be built with solid walls on all "
    "sides for the bottom layer. Make the hollow interior with only the "
    "outer walls for middle layers and a solid roof on the top layer.\n"
    "Blueprint: '''json [[[1,1,1],[1,1,1],[1,1,1]],[[1,-1,1],[1,-1,1],"
    "[1,1,1]],[[1,-1,1],[1,-1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]'''\n"
    + BLUEPRINT_CONSTRAINTS
)

SPATIAL_UNDERSTANDING_PROMPT = (
    "Build the architecture based on the instruction and reference image. "
    "The instruction provides the relative coordinates of every block in "
    "reference image, and please identify the structure and summarize it "
    "as an overall blueprint matrix. The blueprint is a three-dimension "
    "list, and the dimensions of the blueprint matrix are in the order of "
    "height(positive y), length(positive z) and width(positive x). "
    'Fill in "-1" into the blueprint to denote as "air". The elements '
    "of the list matrix are integers filled according to the given "
    "mapping table.\n"
    "Here is an example.\n"
    + BLUEPRINT_EXAMPLE + "\n"
    + BLUEPRINT_CONSTRAINTS
)


def _build_block_types(block_materials):
    """Convert block_materials list to {name: id} mapping dict."""
    return {
        **{"air": -1},
        **{block_materials[i]: (i + 1) for i in range(len(block_materials))}
    }


class MineAnyBuildScenario(Scenario):
    name = "mineanybuild"
    description = "SaDil/MineAnyBuild"
    tags = ["creativity", "design", "spatial", "multimodal"]

    SUBSETS = [
        "creativity", "spatial_planning", "spatial_understanding",
        "spatial_reasoning", "spatial_commonsense",
    ]

    TASK_FILES = {
        "creativity": "task/Task_Creativity.json",
        "spatial_planning": "task/Task_Spatial_Planning.json",
        "spatial_understanding": "task/Task_Spatial_Understanding.json",
        "spatial_reasoning": "task/Task_Spatial_Reasoning.json",
        "spatial_commonsense": "task/Task_Spatial_Commonsense.json",
    }

    def __init__(self, subset="creativity"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(
                f"Unknown subset '{subset}'. Must be one of {self.SUBSETS}"
            )
        self.subset = subset

    def _download_data(self, output_path):
        """Download JSON task files and images from HuggingFace."""
        data_dir = os.path.join(output_path, "MineAnyBuild")

        # Always need architectures.json
        arch_path = os.path.join(data_dir, "architectures.json")
        if not os.path.exists(arch_path):
            hf_hub_download(
                "SaDil/MineAnyBuild", "architectures.json",
                repo_type="dataset", local_dir=data_dir,
            )

        # Download task file
        task_file = self.TASK_FILES[self.subset]
        task_path = os.path.join(data_dir, task_file)
        if not os.path.exists(task_path):
            hf_hub_download(
                "SaDil/MineAnyBuild", task_file,
                repo_type="dataset", local_dir=data_dir,
            )

        # Download images for multimodal tasks
        if self.subset != "creativity":
            img_dir = os.path.join(data_dir, "data", "images")
            if not os.path.exists(img_dir):
                snapshot_download(
                    "SaDil/MineAnyBuild", repo_type="dataset",
                    local_dir=data_dir, allow_patterns=["data/images/**"],
                )

        return data_dir

    def _resolve_image(self, data_dir, image_path):
        """Resolve image path from dataset-relative to absolute."""
        # Paths in dataset use ./images/ or /data/images/ prefix
        clean = image_path.lstrip("./").lstrip("/")
        # Try data/ prefix first (task files use /data/images/...)
        candidate = os.path.join(data_dir, clean)
        if os.path.exists(candidate):
            return candidate
        # Try with data/ prefix added
        candidate = os.path.join(data_dir, "data", clean)
        if os.path.exists(candidate):
            return candidate
        # Try images/ directly under data_dir
        candidate = os.path.join(data_dir, "data", "images",
                                 os.path.basename(image_path))
        if os.path.exists(candidate):
            return candidate
        return os.path.join(data_dir, clean)

    def _get_creativity_instances(self, data_dir):
        """Creativity task: text-only, no reference image."""
        with open(os.path.join(data_dir, self.TASK_FILES["creativity"])) as f:
            tasks = json.load(f)
        with open(os.path.join(data_dir, "architectures.json")) as f:
            arch_list = json.load(f)
        arch_map = {a["id"]: a for a in arch_list}

        instances = []
        for task in tasks:
            arch = arch_map.get(task["AR_id"])
            block_materials = task["block_materials"]
            instruction = task["instruction"]

            prompt = (
                SYSTEM_PROMPT_BUILDER + "\n\n"
                + CREATIVITY_PROMPT + "\n\n"
                "Now, take a breath and continue.\n"
                f"Block materials: {json.dumps(block_materials)}\n"
                f"Instruction: {instruction}\n"
                "Output: "
            )

            # Ground truth blueprint from architectures.json
            gt_blueprint = json.dumps(arch["blueprint"]) if arch else ""

            references = []
            if gt_blueprint:
                references.append(Reference(
                    Output(text=gt_blueprint), tags=[CORRECT_TAG],
                ))

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=task["id"],
            ))
        return instances

    def _get_spatial_planning_instances(self, data_dir):
        """Spatial planning: multimodal (reference image + text)."""
        with open(os.path.join(
                data_dir, self.TASK_FILES["spatial_planning"])) as f:
            tasks = json.load(f)
        with open(os.path.join(data_dir, "architectures.json")) as f:
            arch_list = json.load(f)
        arch_map = {a["id"]: a for a in arch_list}

        instances = []
        for task in tasks:
            arch = arch_map.get(task["AR_id"])
            block_materials = task["block_materials"]
            block_types = _build_block_types(block_materials)
            instruction = task["instruction"]
            is_concrete = task["id"].endswith("_0")

            if is_concrete:
                task_prompt = SPATIAL_PLAN_CONCRETE_PROMPT
            else:
                task_prompt = SPATIAL_PLAN_SIMPLE_PROMPT

            prompt_text = (
                SYSTEM_PROMPT_BUILDER + "\n\n"
                + task_prompt + "\n\n"
                "Now, take a breath and continue.\n"
                f"Block materials: {json.dumps(block_types)}\n"
                f"Instruction: {instruction}\n"
                "Reference image: "
            )

            image_path = self._resolve_image(data_dir, task["image"])

            multimedia = MultimediaObject([
                MediaObject(content_type="text/plain", text=prompt_text),
                MediaObject(content_type="image/png", location=image_path),
                MediaObject(content_type="text/plain", text="Output: "),
            ])

            gt_blueprint = json.dumps(arch["blueprint"]) if arch else ""
            references = []
            if gt_blueprint:
                references.append(Reference(
                    Output(text=gt_blueprint), tags=[CORRECT_TAG],
                ))

            instances.append(Instance(
                input=Input(multimedia_content=multimedia),
                references=references,
                split=TEST_SPLIT,
                id=task["id"],
            ))
        return instances

    def _get_spatial_understanding_instances(self, data_dir):
        """Spatial understanding: multimodal (image + coordinate text)."""
        with open(os.path.join(
                data_dir, self.TASK_FILES["spatial_understanding"])) as f:
            tasks = json.load(f)
        with open(os.path.join(data_dir, "architectures.json")) as f:
            arch_list = json.load(f)
        arch_map = {a["id"]: a for a in arch_list}

        instances = []
        for task in tasks:
            arch = arch_map.get(task["AR_id"])
            block_materials = task["block_materials"]
            block_types = _build_block_types(block_materials)
            instruction = task["instruction"]

            prompt_text = (
                SYSTEM_PROMPT_BUILDER + "\n\n"
                + SPATIAL_UNDERSTANDING_PROMPT + "\n\n"
                "Now, take a breath and continue.\n"
                f"Block materials: {json.dumps(block_types)}\n"
                f"Instruction: {instruction}\n"
                "Reference image: "
            )

            image_path = self._resolve_image(data_dir, task["image"])

            multimedia = MultimediaObject([
                MediaObject(content_type="text/plain", text=prompt_text),
                MediaObject(content_type="image/png", location=image_path),
                MediaObject(content_type="text/plain", text="Output: "),
            ])

            gt_blueprint = json.dumps(arch["blueprint"]) if arch else ""
            references = []
            if gt_blueprint:
                references.append(Reference(
                    Output(text=gt_blueprint), tags=[CORRECT_TAG],
                ))

            instances.append(Instance(
                input=Input(multimedia_content=multimedia),
                references=references,
                split=TEST_SPLIT,
                id=task["id"],
            ))
        return instances

    def _get_spatial_reasoning_instances(self, data_dir):
        """Spatial reasoning: multimodal VQA with exact match."""
        with open(os.path.join(
                data_dir, self.TASK_FILES["spatial_reasoning"])) as f:
            tasks = json.load(f)

        instances = []
        for task in tasks:
            question = task["question"]
            answer = task["answer"]
            # Fix known typo in ground truth
            if answer == "Ture":
                answer = "True"

            prompt_text = (
                SYSTEM_PROMPT_REASONING + "\n\n"
                "You need to answer this question with a visual image.\n"
                f"Question: {question}\n"
            )

            image_path = self._resolve_image(
                data_dir, task["options_image"])

            suffix_text = (
                "You must output ONLY one option (from A,B,C,D,True,False) "
                "without any reason. IMPORTANT: You can only answer one "
                "letter or one word (from A,B,C,D,True,False). Your answer:"
            )

            multimedia = MultimediaObject([
                MediaObject(content_type="text/plain", text=prompt_text),
                MediaObject(content_type="image/jpeg", location=image_path),
                MediaObject(content_type="text/plain", text=suffix_text),
            ])

            references = [Reference(
                Output(text=answer), tags=[CORRECT_TAG],
            )]

            instances.append(Instance(
                input=Input(multimedia_content=multimedia),
                references=references,
                split=TEST_SPLIT,
                id=task["id"],
            ))
        return instances

    def _get_spatial_commonsense_instances(self, data_dir):
        """Spatial commonsense: multimodal QA."""
        with open(os.path.join(
                data_dir, self.TASK_FILES["spatial_commonsense"])) as f:
            tasks = json.load(f)

        instances = []
        for task in tasks:
            question = task["question"]
            ref_ans = task["ref_ans"]
            images = task["images"]
            image_desps = task["image_desp"]

            prompt_text = (
                SYSTEM_PROMPT_COMMONSENSE + "\n\n"
                "You will receive a question and an image. Answer the "
                "question based on the image, focusing on spatial "
                "commonsense. Your response must not exceed 70 words. "
                "Do not include any additional content or thoughts. "
                "Now, take a breath and continue.\n"
                f"Instruction: {question}\n"
            )

            media_objects = [
                MediaObject(content_type="text/plain", text=prompt_text),
            ]

            for i, img in enumerate(images):
                img_path = self._resolve_image(data_dir, img)
                desp = image_desps[i] if i < len(image_desps) else ""
                media_objects.append(MediaObject(
                    content_type="text/plain",
                    text=f"The next image is {desp}",
                ))
                media_objects.append(MediaObject(
                    content_type="image/jpeg", location=img_path,
                ))

            media_objects.append(MediaObject(
                content_type="text/plain", text="\nYour answer: ",
            ))

            multimedia = MultimediaObject(media_objects)

            references = [Reference(
                Output(text=ref_ans), tags=[CORRECT_TAG],
            )]

            instances.append(Instance(
                input=Input(multimedia_content=multimedia),
                references=references,
                split=TEST_SPLIT,
                id=task["id"],
            ))
        return instances

    def get_instances(self, output_path):
        data_dir = self._download_data(output_path)

        dispatch = {
            "creativity": self._get_creativity_instances,
            "spatial_planning": self._get_spatial_planning_instances,
            "spatial_understanding": self._get_spatial_understanding_instances,
            "spatial_reasoning": self._get_spatial_reasoning_instances,
            "spatial_commonsense": self._get_spatial_commonsense_instances,
        }

        return dispatch[self.subset](data_dir)
