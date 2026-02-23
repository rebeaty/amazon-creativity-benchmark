"""
HELM Scenario: ROBOTOOLBENCH — Robotic Tool Design and Action Planning

Paper: "VLMgineer: Vision Language Models as Robotic Toolsmiths"
       (arXiv:2507.12644, ICLR 2026)
Website: https://vlmgineer.github.io/
Authors: George Jiayuan Gao, Tianyu Li, Junyao Shi, Yihan Li, Zizhe Zhang,
         Nadia Figueroa, Dinesh Jayaraman (University of Pennsylvania)

Task: Given a robotic manipulation task description, design a custom tool
attachment for a Franka Panda robot (output as URDF XML) and specify an
action sequence (output as a waypoint array). Tests creative engineering
problem-solving: the model must invent a physical tool that solves a novel
manipulation challenge.

This adapts VLMgineer's RoboToolBench for text-based LLM evaluation.
The original framework evaluates tool + action pairs in PyBullet physics
simulation; this scenario tests the creative generation step (design +
planning) independently of simulation execution.

Prompt (adapted from VLMgineer system prompt, paper Section 3):
  "You are a robotics hardware and controls expert. You operate with boldness
   and brilliance in the physical realm.
   Task: {task_description}
   Robot: Franka Panda 7-DOF arm
   Design a custom URDF tool attachment and action sequence to solve the task.
   ..."

System prompt source: VLMgineer paper Section 3 ("Mission Introduction")
Task descriptions: Verbatim from paper Table 1 / Appendix task specifications
Fields used:   12 task descriptions (generated in-code; no download required)
Fields skipped: N/A
Evaluation: custom (simulation-based reward requires PyBullet; see metric_notes.md)
  Tier 1 (HELM-computable): valid XML/URDF structure, correct array dimensions
  Tier 2 (simulation required): task reward in PyBullet (0–1 normalized)
Dataset: 12 instances — one per RoboToolBench task
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

# 12 RoboToolBench tasks — names and descriptions verbatim from paper Table 1
# and task environment specifications (arXiv:2507.12644)
_TASKS = [
    {
        "name": "BringCube",
        "description": (
            "A red cube is placed on a desk at an unreachable location for the "
            "robot arm. Move the red cube closer to the robot so it reaches the "
            "target zone."
        ),
        "initial_state": "Red cube on desk surface, positioned out of the robot's reach.",
        "reward": "Normalized distance progress of the cube toward the target zone.",
    },
    {
        "name": "CleanTable",
        "description": (
            "Several colorful cubes are scattered across a desk, representing "
            "dust or debris. Push all cubes into a circular green target boundary "
            "to clean the table."
        ),
        "initial_state": "Multiple colored cubes scattered on a desk; green circular target boundary on the surface.",
        "reward": "Average normalized progress of each cube toward the circular target.",
    },
    {
        "name": "DislodgeCube",
        "description": (
            "A red cube is trapped inside a transparent vertical pipe with an "
            "exit at both the top and bottom. Extract the cube through either "
            "exit without damaging the pipe."
        ),
        "initial_state": "Red cube inside a transparent white pipe; two open exits at top and bottom.",
        "reward": "Best normalized distance of the cube to the nearest pipe exit.",
    },
    {
        "name": "ElevatePlate",
        "description": (
            "A white plate is resting flat on a desk. Lift the plate to a "
            "specified target height above the desk surface."
        ),
        "initial_state": "White plate lying flat on a desk surface.",
        "reward": "Normalized vertical displacement of the plate toward the target height.",
    },
    {
        "name": "GatherSpheres",
        "description": (
            "Multiple purple spheres are sitting inside a three-walled container "
            "(open on one side). Collect the spheres and elevate them above "
            "0.3 meters from the ground."
        ),
        "initial_state": "Purple spheres in a three-walled container on a desk.",
        "reward": "Average normalized height of each sphere, capped at the 0.3m target.",
    },
    {
        "name": "HighObject",
        "description": (
            "A green cube is sitting on a high shelf above the desk level. "
            "An empty box is positioned below on the desk. Retrieve the green "
            "cube from the shelf and place it inside the box."
        ),
        "initial_state": "Green cube on upper shelf; empty box on desk below the shelf.",
        "reward": "Combined score: in-box confirmation + distance signal + descent bonus.",
    },
    {
        "name": "LiftBox",
        "description": (
            "A brown box is sitting on a desk surface. Elevate the box to above "
            "0.25 meters from the ground."
        ),
        "initial_state": "Brown box resting on a flat desk surface.",
        "reward": "Normalized progress of the box toward the 0.25m height threshold.",
    },
    {
        "name": "MoveBall",
        "description": (
            "A red ball is positioned on the left side of the desk workspace. "
            "Translate the ball from the left side to the right side of the desk."
        ),
        "initial_state": "Red ball on the left side of a desk surface.",
        "reward": "Balanced rightward progress of the ball with speed modulation.",
    },
    {
        "name": "OneBook",
        "description": (
            "Five books are stacked between two vertical holders. Extract only "
            "the middle book (the 3rd from the bottom) from the stack without "
            "toppling the remaining books."
        ),
        "initial_state": "Five books standing upright between two book-end holders.",
        "reward": "Balanced extraction success of the 3rd book + stability of remaining books.",
    },
    {
        "name": "ScoreGoal",
        "description": (
            "A hockey puck and a 3D goal structure are on the ground, positioned "
            "far from the robot. Move the puck into the goal's bounding box."
        ),
        "initial_state": "Hockey puck and 3D goal frame on the ground; robot is distant from both.",
        "reward": "Full credit for puck inside goal bounding box; linear scaling for approach progress.",
    },
    {
        "name": "SnatchCookie",
        "description": (
            "A transparent jar on the desk contains several cookies. Extract at "
            "least one cookie from the jar."
        ),
        "initial_state": "Transparent jar with cookies on a desk surface.",
        "reward": "Extraction verification of at least one cookie + height-based partial credit.",
    },
    {
        "name": "TurkeyLegs",
        "description": (
            "A silver cooking pot on a desk contains turkey legs, and an empty "
            "box is nearby. Transfer all turkey legs from the pot into the box "
            "while keeping the pot stationary on the desk."
        ),
        "initial_state": "Silver pot with turkey legs and empty box on a desk surface.",
        "reward": "Combined pot-stability score + turkey leg transfer verification.",
    },
]

# System prompt verbatim from VLMgineer paper Section 3 ("Mission Introduction")
_SYSTEM_PROMPT = (
    "You are a robotics hardware and controls expert operating a Franka Panda "
    "7-DOF robot arm. You operate with boldness and brilliance in the physical "
    "realm, designing novel tools and action plans to solve manipulation challenges."
)

_TOOL_CONSTRAINTS = """\
Tool Design Constraints:
- Use only 3D rectangular box components (no spheres, cylinders, or meshes)
- Each tool component must weigh no more than a few grams
- Attach the tool to either: (a) the panda_virtual link for non-gripper tools,
  or (b) the gripper finger links (panda_leftfinger / panda_rightfinger)
- Components must attach geometrically directly to their parent link (no gaps)
- Output the tool as complete URDF XML"""

_ACTION_CONSTRAINTS = """\
Action Sequence Format:
- Output as a NumPy array of shape N×7: [x, y, z, roll, pitch, yaw, gripper]
- Positions (x, y, z) in meters relative to robot base frame
- Orientations (roll, pitch, yaw) in radians
- Gripper state: 0 = open, 1 = closed
- Include enough waypoints to fully complete the task"""

_PROMPT_TEMPLATE = """\
{system}

Task: {name}
Description: {description}
Initial State: {initial_state}
Success Criterion: {reward}

{tool_constraints}

{action_constraints}

Provide your complete solution with:
1. Tool design rationale (2–3 sentences explaining why this tool solves the task)
2. Full URDF XML for the tool attachment
3. Action sequence array (with brief comments explaining each phase)"""


class RoboToolBenchScenario(Scenario):
    """
    RoboToolBench — creative engineering design for robotic manipulation.

    12 tasks requiring the model to invent a novel URDF tool attachment and
    action sequence for a Franka Panda robot. Tests creative problem-solving:
    given a manipulation challenge, design a physical tool from scratch.

    Evaluation requires PyBullet simulation for task reward; URDF validity and
    array format can be checked automatically. See metric_notes.md.
    """

    name = "robotoolbench"
    description = "arXiv:2507.12644 (VLMgineer / RoboToolBench, UPenn / ICLR 2026)"
    tags = ["creativity", "engineering", "code_generation", "robotics", "open_ended_generation"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances = []

        for task in _TASKS:
            prompt = _PROMPT_TEMPLATE.format(
                system=_SYSTEM_PROMPT,
                name=task["name"],
                description=task["description"],
                initial_state=task["initial_state"],
                reward=task["success_criterion"] if "success_criterion" in task else task["reward"],
                tool_constraints=_TOOL_CONSTRAINTS,
                action_constraints=_ACTION_CONSTRAINTS,
            )

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],   # No gold reference; evaluated by simulation pipeline
                    split=TEST_SPLIT,
                )
            )

        return instances  # 12 instances, one per RoboToolBench task
