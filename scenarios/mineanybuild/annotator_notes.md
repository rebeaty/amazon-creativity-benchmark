# Annotator Requirements: MineAnyBuild

Source: mineanybuild/evaluator.py in https://github.com/MineAnyBuild/MineAnyBuild

## Configuration for LLM-as-Judge

Judge model: GPT-4.1 (paper's choice)
Tasks requiring judge: Creativity, Spatial Planning, Spatial Understanding,
                       Spatial Commonsense

## Creativity Task — Critic Prompt

System: "You are an expert Minecraft builder in a flat Minecraft Java 1.20.4
server and an expert architecture critic."

User prompt:
```
Give a grade from 1 to 10 to the following Minecraft architectures from
different views. You should give the grade based on how well they are
presented and correspond together to the building instructions in the
following aspects:

- Creativity: from *boring, dull*(1) to *mediocre, normal*(5) and
  *blue sky thinking, inspiring*(10).
- Completeness: from *nothing, abandoned*(1) to *partial, incomplete*(5)
  and *masterfully completed, perfectly realized*(10).
- Complexity: from *simplistic, basic*(1) to *straightforward, moderate*(5)
  and *challenging, hardcore*(10).
- Architecture Structure: from *boxy, rudimentary*(1) to *intuitive,
  modest*(5) and *sophisticated, intricate*(10).
- Overall Aesthetic, Atmosphere and Fidelity: from *stark, bare*(1) to
  *appealing, unusual*(5) and *epic, masterpiece*(10).
```

Output format (JSON with grade + comment per dimension):
```json
{
    "Creativity": {"grade": 6, "comment": "..."},
    "Completeness": {"grade": 5, "comment": "..."},
    "Complexity": {"grade": 6, "comment": "..."},
    "Architecture Structure": {"grade": 6, "comment": "..."},
    "Overall Aesthetic, Atmosphere and Fidelity": {"grade": 5, "comment": "..."}
}
```

Weighted score: `0.8 * Creativity + 0.05 * (Completeness + Complexity +
Architecture_Structure + Overall_Aesthetic)`

## Spatial Planning Task — Critic Prompt

Same system prompt. Reference architecture scores default to 8.

Dimensions:
- Completeness (Instruction Following): 1-10
- Complexity: 1-10
- Overall Aesthetic, Atmosphere and Fidelity: 1-10

Weighted score: `0.3 * Completeness + 0.3 * Complexity + 0.4 * Overall_Aesthetic`

Input includes both the reference ground-truth image and the model's built
architecture image.

## Spatial Understanding Task — Critic Prompt

Same system prompt. Reference architecture scores default to 10.

Single dimension:
- Instruction Following (Completeness): 1-10

Input includes reference image and model's built image.

## Spatial Commonsense Task — Critic Prompt

System: "You are an expert in the field of multi-modal large language models
and answer proofreading."

User prompt:
```
You will get the output result of a multi-modal large language model and a
standard answer. You need to compare the two and score the output result of
the MLLM. What you need to note is:
1) Evaluate the matching degree between the output result of the large model
   and the standard answer. It is not necessary for the contents of the two
   to be completely the same, but the tendency of the answers must be the
   same to be considered a correct match.
2) You need to score the matching degree, with a full score of 10. If it is
   a correct match, please score at least 8 points or more. If it is a wrong
   match, please score at least 3 points or less.
3) You need to carefully check the key information in the standard answer,
   such as spatial position and direction, action tendency, and spatial common
   sense reasoning.
```

Output format: `{"score": 5, "reason": "..."}`

Success rate: score >= 7 counts as success.

## Adaptation Notes for HELM

The original evaluation pipeline requires:
1. Parsing model output into 3D blueprint matrix
2. Building the structure in Minecraft via Mineflayer
3. Recording/screenshotting the built structure from multiple angles
4. Running GPT-4.1 critic on the screenshots

For HELM without Minecraft execution, two alternative approaches:
1. **Blueprint-only evaluation**: Use block_matching() to directly compare
   predicted and ground-truth 3D matrices (see metric_notes.md)
2. **Adapted LLM-as-judge on blueprints**: Modify the critic prompts to
   evaluate the JSON blueprint text directly instead of screenshots,
   assessing spatial coherence, material variety, and structural logic
   from the matrix representation

The creativity task is most amenable to text-only evaluation since the
model freely designs its own structure (no need to match a reference).
