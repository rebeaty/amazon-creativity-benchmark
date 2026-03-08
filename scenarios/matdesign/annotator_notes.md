# Annotator Requirements: MATDESIGN

Source: ArXiv 2501.13299v2 - Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents
Repository: https://github.com/shri071/Hypothesis-Generation-for-Materials-Discovery-and-Design-Using-Goal-Driven-and-Constraint-Guided-LLM

## Overview: AccelMat Multi-Agent Framework

The MATDESIGN benchmark uses a multi-agent iterative framework called **AccelMat** for evaluation:

1. **Hypotheses Generation Agent (HGA)** - GPT-4o
2. **Three Critic Agents (CA)** - GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Flash
3. **Summarizer Agent (SA)** - GPT-4o
4. **Evaluation Agent (EA)** - OpenAI-o1-preview

## Task Format

Models receive:
- **Goal Statement**: Materials science objective (e.g., "Develop a self-healing hydrogel...")
- **Constraints**: 4-6 numbered requirements with specific quantitative/qualitative criteria

Models must generate:
- **20 Suggestions** in JSON format
- Each suggestion must include:
  * Materials: Specific materials to use
  * Methods_to_develop_the_materials_suggested: Synthesis/development methods
  * Reasoning: Explanation of how the suggestion meets goal and constraints

## Evaluation Process

### Phase 1: Multi-Agent Critique (Iterative)

Each of the three critic agents (GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Flash) independently evaluates all 20 suggestions.

**Critic Configuration:**
- Model: GPT-4o, Claude-3.5-Sonnet, or Gemini-1.5-Flash
- Temperature: 0.7
- Task: Evaluate alignment with goal statement and constraints

**Evaluation Dimensions:**
For each of the 20 suggestions, critics provide:

1. **Meets_the_goal_statement_and_satisfies_all_constraints_strictly**: YES/NO
2. **Reasoning**: Detailed explanation of the evaluation

**Example Feedback Format:**
```json
{
  "Feedback_for_suggestion_1": {
    "Meets_the_goal_statement_and_satisfies_all_constraints_strictly": "NO",
    "Reasoning": "While polyurethane provides good mechanical properties and environmental resistance, the use of hydrophobic healing agents may not be ideal for multiple healing cycles. Additionally, hydrophobic agents may not be efficiently released upon water exposure."
  },
  "Feedback_for_suggestion_2": {
    "Meets_the_goal_statement_and_satisfies_all_constraints_strictly": "YES",
    "Reasoning": "Acrylic polymer with embedded hollow fibers containing a hydrophilic healing agent meets the constraints. Acrylic polymers are durable and weather-resistant, and hollow fibers ensure continuous supply of the healing agent when exposed to moisture. This approach is also scalable via spray-painting."
  },
  ...
  "Feedback_for_suggestion_20": {
    "Meets_the_goal_statement_and_satisfies_all_constraints_strictly": "YES",
    "Reasoning": "..."
  },
  "Overall_Feedback_for_improvement": "Focus on materials that demonstrate both rapid self-healing and multiple healing cycles. Ensure healing mechanisms are triggered by simple environmental factors."
}
```

### Phase 2: Feedback Summarization

The **Summarizer Agent (GPT-4o)** consolidates feedback from all three critics:
- Identifies common concerns across critics
- Highlights suggestions that received unanimous approval
- Provides structured guidance for refinement

### Phase 3: Iterative Refinement

If any suggestions receive "NO" evaluations:
1. HGA receives consolidated feedback
2. HGA refines the 20 suggestions
3. Critics re-evaluate refined suggestions
4. Process repeats up to 5 iterations or until all suggestions receive "YES"

**Stopping Criteria:**
- All 20 suggestions receive "YES" from all critics (unanimous agreement)
- Maximum 5 refinement iterations reached

### Phase 4: Final Evaluation

The **Evaluation Agent (OpenAI-o1-preview)** performs final scoring:
- Assesses novelty and scientific soundness
- Evaluates feasibility and practicality
- Compares to reference materials/methods from original paper (for context)
- Provides aggregate quality score

## Critic Prompt Template

```
You are an expert {expert_list} capable of doing impactful materials discovery and design. Given a goal statement, additional constraints, and a list of suggestions about materials design and discovery, your task is to evaluate each suggestion such that it meets the goal statement and satisfies all the constraints strictly.

Goal Statement:
{goal_statement}

Constraints:
{constraint_list}

Suggestions:
{generated_suggestions}

Given the above goal statement, constraints and suggestions about materials design and discovery, evaluate each suggestion and generate detailed feedback which will help the suggestion generation process to generate suggestions such that they help achieve goal statement and satisfy all the constraints strictly. The detailed feedback should be in the below JSON format strictly:

{
  "Feedback_for_suggestion_1": {
    "Meets_the_goal_statement_and_satisfies_all_constraints_strictly": "YES/NO",
    "Reasoning": "..."
  },
  ...
  "Feedback_for_suggestion_20": {
    "Meets_the_goal_statement_and_satisfies_all_constraints_strictly": "YES/NO",
    "Reasoning": "..."
  },
  "Overall_Feedback_for_improvement": "..."
}
```

## Expert List Generation

Before evaluation, an expert list is generated for the specific goal:
- Model: GPT-4o
- Temperature: 0.7
- Prompt: "Generate a list of experts required to achieve the below mentioned goal: {goal_statement}. Just list the top 5 experts in the format 'Expert_1, Expert_2, Expert_3, Expert_4, Expert_5'"

Example: "Materials Scientist, Chemical Engineer, Polymer Chemist, Nanotechnology Expert, Corrosion Specialist"

## Metrics

### Primary Metrics (from AccelMat Framework)
1. **Convergence Rate**: Number of iterations needed to reach unanimous "YES"
2. **Constraint Satisfaction Rate**: Percentage of suggestions meeting all constraints
3. **Consensus Score**: Agreement level across the three critics

### Quality Metrics (from Evaluation Agent)
1. **Novelty**: How innovative are the suggestions compared to existing approaches?
2. **Feasibility**: Can the suggestions be implemented with current technology?
3. **Specificity**: Are materials and methods sufficiently detailed?
4. **Scientific Soundness**: Do the suggestions align with materials science principles?

## Implementation Notes for HELM

1. **Multi-model evaluation**: Requires orchestrating GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-Flash
2. **Iterative refinement**: Need to support multiple rounds of generation-evaluation-refinement
3. **JSON parsing**: Suggestions and feedback must be parsed as structured JSON
4. **Reference comparison**: Ground truth from papers provides context but models should generate NOVEL suggestions
5. **Expert contextualization**: Each evaluation should be contextualized with domain-specific expertise

## Example Evaluation Scenario

**Goal Statement**: "Develop a scalable extrinsic self-healing coating system for corrosion protection of metallic structures in offshore environments."

**Constraints**:
1. Self-healing triggered by simple environmental factor (e.g., water)
2. Allow multiple healing events
3. Maintain structural integrity after mechanical damage
4. Compatible with scalable application techniques
5. Single-component healing system (not multi-component)

**Suggestion Example (would receive "YES")**:
```json
{
  "Materials": "Core-shell nanofibers synthesized using coaxial electrospinning with organosilane compounds (silyl esters) as the self-healing agent. Metallic substrates (e.g., steel) for testing.",
  "Methods_to_develop_the_materials_suggested": "Use coaxial electrospinning to create core-shell nanofibers with silyl ester healing agent in the core. Develop spray-painting technique by prior dispersion of nanofibers for scalability. Incorporate water-reactive organosilane that heals upon water exposure without additional catalysts.",
  "Reasoning": "This approach satisfies all constraints: (1) water-triggered healing via organosilane hydrolysis, (2) multiple healing events from nanofiber core reservoir, (3) structural integrity maintained by shell protection, (4) scalable spray-painting application, (5) single-component organosilane system. The core-shell structure provides controlled release while the water-reactive chemistry ensures autonomous healing in offshore environments."
}
```

**Suggestion Example (would receive "NO")**:
```json
{
  "Materials": "Polyurethane coating with hydrophobic healing agents encapsulated in microcapsules.",
  "Methods_to_develop_the_materials_suggested": "Synthesize polyurethane matrix and embed microcapsules containing hydrophobic healing agents. Apply via spray coating.",
  "Reasoning": "Polyurethane provides good mechanical properties and the microcapsules can release healing agents upon damage."
}
```

**Reasoning for "NO"**: Hydrophobic agents may not be efficiently released upon water exposure (violates constraint 1), may not support multiple healing cycles effectively (constraint 2), and lacks detail on the healing mechanism specificity.

## Notes

- The multi-agent framework helps reduce individual model biases
- Iterative refinement allows models to learn from critique
- The framework was validated on 50 recent materials science papers (2024)
- Average convergence: 2-3 iterations for unanimous approval
- Dataset includes diverse materials domains: hydrogels, coatings, composites, polymers, etc.
