# Annotator Notes: Arena-Hard v2.0 Creative Writing

Source: arXiv:2406.11939; https://github.com/lmarena/arena-hard-auto

## Original Evaluation Method

Arena-Hard uses **automated pairwise comparison**:
- Each model response is compared head-to-head against a reference model baseline
- Judges: GPT-4.1 and Gemini-2.5 (upgraded in v2.0 from GPT-4-Turbo in v1)
- Win/tie/loss counted → ELO-style win rate computed
- Separability metric measures how well the benchmark distinguishes models

## LLM-as-Judge Configuration (Single-Response)

Since HELM uses single-response evaluation (not pairwise), the judge below
assesses absolute quality rather than relative win rate.

**Judge model:** GPT-4o or GPT-4.1 (recommended)
**Evaluation type:** Single-response quality assessment
**Dimensions:** creativity, coherence, relevance, style_execution
**Scale:** 1–10 per dimension

## Judge Prompt Template

You are an expert creative writing evaluator.

**Prompt:** {PROMPT}
**Response:** {RESPONSE}

Rate this creative writing response on four dimensions (1 = very poor, 10 = excellent):

1. **Creativity (1–10):** Is the response original and imaginative?
   Does it avoid clichés and bring a fresh perspective?
   10 = genuinely surprising, inventive approach.
   1 = generic, predictable, or formulaic.

2. **Coherence (1–10):** Is the response well-structured and internally consistent?
   Does it hold together as a complete piece of writing?
   10 = polished, flows naturally, clear arc.
   1 = disjointed, contradictory, or incomplete.

3. **Relevance (1–10):** Does the response directly address what the prompt requested?
   10 = perfectly matches the prompt's genre, tone, and requirements.
   1 = off-topic or ignores key constraints in the prompt.

4. **Style Execution (1–10):** How well does the response execute the intended
   creative form (poem meter, dialogue rhythm, story pacing, etc.)?
   10 = masterful command of the form.
   1 = fails to demonstrate the expected style conventions.

Provide ratings as:
Creativity: [1–10]
Coherence: [1–10]
Relevance: [1–10]
Style Execution: [1–10]
Overall: [mean, rounded to 1 decimal]

## Pairwise Evaluation (for reproducing original benchmark)

To reproduce the original Arena-Hard pairwise scoring:

```python
PAIRWISE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate
the quality of the responses provided by two AI assistants to the user
question displayed below. You should choose the assistant that follows
the user's instructions and answers the user's question better.
Your evaluation should consider factors such as the helpfulness,
relevance, accuracy, depth, creativity, and level of detail of their
responses. Avoid any position biases and ensure that the order in which
the responses were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation.
Do not favor certain names of the assistants. Be as objective as possible.
Output your final verdict by strictly following this format:
\"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better,
and \"[[C]]\" for a tie."""
```

For pairwise win-rate, run all models on the same 250 prompts and compare
responses pairwise using the judge above (or use ELO rating from win counts).

## Prompt Diversity Notes

The 250 creative writing prompts span:

| Type | Examples |
|------|---------|
| Poetry | Multi-language (Spanish, Chinese, Russian, French, Vietnamese, Greek) |
| Rap / Hip-hop | Themed, character-based, narrative raps |
| Song lyrics | Themed compositions ("cloud + fleeting love", etc.) |
| Short stories | Character-driven, genre fiction |
| Scripts / Dialogues | Barber-customer conversations, character interactions |
| Jokes / Humor | Stand-up style, situational comedy |
| Other | Philosophical responses, explanations in creative style |

## Baseline Context (from Arena leaderboard)

Top performers on creative writing subset (approximate):
- Claude 3.5 Sonnet: among top-3 on creative tasks
- GPT-4o: strong on coherence and style execution
- Gemini Ultra: competitive on multilingual prompts

The creative writing subset shows **higher model differentiation** than the
coding/math splits, making it effective for ranking creative LLMs.
