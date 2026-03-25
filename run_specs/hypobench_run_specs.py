"""HELM Run Specs for hypobench."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_NOVELTY = """\
Evaluate the NOVELTY of the generated scientific hypothesis.
Consider how original, unprecedented, and non-obvious the hypothesis is relative to existing knowledge.

Score 1: Hypothesis is a restatement of known facts with no novelty
Score 2: Minimal novelty; hypothesis is a minor variation of known ideas
Score 3: Somewhat novel hypothesis that goes slightly beyond known knowledge
Score 4: Notably novel hypothesis presenting a genuinely new perspective
Score 5: Highly novel hypothesis presenting an original, unprecedented scientific claim
"""

_RUBRIC_LLM_JUDGE_SIGNIFICANCE = """\
Evaluate the SIGNIFICANCE of the generated scientific hypothesis.
Consider its potential scientific impact, importance, and relevance to advancing knowledge.

Score 1: Hypothesis is trivial with no scientific significance
Score 2: Low significance with minimal potential scientific impact
Score 3: Moderate significance with some potential scientific value
Score 4: High significance with clear potential for important scientific contributions
Score 5: Exceptional significance with the potential for major scientific breakthroughs
"""

_RUBRIC_LLM_JUDGE_VERIFIABILITY = """\
Evaluate the VERIFIABILITY of the generated scientific hypothesis.
Consider whether the hypothesis can be empirically tested with current or near-future experimental methods.

Score 1: Hypothesis is entirely unverifiable or untestable
Score 2: Hypothesis is very difficult to verify with major experimental challenges
Score 3: Hypothesis is testable but with significant methodological hurdles
Score 4: Hypothesis is clearly testable with feasible experimental methods
Score 5: Hypothesis is easily verifiable with straightforward experimental approaches
"""


@run_spec_function("hypobench")
def get_hypobench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.hypobench_scenario.HypoBenchScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_novelty"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_significance"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_verifiability"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_novelty",
                "rubric": _RUBRIC_LLM_JUDGE_NOVELTY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_significance",
                "rubric": _RUBRIC_LLM_JUDGE_SIGNIFICANCE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_verifiability",
                "rubric": _RUBRIC_LLM_JUDGE_VERIFIABILITY,
            },
        ),
    ]

    return RunSpec(
        name="hypobench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "hypobench"],
        annotators=annotators,
    )
