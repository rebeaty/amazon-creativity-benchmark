"""HELM Run Specs for creative_process."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_CREATIVITY = """\
Evaluate the CREATIVITY of the generated response to a creative process task.
Consider originality, divergent thinking, unexpected connections, and novelty of ideas.

Score 1: Completely unoriginal, predictable response
Score 2: Mostly conventional ideas with little creative merit
Score 3: Some creative elements mixed with conventional thinking
Score 4: Genuinely creative response with strong original ideas
Score 5: Highly creative, original, and surprising response demonstrating exceptional creative thinking
"""

_RUBRIC_LLM_JUDGE_ORIGINALITY = """\
Evaluate the ORIGINALITY of the generated response to a creative process task.
Consider how novel, unique, and unprecedented the ideas are compared to typical responses.

Score 1: Ideas are completely derivative and unoriginal
Score 2: Slightly varied but largely unoriginal ideas
Score 3: Some original elements but draws heavily on common ideas
Score 4: Notably original ideas that stand out from typical responses
Score 5: Exceptionally original ideas that are novel, unique, and previously unthought-of
"""


@run_spec_function("creative_process")
def get_creative_process_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.creative_process_scenario.CreativeProcessScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_creativity"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_originality"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_creativity",
                "rubric": _RUBRIC_LLM_JUDGE_CREATIVITY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_originality",
                "rubric": _RUBRIC_LLM_JUDGE_ORIGINALITY,
            },
        ),
    ]

    return RunSpec(
        name="creative_process",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "creative_process"],
        annotators=annotators,
    )
