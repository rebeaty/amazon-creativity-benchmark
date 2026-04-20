"""HELM Run Specs for slang_generation."""

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
Evaluate the CREATIVITY of the generated slang term or expression.
Consider originality, inventiveness, and how novel the slang construction is.

Score 1: Slang term is entirely conventional or already widely used
Score 2: Minimal creativity; slight variation on existing slang
Score 3: Somewhat creative slang with some novel elements
Score 4: Genuinely creative, inventive slang term
Score 5: Highly creative, original slang demonstrating exceptional linguistic inventiveness
"""

_RUBRIC_LLM_JUDGE_RELEVANCE = """\
Evaluate the RELEVANCE of the generated slang term to the target context or meaning.
Consider how well the slang captures and conveys its intended meaning.

Score 1: Slang has no relevance to the intended meaning or context
Score 2: Minimally relevant with poor semantic connection
Score 3: Somewhat relevant with moderate semantic alignment
Score 4: Mostly relevant with good semantic connection to the intended meaning
Score 5: Perfectly relevant; slang precisely and cleverly captures its intended meaning
"""


@run_spec_function("slang_generation")
def get_slang_generation_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.slang_generation_scenario.SlangGenerationScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_relevance"}),
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
                "metric_name": "llm_judge_relevance",
                "rubric": _RUBRIC_LLM_JUDGE_RELEVANCE,
            },
        ),
    ]

    return RunSpec(
        name="slang_generation",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "slang_generation"],
        annotators=annotators,
    )
