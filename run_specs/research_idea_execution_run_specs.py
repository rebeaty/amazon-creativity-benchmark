"""HELM Run Specs for research_idea_execution."""

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
Evaluate the NOVELTY of the generated research idea execution plan.
Consider how original, innovative, and unprecedented the research approach is.

Score 1: Research plan is entirely conventional with no novel elements
Score 2: Minimal novelty; mostly follows standard research approaches
Score 3: Some novel elements within a mostly conventional framework
Score 4: Notably novel approach with genuinely original research elements
Score 5: Highly novel research plan with exceptional originality and innovation
"""

_RUBRIC_LLM_JUDGE_FEASIBILITY = """\
Evaluate the FEASIBILITY of the generated research idea execution plan.
Consider whether the proposed methods, timeline, and resources are realistic and achievable.

Score 1: Plan is completely infeasible with impossible methods or timelines
Score 2: Mostly infeasible with major practical obstacles
Score 3: Partially feasible with significant practical challenges
Score 4: Mostly feasible with reasonable methods and achievable goals
Score 5: Highly feasible with realistic, well-planned, and achievable execution
"""

_RUBRIC_LLM_JUDGE_QUALITY = """\
Evaluate the overall QUALITY of the generated research idea execution plan.
Consider clarity, comprehensiveness, scientific rigor, and potential impact of the plan.

Score 1: Plan is of very poor quality with major gaps in all dimensions
Score 2: Low quality plan with significant weaknesses in rigor or clarity
Score 3: Adequate quality plan meeting basic research planning criteria
Score 4: Good quality plan with strong scientific rigor and clarity
Score 5: Excellent quality plan demonstrating exceptional scientific rigor, clarity, and potential impact
"""


@run_spec_function("research_idea_execution")
def get_research_idea_execution_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.research_idea_execution_scenario.ResearchIdeaExecutionScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_feasibility"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_quality"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4o",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "llm_judge_novelty",
                "rubric": _RUBRIC_LLM_JUDGE_NOVELTY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4o",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "llm_judge_feasibility",
                "rubric": _RUBRIC_LLM_JUDGE_FEASIBILITY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4o",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "llm_judge_quality",
                "rubric": _RUBRIC_LLM_JUDGE_QUALITY,
            },
        ),
    ]

    return RunSpec(
        name="research_idea_execution",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "research_idea_execution"],
        annotators=annotators,
    )
