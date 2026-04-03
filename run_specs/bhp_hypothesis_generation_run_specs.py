"""HELM Run Specs for bhp_hypothesis_generation."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_RUBRIC_HYPOTHESIS_QUALITY = """\
Evaluate the quality of the generated biomedical hypothesis given the research background.
Consider scientific grounding, novelty, specificity, and logical coherence.

Score 1: Hypothesis is nonsensical, irrelevant, or scientifically invalid
Score 2: Hypothesis is vaguely related but lacks specificity or scientific rigor
Score 3: Hypothesis is reasonable but not particularly novel or insightful
Score 4: Hypothesis is well-grounded, specific, and shows good scientific reasoning
Score 5: Hypothesis is excellent — novel, specific, scientifically rigorous, and testable
"""


@run_spec_function("bhp_hypothesis_generation")
def get_bhp_hypothesis_generation_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.bhp_hypothesis_generation_scenario.BHPHypothesisGenerationScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "hypothesis_quality"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "google/gemini-2.0-flash-lite",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "hypothesis_quality",
                "rubric": _RUBRIC_HYPOTHESIS_QUALITY,
            },
        ),
    ]

    return RunSpec(
        name="bhp_hypothesis_generation",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "bhp_hypothesis_generation"],
        annotators=annotators,
    )
