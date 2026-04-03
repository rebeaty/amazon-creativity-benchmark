"""HELM Run Specs for crowdcounter."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_RUBRIC_COUNTERSPEECH = """\
Evaluate the quality of the generated counterspeech response to the given hate speech.
Consider whether the response effectively addresses the hateful content, promotes understanding, and discourages harmful behavior.

Score 1: Response is irrelevant, inflammatory, or itself contains hateful content
Score 2: Response acknowledges the hate speech but is ineffective or generic
Score 3: Response is reasonable but lacks depth or persuasiveness
Score 4: Response is thoughtful, addresses the core issue, and promotes understanding
Score 5: Response is excellent — empathetic, persuasive, specific, and effectively counters the hate
"""


@run_spec_function("crowdcounter")
def get_crowdcounter_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.crowdcounter_scenario.CrowdCounterScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=5,  # ASSUMPTION: few-shot, TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "counterspeech_quality"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "google/gemini-2.0-flash-lite",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "counterspeech_quality",
                "rubric": _RUBRIC_COUNTERSPEECH,
            },
        ),
    ]

    return RunSpec(
        name="crowdcounter",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "crowdcounter"],
        annotators=annotators,
    )
