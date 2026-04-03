"""HELM Run Specs for data_narrative."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_RUBRIC_NARRATIVE_QUALITY = """\
Evaluate the quality of the generated data narrative based on the given data table and topic.
Consider accuracy of data interpretation, insightfulness, clarity, and narrative coherence.

Score 1: Narrative is factually wrong, irrelevant, or nonsensical
Score 2: Narrative mentions the topic but misinterprets data or lacks insight
Score 3: Narrative is reasonable but generic, missing key trends or patterns
Score 4: Narrative accurately describes key trends with good insight and clarity
Score 5: Narrative is excellent — insightful, accurate, well-structured, and highlights significant patterns
"""


@run_spec_function("data_narrative")
def get_data_narrative_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.data_narrative_scenario.DataNarrativeScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "narrative_quality"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "google/gemini-2.0-flash-lite",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "narrative_quality",
                "rubric": _RUBRIC_NARRATIVE_QUALITY,
            },
        ),
    ]

    return RunSpec(
        name="data_narrative",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "data_narrative"],
        annotators=annotators,
    )
