"""HELM Run Specs for robotoolbench."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("robotoolbench")
def get_robotoolbench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.robotoolbench_scenario.RoboToolBenchScenario",
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
        MetricSpec(class_name="metrics.xml_validity_metric.XmlValidityMetric", args={}),
        MetricSpec(class_name="metrics.array_dimensions_metric.ArrayDimensionsMetric", args={}),
    ]

    return RunSpec(
        name="robotoolbench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "robotoolbench"],
        annotators=None,
    )
