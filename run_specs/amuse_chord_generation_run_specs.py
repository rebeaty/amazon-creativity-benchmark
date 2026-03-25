"""HELM Run Specs for amuse_chord_generation."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("amuse_chord_generation")
def get_amuse_chord_generation_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.amuse_chord_generation_scenario.AmuseChordGenerationScenario",
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
        num_outputs=30,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.disinformation_metrics.DisinformationMetric", args={}),
        MetricSpec(class_name="metrics.jsd_metric.JSDMetric", args={}),
    ]

    return RunSpec(
        name="amuse_chord_generation",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "amuse_chord_generation"],
        annotators=None,
    )
