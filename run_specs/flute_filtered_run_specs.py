"""HELM Run Specs for flute_filtered."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("flute_filtered")
def get_flute_filtered_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.flute_filtered_scenario.FLUTEFilteredScenario",
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
        max_tokens=16,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={}),
    ]

    return RunSpec(
        name="flute_filtered",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "flute_filtered"],
        annotators=None,
    )
