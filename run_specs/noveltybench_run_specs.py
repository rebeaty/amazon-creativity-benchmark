"""HELM Run Specs for noveltybench."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("noveltybench")
def get_noveltybench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.noveltybench_scenario.NoveltyBenchScenario",
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
        num_outputs=10,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        # TODO: no metrics in registry, using fallback
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicReferenceMetric", args={}),
    ]

    return RunSpec(
        name="noveltybench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "noveltybench"],
        annotators=None,
    )
