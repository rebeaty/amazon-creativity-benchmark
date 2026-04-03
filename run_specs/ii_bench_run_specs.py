"""HELM Run Specs for ii_bench."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_MULTIPLE_CHOICE_JOINT,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("ii_bench")
def get_ii_bench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.ii_bench_scenario.IIBenchScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="Answer: ",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric", args={}),
    ]

    return RunSpec(
        name="ii_bench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "ii_bench"],
        annotators=None,
    )
