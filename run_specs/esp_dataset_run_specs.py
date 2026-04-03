"""HELM Run Specs for esp_dataset."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("esp_dataset")
def get_esp_dataset_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.esp_dataset_scenario.ESPDatasetScenario",
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
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]}),
        MetricSpec(class_name="helm.benchmark.metrics.image_generation.clip_score_metrics.CLIPScoreMetric", args={"model_name": "openai/clip-vit-base-patch32"}),
    ]

    return RunSpec(
        name="esp_dataset",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "esp_dataset"],
        annotators=None,
    )
