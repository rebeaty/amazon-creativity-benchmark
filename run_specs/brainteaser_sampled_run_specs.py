"""HELM Run Specs (SAMPLED MIRROR) for BrainTeaser.

Two evaluation units matching the two HF configs, each with the
reproducible 200-item sampler applied at the scenario level:
  - brainteaser_sampled_sentence_puzzle
  - brainteaser_sampled_word_puzzle

Metrics / adapter identical to the original [brainteaser_run_specs.py]
so scores are directly comparable once you pool (or choose to keep
subtask-level resolution).
"""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_MULTIPLE_CHOICE_JOINT,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _build_spec(subtask: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.brainteaser_sampled_scenario.BrainteaserSampledScenario",
        args={"subtask": subtask},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="Answer: ",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["exact_match"]},
        ),
    ]

    return RunSpec(
        name=f"brainteaser_sampled:subtask={subtask}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "brainteaser_sampled"],
        annotators=None,
    )


@run_spec_function("brainteaser_sampled_sentence_puzzle")
def get_brainteaser_sampled_sentence_puzzle_spec() -> RunSpec:
    return _build_spec("sentence_puzzle")


@run_spec_function("brainteaser_sampled_word_puzzle")
def get_brainteaser_sampled_word_puzzle_spec() -> RunSpec:
    return _build_spec("word_puzzle")
