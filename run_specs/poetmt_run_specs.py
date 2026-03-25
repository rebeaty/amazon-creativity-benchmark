"""HELM Run Specs for poetmt."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_BEAUTY_OF_SOUND = """\
Evaluate the BEAUTY OF SOUND in the translated Chinese classical poem.
Consider phonetic qualities, rhythm, musicality, and tonal beauty of the translation.

Score 1: Translation has no phonetic beauty; sounds harsh or discordant
Score 2: Minimal phonetic qualities with poor rhythm or musicality
Score 3: Some phonetic appeal with moderate rhythmic qualities
Score 4: Good phonetic beauty with pleasing rhythm and sound patterns
Score 5: Excellent phonetic beauty with outstanding rhythm, musicality, and tonal quality
"""

_RUBRIC_LLM_JUDGE_BEAUTY_OF_FORM = """\
Evaluate the BEAUTY OF FORM in the translated Chinese classical poem.
Consider structural fidelity, formal constraints adherence, visual/structural elegance of the translation.

Score 1: Translation ignores all formal constraints of the original
Score 2: Minimal formal fidelity with poor structural quality
Score 3: Some formal elements maintained with adequate structure
Score 4: Good formal fidelity with clear structural elegance
Score 5: Excellent formal beauty with perfect adherence to poetic structure
"""

_RUBRIC_LLM_JUDGE_BEAUTY_OF_MEANING = """\
Evaluate the BEAUTY OF MEANING in the translated Chinese classical poem.
Consider semantic depth, preservation of imagery, philosophical resonance, and meaning fidelity.

Score 1: Translation completely loses the meaning and imagery of the original
Score 2: Minimal meaning preserved with major losses of imagery or depth
Score 3: Core meaning preserved with some loss of nuance or imagery
Score 4: Good meaning fidelity with most imagery and depth preserved
Score 5: Excellent semantic beauty that fully captures the depth and imagery of the original
"""


@run_spec_function("poetmt")
def get_poetmt_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.poetmt_scenario.PoetMTScenario",
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
        MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_beauty_of_sound"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_beauty_of_form"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_beauty_of_meaning"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_beauty_of_sound",
                "rubric": _RUBRIC_LLM_JUDGE_BEAUTY_OF_SOUND,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_beauty_of_form",
                "rubric": _RUBRIC_LLM_JUDGE_BEAUTY_OF_FORM,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_beauty_of_meaning",
                "rubric": _RUBRIC_LLM_JUDGE_BEAUTY_OF_MEANING,
            },
        ),
    ]

    return RunSpec(
        name="poetmt",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "poetmt"],
        annotators=annotators,
    )
