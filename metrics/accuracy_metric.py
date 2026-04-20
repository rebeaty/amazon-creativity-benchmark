from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG


class AccuracyMetric(Metric):
    """Exact-match accuracy: 1.0 if completion matches the CORRECT_TAG reference (case-insensitive strip)."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        correct_refs = [
            ref for ref in request_state.instance.references if CORRECT_TAG in ref.tags
        ]
        if not correct_refs:
            return [Stat(MetricName("accuracy")).add(0.0)]

        correct_text = correct_refs[0].output.text.strip()
        score = 1.0 if completion.lower() == correct_text.lower() else 0.0
        return [Stat(MetricName("accuracy")).add(score)]
