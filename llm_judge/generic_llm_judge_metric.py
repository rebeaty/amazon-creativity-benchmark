"""Generic per-metric LLM-as-Judge metric for the Amazon Creativity Benchmark."""
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class GenericLLMJudgeMetric(Metric):
    """Reads a single numeric annotation produced by GenericLLMJudgeAnnotator."""

    def __init__(self, metric_name: str):
        super().__init__()
        self.metric_name = metric_name

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        annotations = request_state.annotations or {}
        score = float(annotations.get(self.metric_name, 0))
        return [Stat(MetricName(self.metric_name)).add(score)]
