from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class CorrelationMetric(Metric):
    """Per-instance prediction/reference pair emitter for Pearson or Spearman correlation.

    evaluate_generation emits two stats per instance (pred and true values).
    Aggregate correlation must be computed downstream from collected stat values.
    """

    def __init__(self, correlation_type: str):
        super().__init__()
        assert correlation_type in ("pearson", "spearman"), (
            "correlation_type must be 'pearson' or 'spearman'"
        )
        self.correlation_type = correlation_type

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        references = request_state.instance.references
        metric_name = f"{self.correlation_type}_correlation"

        try:
            pred_value = float(completion)
        except ValueError:
            pred_value = 0.0

        try:
            true_value = float(references[0].output.text.strip()) if references else 0.0
        except ValueError:
            true_value = 0.0

        # Emit signed error as per-instance proxy; aggregate correlation requires
        # collecting all pred/true pairs downstream.
        signed_error = pred_value - true_value
        pred_stat = Stat(MetricName(f"{metric_name}_pred")).add(pred_value)
        true_stat = Stat(MetricName(f"{metric_name}_true")).add(true_value)
        error_stat = Stat(MetricName(metric_name)).add(signed_error)
        return [pred_stat, true_stat, error_stat]
