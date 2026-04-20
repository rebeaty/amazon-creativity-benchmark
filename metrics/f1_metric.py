"""F1 metric: token-level F1 between prediction and gold reference, stat named 'f1'."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_reference_metrics import f1_score
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class F1Metric(Metric):
    """Token-level F1 score between the first prediction and gold references.

    Produces a stat named 'f1' (not 'f1_score') to match registry expectations.
    Score is the max F1 across all gold references.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        references = request_state.instance.references
        if not references:
            return [Stat(MetricName("f1")).add(0.0)]

        pred = request_state.result.completions[0].text.strip()
        score = max(f1_score(ref.output.text, pred) for ref in references)
        return [Stat(MetricName("f1")).add(score)]
