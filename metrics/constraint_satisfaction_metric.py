import ast
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# TODO: Implement full constraint extraction from NeoCoder annotator.
# Reference: https://github.com/JHU-CLSP/NeoCoder/blob/main/src/utils/configs.py#L240
# When the annotator is configured, replace the proxy below with:
#   annotations = request_state.annotations
#   score = float(annotations.get("constraint_satisfaction", 0.0))


class ConstraintSatisfactionMetric(Metric):
    """Fraction of constraints satisfied by the generated output.

    Falls back to code-validity as a proxy when annotator is not configured.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        # Use annotator output when available
        annotations = request_state.annotations or {}
        if "constraint_satisfaction" in annotations:
            score = float(annotations["constraint_satisfaction"])
        else:
            # Proxy: syntactic validity of generated code
            try:
                ast.parse(completion)
                score = 1.0
            except SyntaxError:
                score = 0.0

        return [Stat(MetricName("constraint_satisfaction")).add(score)]
