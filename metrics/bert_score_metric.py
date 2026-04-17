import contextlib
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# Since transformers PR #36963, init_empty_weights is native to transformers and models are
# ALWAYS loaded on meta device. Patching is_accelerate_available no longer prevents this.
# We must replace init_empty_weights with a no-op context manager so bert_score can call
# model.to(device) without hitting the "Cannot copy out of meta tensor" error.
@contextlib.contextmanager
def _noop_init_empty_weights(include_buffers=False):
    yield

import transformers.modeling_utils as _mu
import transformers.integrations.accelerate as _ta
_mu.init_empty_weights = _noop_init_empty_weights
_ta.init_empty_weights = _noop_init_empty_weights


class BertScoreMetric(Metric):
    """Computes BERTScore F1 between prediction and reference using bert-base-uncased."""

    def __init__(self, model_type: str = "bert-base-uncased"):
        super().__init__()
        self._model_type = model_type

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        prediction = request_state.result.completions[0].text.strip()
        references = [ref.output.text.strip() for ref in request_state.instance.references if ref.output.text.strip()]

        if not references or not prediction:
            return [Stat(MetricName("bert_score")).add(0.0)]

        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        import torch
        from bert_score import score as bert_score_fn

        _, _, F1 = bert_score_fn(
            [prediction],
            [references[0]],
            model_type=self._model_type,
            lang="en",
            verbose=False,
            device="cpu",
        )
        return [Stat(MetricName("bert_score")).add(float(F1[0].item()))]
