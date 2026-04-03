"""Generic per-metric LLM-as-Judge annotator for the Amazon Creativity Benchmark."""
import re
from typing import Any, Dict

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


class GenericLLMJudgeAnnotator(Annotator):
    """Calls an LLM judge once per metric with a rubric-specific prompt.

    Each instance of this annotator handles exactly ONE metric dimension.
    The annotation key written is the metric_name, so each AnnotatorSpec
    must use a distinct metric_name.

    The ``auto_client`` parameter is auto-injected by HELM's AnnotatorFactory.
    """

    name = "generic_llm_judge"

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        metric_name: str,
        rubric: str,
    ):
        self._auto_client = auto_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.metric_name = metric_name
        self.rubric = rubric

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()
        input_text = request_state.instance.input.text

        # Include reference (baseline) for pairwise comparison if available
        reference_text = ""
        if request_state.instance.references:
            reference_text = request_state.instance.references[0].output.text

        prompt = (
            f"{self.rubric}\n\n"
            f"Instruction:\n{input_text}\n\n"
        )
        if reference_text:
            prompt += f"Reference response:\n{reference_text}\n\n"
        prompt += (
            f"Generated response:\n{completion}\n\n"
            f"Provide only the integer score (e.g., 3):"
        )

        request = Request(
            model=self.judge_model_name,
            model_deployment=self.judge_model_name,
            prompt=prompt,
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_new_tokens,
            num_completions=1,
        )

        try:
            result = self._auto_client.make_request(request)
            if not result.success:
                return {self.metric_name: 0}
            score_text = result.completions[0].text.strip()
            match = re.search(r'\d+', score_text)
            score = int(match.group()) if match else 0
        except Exception:
            score = 0

        return {self.metric_name: score}
