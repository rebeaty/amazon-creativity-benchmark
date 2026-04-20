"""Generic per-metric LLM-as-Judge annotator for the Amazon Creativity Benchmark."""
import re
from typing import Any, Dict

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


BACKUP_JUDGE_MODEL = "google/gemini-2.5-flash-lite"


class GenericLLMJudgeAnnotator(Annotator):
    """Calls an LLM judge once per metric with a rubric-specific prompt.

    Each instance handles exactly ONE metric dimension. ``self.name`` is set
    per-instance from ``metric_name`` so multiple annotator specs on the same
    run do not overwrite each other's annotations in ``request_state.annotations``.

    The ``auto_client`` parameter is auto-injected by HELM's AnnotatorFactory.
    """

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
        self.name = f"generic_llm_judge_{metric_name}"

    def _call_judge(self, model_name: str, prompt: str) -> int:
        request = Request(
            model=model_name,
            model_deployment=model_name,
            prompt=prompt,
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_new_tokens,
            num_completions=1,
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            raise RuntimeError(f"Judge call failed for model {model_name}")
        score_text = result.completions[0].text.strip()
        match = re.search(r'\d+', score_text)
        return int(match.group()) if match else 0

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()
        input_text = request_state.instance.input.text

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

        try:
            score = self._call_judge(self.judge_model_name, prompt)
        except Exception:
            try:
                score = self._call_judge(BACKUP_JUDGE_MODEL, prompt)
            except Exception:
                score = -100

        return {self.metric_name: score}
