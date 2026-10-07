from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from automatic_lecture_tex.config import LLMConfig
from automatic_lecture_tex.llm import (
    StructuredBackendAmbiguousRejectionError,
    StructuredInputTooLargeError,
    StructuredOutputTruncatedError,
)
from automatic_lecture_tex.llm_robust import LectureModelClient


class _Payload(BaseModel):
    ok: bool


class _FakeBadRequestError(Exception):
    pass


class _Completions:
    def __init__(self, message: str) -> None:
        self.message = message
        self.max_tokens = []

    def create(self, **kwargs):
        self.max_tokens.append(kwargs["max_tokens"])
        if len(self.max_tokens) == 1:
            raise _FakeBadRequestError(self.message)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok":true}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=27233,
                completion_tokens=8,
                total_tokens=27241,
            ),
        )


def _client(monkeypatch, message: str) -> tuple[LectureModelClient, _Completions]:
    import automatic_lecture_tex.llm_robust as robust

    monkeypatch.setattr(robust, "BadRequestError", _FakeBadRequestError)
    client = LectureModelClient(
        LLMConfig(
            model="fake",
            max_tokens=32768,
            max_retries=0,
        )
    )
    completions = _Completions(message)
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def test_split_aware_structured_call_shrinks_output_budget_before_splitting(
    monkeypatch,
) -> None:
    client, completions = _client(
        monkeypatch,
        "This model's maximum context length is 60000 tokens. "
        "However, you requested 32768 output tokens and your prompt contains at least "
        "27233 input tokens, for a total of at least 60001 tokens.",
    )

    result = client._structured(
        "prompt",
        _Payload,
        operation="graph_revision_proposal",
        max_tokens=32768,
        split_oversized_task=True,
    )

    assert result.ok is True
    assert completions.max_tokens == [32768, 16384]


def test_split_aware_structured_call_still_delegates_true_input_overflow(
    monkeypatch,
) -> None:
    client, _ = _client(
        monkeypatch,
        "input length (61000) exceeds the model maximum context length (60000)",
    )

    with pytest.raises(StructuredInputTooLargeError, match="input cannot fit"):
        client._structured(
            "prompt",
            _Payload,
            operation="graph_revision_proposal",
            max_tokens=4096,
            split_oversized_task=True,
        )



class _TruncatedCompletions:
    def __init__(self) -> None:
        self.max_tokens = []

    def create(self, **kwargs):
        self.max_tokens.append(kwargs["max_tokens"])
        budget = kwargs["max_tokens"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=""),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=budget,
                total_tokens=100 + budget,
            ),
        )


def test_split_aware_structured_call_reports_output_truncation_separately() -> None:
    client = LectureModelClient(
        LLMConfig(
            model="fake",
            max_tokens=32768,
            max_retries=0,
        )
    )
    completions = _TruncatedCompletions()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with pytest.raises(StructuredOutputTruncatedError) as caught:
        client._structured(
            "prompt",
            _Payload,
            operation="graph_revision_proposal",
            max_tokens=16384,
            split_oversized_task=True,
        )

    assert caught.value.max_tokens == 16384
    assert caught.value.raw_chars == 0
    assert completions.max_tokens == [16384]



def test_split_aware_structured_call_delegates_ambiguous_context_or_params_400(
    monkeypatch,
) -> None:
    client, completions = _client(
        monkeypatch,
        "Provider returned error: invalid_request_error: The request was rejected. "
        "Possible causes: input exceeds the model's maximum context length, "
        "or the request contains invalid parameters.",
    )

    with pytest.raises(StructuredBackendAmbiguousRejectionError) as caught:
        client._structured(
            "prompt",
            _Payload,
            operation="graph_revision_proposal",
            max_tokens=32768,
            split_oversized_task=True,
        )

    assert "possible context overflow or invalid parameters" in str(caught.value)
    assert isinstance(caught.value.backend_error, _FakeBadRequestError)
    assert completions.max_tokens == [32768]


def test_non_split_call_does_not_reclassify_ambiguous_provider_400(monkeypatch) -> None:
    client, completions = _client(
        monkeypatch,
        "The request was rejected. Possible causes: input exceeds the model's maximum "
        "context length, or the request contains invalid parameters.",
    )

    with pytest.raises(_FakeBadRequestError):
        client._structured(
            "prompt",
            _Payload,
            operation="ordinary_structured",
            max_tokens=32768,
            split_oversized_task=False,
        )

    assert completions.max_tokens == [32768]


def test_plain_invalid_parameters_400_is_not_treated_as_context_overflow(monkeypatch) -> None:
    client, completions = _client(
        monkeypatch,
        "invalid_request_error: request contains invalid parameters",
    )

    with pytest.raises(_FakeBadRequestError):
        client._structured(
            "prompt",
            _Payload,
            operation="graph_revision_proposal",
            max_tokens=32768,
            split_oversized_task=True,
        )

    assert completions.max_tokens == [32768]
