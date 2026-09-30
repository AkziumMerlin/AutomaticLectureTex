from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from automatic_lecture_tex.config import LLMConfig
from automatic_lecture_tex.llm import StructuredTaskTooLargeError
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

    with pytest.raises(StructuredTaskTooLargeError, match="input cannot fit"):
        client._structured(
            "prompt",
            _Payload,
            operation="graph_revision_proposal",
            max_tokens=4096,
            split_oversized_task=True,
        )
