import json
from types import SimpleNamespace

from pydantic import BaseModel

from automatic_lecture_tex.llm_robust import LectureModelClient


class Payload(BaseModel):
    value: str


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.max_tokens = []
        self.response_formats = []

    def create(self, **kwargs):
        self.max_tokens.append(kwargs["max_tokens"])
        self.response_formats.append(kwargs.get("response_format"))
        return self.responses.pop(0)


def _response(content: str, *, finish_reason: str | None, completion_tokens: int = 0):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )


def _client(responses, *, retries: int = 2):
    client = LectureModelClient.__new__(LectureModelClient)
    client.config = SimpleNamespace(
        model="model",
        temperature=0.0,
        max_tokens=4096,
        max_retries=retries,
        thinking=False,
    )
    completions = _Completions(responses)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._record_usage = lambda *args, **kwargs: None
    return client, completions


def test_structured_infers_truncation_from_unterminated_json_with_stop_reason() -> None:
    client, completions = _client(
        [
            _response('{"value":"cut', finish_reason="stop", completion_tokens=8),
            _response('{"value":"complete"}', finish_reason="stop", completion_tokens=20),
        ]
    )

    result = client._structured("prompt", Payload, max_tokens=2048, operation="test")

    assert result.value == "complete"
    assert completions.max_tokens == [2048, 4096]
    assert completions.response_formats[0] is not None
    assert completions.response_formats[1] is None


def test_structured_infers_truncation_from_unclosed_object_with_stop_reason() -> None:
    client, completions = _client(
        [
            _response('{"value":"cut"', finish_reason="stop", completion_tokens=8),
            _response('{"value":"complete"}', finish_reason="stop", completion_tokens=20),
        ]
    )

    result = client._structured("prompt", Payload, max_tokens=2048, operation="test")

    assert result.value == "complete"
    assert completions.max_tokens == [2048, 4096]
    assert completions.response_formats == [completions.response_formats[0], None]


def test_balanced_malformed_json_does_not_infer_truncation() -> None:
    client, completions = _client(
        [
            _response('{"value" "broken"}', finish_reason="stop", completion_tokens=8),
            _response('{"value":"complete"}', finish_reason="stop", completion_tokens=20),
        ]
    )

    result = client._structured("prompt", Payload, max_tokens=2048, operation="test")

    assert result.value == "complete"
    assert completions.max_tokens == [2048, 2048]
    assert all(item is not None for item in completions.response_formats)


def test_structurally_truncated_json_is_not_locally_completed() -> None:
    client, _ = _client([], retries=0)

    try:
        client._parse_json('{"value":"cut', Payload)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("partial JSON must not be repaired into a valid object")


def test_cli_pipeline_uses_robust_llm_client() -> None:
    from automatic_lecture_tex.pipeline_robust import Pipeline

    assert Pipeline.llm.fget.__globals__["LectureModelClient"] is LectureModelClient
