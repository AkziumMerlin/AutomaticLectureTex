from types import SimpleNamespace

from pydantic import BaseModel

from automatic_lecture_tex.llm import LectureModelClient


class Payload(BaseModel):
    value: str


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.max_tokens = []
        self.messages = []

    def create(self, **kwargs):
        self.max_tokens.append(kwargs["max_tokens"])
        self.messages.append(kwargs["messages"])
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


def test_structured_doubles_output_budget_after_length_truncation() -> None:
    client, completions = _client(
        [
            _response('{"value":"cut', finish_reason="length", completion_tokens=2048),
            _response('{"value":"complete"}', finish_reason="stop", completion_tokens=20),
        ]
    )

    result = client._structured("prompt", Payload, max_tokens=2048, operation="test")

    assert result.value == "complete"
    assert completions.max_tokens == [2048, 4096]


def test_structured_can_grow_across_all_retries() -> None:
    client, completions = _client(
        [
            _response('{"value":"cut', finish_reason="length", completion_tokens=2048),
            _response('{"value":"still cut', finish_reason="length", completion_tokens=4096),
            _response('{"value":"complete"}', finish_reason="stop", completion_tokens=20),
        ]
    )

    result = client._structured("prompt", Payload, max_tokens=2048, operation="test")

    assert result.value == "complete"
    assert completions.max_tokens == [2048, 4096, 8192]


def test_structured_keeps_budget_for_non_truncation_parse_error() -> None:
    client, completions = _client(
        [
            _response('{not json}', finish_reason="stop", completion_tokens=10),
            _response('{"value":"complete"}', finish_reason="stop", completion_tokens=20),
        ]
    )

    result = client._structured("prompt", Payload, max_tokens=2048, operation="test")

    assert result.value == "complete"
    assert completions.max_tokens == [2048, 2048]


def test_non_guided_retry_keeps_schema_instruction() -> None:
    client, completions = _client(
        [
            _response('{not json}', finish_reason="stop", completion_tokens=10),
            _response('{"value":"complete"}', finish_reason="stop", completion_tokens=20),
        ]
    )

    result = client._structured(
        "prompt",
        Payload,
        max_tokens=2048,
        guided_json=False,
        operation="test",
    )

    assert result.value == "complete"
    retry_text = completions.messages[1][1]["content"][0]["text"]
    assert "JSON schema:" in retry_text
