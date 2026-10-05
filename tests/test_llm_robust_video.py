from __future__ import annotations

import threading
from types import SimpleNamespace

from pydantic import BaseModel

from automatic_lecture_tex.config import LLMConfig
from automatic_lecture_tex.llm_robust import LectureModelClient


class Payload(BaseModel):
    value: str


def test_robust_client_sends_video_url_with_model_override(tmp_path) -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    client.config = LLMConfig(
        model="deepseek/deepseek-v4.1-flash",
        compatibility_mode="generic",
        structured_output_mode="native",
        max_tokens=128,
        max_retries=0,
    )
    client._usage_lock = threading.Lock()
    client.reset_usage()
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"value":"ok"}'),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=3,
                    total_tokens=15,
                    cost=0.001,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
            )

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    result = client._structured(
        "Inspect native video.",
        Payload,
        videos=[video],
        model="qwen/qwen3.8-omni-flash",
        operation="knowledge_extract_native_video",
    )

    assert result.value == "ok"
    assert captured["model"] == "qwen/qwen3.8-omni-flash"
    content = captured["messages"][1]["content"]
    video_part = next(item for item in content if item["type"] == "video_url")
    assert video_part["video_url"]["url"] == "data:video/mp4;base64,dmlkZW8="
    assert (
        client.usage_snapshot()["by_operation"]["knowledge_extract_native_video"]["cost_usd"]
        == 0.001
    )



def test_robust_client_falls_back_when_guided_json_returns_wrong_top_level_type() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    client.config = LLMConfig(
        model="fake",
        compatibility_mode="generic",
        structured_output_mode="native",
        max_tokens=128,
        max_retries=1,
    )
    client._usage_lock = threading.Lock()
    client.reset_usage()
    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            raw = "[]" if len(requests) == 1 else '{"value":"ok"}'
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=raw),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    cost=0.0,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
            )

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    result = client._structured("Return payload.", Payload)

    assert result.value == "ok"
    assert "response_format" in requests[0]
    assert "response_format" not in requests[1]
    retry_text = requests[1]["messages"][1]["content"][0]["text"]
    assert "JSON schema:" in retry_text
    assert "previous response was invalid" in retry_text.lower()
