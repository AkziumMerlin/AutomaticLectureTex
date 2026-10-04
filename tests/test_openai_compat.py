from __future__ import annotations

import sys
from types import ModuleType

import pytest

from automatic_lecture_tex.openai_compat import make_openai_client, resolve_api_key


def test_resolve_api_key_prefers_named_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    assert (
        resolve_api_key(api_key="ignored", api_key_env="OPENROUTER_API_KEY")
        == "sk-or-test"
    )


def test_missing_named_api_key_fails_fast(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        resolve_api_key(api_key=None, api_key_env="OPENROUTER_API_KEY")


def test_make_openai_client_forwards_base_url_headers_and_timeout(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    client = make_openai_client(
        base_url="https://openrouter.ai/api/v1",
        api_key=None,
        api_key_env="OPENROUTER_API_KEY",
        timeout_seconds=42.0,
        default_headers={"HTTP-Referer": "https://example.test"},
    )

    assert isinstance(client, FakeOpenAI)
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "sk-or-test"
    assert captured["timeout"] == 42.0
    assert captured["default_headers"]["HTTP-Referer"] == "https://example.test"
