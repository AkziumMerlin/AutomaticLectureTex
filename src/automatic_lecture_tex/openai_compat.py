from __future__ import annotations

import os
from typing import Any


def resolve_api_key(
    *,
    api_key: str | None,
    api_key_env: str | None,
    fallback: str = "EMPTY",
) -> str:
    """Resolve an OpenAI-compatible API key without requiring secrets in YAML."""

    if api_key_env:
        value = os.getenv(api_key_env, "").strip()
        if not value:
            raise RuntimeError(
                f"OpenAI-compatible API key environment variable {api_key_env!r} is not set"
            )
        return value
    value = (api_key or "").strip()
    return value or fallback


def make_openai_client(
    *,
    base_url: str,
    api_key: str | None,
    api_key_env: str | None = None,
    timeout_seconds: float = 300.0,
    default_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI-compatible backend requires the openai Python package"
        ) from exc

    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "api_key": resolve_api_key(api_key=api_key, api_key_env=api_key_env),
        "timeout": timeout_seconds,
        "default_headers": default_headers or None,
    }

    if proxy_url:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Proxying an OpenAI-compatible backend requires httpx"
            ) from exc
        try:
            kwargs["http_client"] = httpx.Client(
                proxy=proxy_url,
                timeout=timeout_seconds,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SOCKS proxy support requires socksio; reinstall the project dependencies "
                "or install socksio before using a socks5 proxy"
            ) from exc

    return OpenAI(**kwargs)
