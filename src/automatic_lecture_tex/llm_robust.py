from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import TypeVar

from openai import BadRequestError
from pydantic import BaseModel
from pydantic_core import ValidationError

from .llm import LectureModelClient as BaseLectureModelClient
from .llm import SYSTEM
from .util import strip_thinking_and_fences

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


_MAX_CONTEXT_RE = re.compile(
    r"maximum context length is\\s+(\\d+)\\s+tokens",
    re.IGNORECASE,
)
_EXPLICIT_MAX_TOKENS_CAP_RE = re.compile(
    r"max_tokens\\s*=\\s*\\d+\\s+cannot be greater than.*?"
    r"(?:max_model_len\\s*=\\s*)?(?:max_total_tokens\\s*=\\s*)?(\\d+)",
    re.IGNORECASE,
)


def _explicit_max_tokens_ceiling(error: Exception) -> int | None:
    """Extract an output-token ceiling explicitly reported by an OpenAI-compatible backend."""

    match = _EXPLICIT_MAX_TOKENS_CAP_RE.search(str(error))
    if match is None:
        return None
    ceiling = int(match.group(1))
    return ceiling if ceiling > 0 else None


def _is_context_overflow_error(error: Exception) -> bool:
    """Recognize backend context/output-budget rejections across common server formats."""

    if _explicit_max_tokens_ceiling(error) is not None:
        return True

    message = str(error)
    lowered = message.lower()
    return bool(
        _MAX_CONTEXT_RE.search(message)
        and ("requested" in lowered or "request fewer" in lowered)
        and ("output tokens" in lowered or "completion tokens" in lowered)
    )


def _json_structure_incomplete(raw: str) -> bool:
    """Return True only when a JSON object/array is structurally unfinished.

    This is a truncation detector, not a JSON repairer. Mismatched closing delimiters are treated as
    malformed rather than incomplete so ordinary syntax errors keep the normal retry budget.
    """

    text = strip_thinking_and_fences(raw)
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return False

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text[min(starts) :]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return False
            opener = stack.pop()
            if (opener, char) not in {("{", "}"), ("[", "]")}:
                return False

    return in_string or bool(stack)


def _looks_like_truncated_json(raw: str, exc: json.JSONDecodeError) -> bool:
    if not raw.strip():
        return False
    if not _json_structure_incomplete(raw):
        return False
    if exc.msg.startswith("Unterminated string"):
        return True

    text = strip_thinking_and_fences(raw)
    near_end = exc.pos >= max(0, len(text) - 128)
    missing_top_level_close = not text.rstrip().endswith(("}", "]"))
    return near_end or missing_top_level_close


class LectureModelClient(BaseLectureModelClient):
    """Lecture model client with backend-independent structured-output recovery.

    Some OpenAI-compatible servers return ``finish_reason='stop'`` for an abruptly cut JSON object.
    Besides the backend signal, infer truncation from structurally unfinished JSON and regenerate the
    whole object with a larger bounded output budget. If server-side guided decoding itself returns
    incomplete JSON with ``stop``, retry using an explicit schema in the prompt and local validation.
    Partial JSON is never locally completed.
    """

    def _structured(
        self,
        prompt: str,
        schema: type[T],
        images: list[Path] | None = None,
        max_tokens: int | None = None,
        *,
        guided_json: bool = True,
        operation: str = "structured",
    ) -> T:
        schema_instruction = "\nJSON schema:\n" + json.dumps(
            schema.model_json_schema(), ensure_ascii=False, separators=(",", ":")
        )

        def build_instruction(use_guided_json: bool) -> str:
            explicit_schema = "" if use_guided_json else schema_instruction
            return (
                f"{prompt}{explicit_schema}\n\n"
                "Return only the JSON object requested by the response schema."
            )

        use_guided_json = guided_json
        content: list[dict] = [{"type": "text", "text": build_instruction(use_guided_json)}]
        for image in images or []:
            mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )

        current_max_tokens = max_tokens or self.config.max_tokens
        max_retry_tokens = current_max_tokens * (2**self.config.max_retries)
        parse_error: json.JSONDecodeError | ValidationError | None = None
        previous_truncated = False
        previous_guided_failure = False
        context_output_ceiling: int | None = None
        last_accepted_max_tokens: int | None = None

        for attempt in range(self.config.max_retries + 1):
            if attempt and parse_error is not None:
                if previous_truncated:
                    failure = (
                        "The previous response ended before the complete JSON object was produced. "
                        f"The new output budget is {current_max_tokens} tokens."
                    )
                else:
                    failure = f"The previous response was invalid ({parse_error})."
                if previous_guided_failure:
                    failure += " Server-side guided JSON was disabled for this retry."
                content[0]["text"] = (
                    f"{build_instruction(use_guided_json)}\n\n{failure} "
                    "Regenerate the complete object from the beginning; do not continue or repair "
                    "the previous partial JSON."
                )

            request_kwargs = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": content},
                ],
                "temperature": self.config.temperature,
                "max_tokens": current_max_tokens,
                "extra_body": self._extra_body(),
            }
            if use_guided_json:
                request_kwargs["response_format"] = self._response_format(schema)

            while True:
                request_kwargs["max_tokens"] = current_max_tokens
                try:
                    response = self.client.chat.completions.create(**request_kwargs)
                    last_accepted_max_tokens = current_max_tokens
                    break
                except BadRequestError as exc:
                    if not _is_context_overflow_error(exc):
                        raise

                    explicit_ceiling = _explicit_max_tokens_ceiling(exc)
                    if explicit_ceiling is not None and explicit_ceiling < current_max_tokens:
                        next_max_tokens = explicit_ceiling
                        recovery = "backend-reported"
                    elif (
                        last_accepted_max_tokens is not None
                        and last_accepted_max_tokens < current_max_tokens
                    ):
                        next_max_tokens = last_accepted_max_tokens
                        recovery = "previously-accepted"
                    else:
                        next_max_tokens = current_max_tokens // 2
                        recovery = "geometric"

                    if next_max_tokens < 256 or next_max_tokens >= current_max_tokens:
                        raise
                    context_output_ceiling = (
                        next_max_tokens
                        if context_output_ceiling is None
                        else min(context_output_ceiling, next_max_tokens)
                    )
                    logger.warning(
                        "[%s] backend rejected max_tokens=%d; retrying with %s ceiling "
                        "max_tokens=%d",
                        operation,
                        current_max_tokens,
                        recovery,
                        context_output_ceiling,
                    )
                    current_max_tokens = context_output_ceiling

            self._record_usage(operation, response)

            choice = response.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            backend_truncated = finish_reason == "length" or (
                finish_reason is None
                and current_max_tokens > 0
                and completion_tokens >= current_max_tokens
            )

            try:
                return self._parse_json(raw, schema)
            except (json.JSONDecodeError, ValidationError) as exc:
                parse_error = exc
                inferred_truncated = (
                    isinstance(exc, json.JSONDecodeError)
                    and _looks_like_truncated_json(raw, exc)
                )
                truncated = backend_truncated or inferred_truncated
                previous_truncated = truncated
                guided_failure = (
                    use_guided_json
                    and inferred_truncated
                    and not backend_truncated
                    and finish_reason == "stop"
                )
                previous_guided_failure = guided_failure
                logger.warning(
                    "[%s] structured parse failed: finish_reason=%r completion_tokens=%d "
                    "max_tokens=%d raw_chars=%d inferred_truncated=%s guided_json=%s: %s",
                    operation,
                    finish_reason,
                    completion_tokens,
                    current_max_tokens,
                    len(raw),
                    inferred_truncated,
                    use_guided_json,
                    exc,
                )
                if guided_failure:
                    logger.warning(
                        "[%s] guided JSON returned structurally incomplete output; "
                        "falling back to prompt-schema JSON",
                        operation,
                    )
                    use_guided_json = False
                retry_ceiling = (
                    min(max_retry_tokens, context_output_ceiling)
                    if context_output_ceiling is not None
                    else max_retry_tokens
                )
                if truncated and current_max_tokens < retry_ceiling:
                    next_max_tokens = min(retry_ceiling, current_max_tokens * 2)
                    logger.warning(
                        "[%s] retrying structurally incomplete output with max_tokens=%d",
                        operation,
                        next_max_tokens,
                    )
                    current_max_tokens = next_max_tokens

        assert parse_error is not None
        raise parse_error
