import json

import pytest
from pydantic import BaseModel

from automatic_lecture_tex.llm import LectureModelClient


class LatexPayload(BaseModel):
    latex: str


def _client() -> LectureModelClient:
    return LectureModelClient.__new__(LectureModelClient)


def test_parse_json_recovers_raw_tab_from_single_slash_theta() -> None:
    raw = '{"latex":"$x ' + "\t" + 'heta = 0$"}'

    parsed = _client()._parse_json(raw, LatexPayload)

    assert parsed.latex == r"$x \theta = 0$"


def test_parse_json_recovers_raw_backspace_from_single_slash_beta() -> None:
    raw = '{"latex":"$' + "\b" + 'eta x$"}'

    parsed = _client()._parse_json(raw, LatexPayload)

    assert parsed.latex == r"$\beta x$"


def test_parse_json_recovers_raw_newline_from_single_slash_nabla() -> None:
    raw = '{"latex":"$' + "\n" + 'abla f = 0$"}'

    parsed = _client()._parse_json(raw, LatexPayload)

    assert parsed.latex == r"$\nabla f = 0$"


def test_parse_json_repairs_invalid_single_slash_latex_escape() -> None:
    raw = r'{"latex":"$\alpha + \lambda + \mu$"}'

    parsed = _client()._parse_json(raw, LatexPayload)

    assert parsed.latex == r"$\alpha + \lambda + \mu$"


def test_parse_json_preserves_real_multiline_prose() -> None:
    raw = '{"latex":"первая строка\nвторая строка"}'.replace(r"\n", "\n")

    parsed = _client()._parse_json(raw, LatexPayload)

    assert parsed.latex == "первая строка\nвторая строка"


def test_parse_json_does_not_hide_structural_truncation() -> None:
    with pytest.raises(json.JSONDecodeError):
        _client()._parse_json('{"latex":"x"', LatexPayload)
