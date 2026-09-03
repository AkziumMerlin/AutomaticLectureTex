import json

from pydantic import BaseModel

from automatic_lecture_tex.llm import LectureModelClient


class LatexPayload(BaseModel):
    latex: str


def test_response_format_uses_json_schema() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    response_format = client._response_format(LatexPayload)

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "LatexPayload"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == LatexPayload.model_json_schema()


def test_parse_json_preserves_latex_backslashes() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    latex = r"\theta + \frac{1}{2} + \beta"
    raw = json.dumps({"latex": latex})

    parsed = client._parse_json(raw, LatexPayload)

    assert parsed.latex == latex
