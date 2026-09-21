import pytest

from automatic_lecture_tex.tex_safety import (
    normalize_math_spans,
    validate_tex_source,
)


def test_normalize_repairs_lost_backslashes_in_common_math_tokens():
    value = (
        "Пусть mathbb{C} и lambda_0 neq 0 quad "
        "forall x рассматривается varepsilon_i."
    )

    normalized = normalize_math_spans(value)

    assert r"\(\mathbb{C}\)" in normalized
    assert r"\(\lambda_0\)" in normalized
    assert r"\(\neq\)" in normalized
    assert r"\(\quad\)" in normalized
    assert r"\(\forall x\)" in normalized
    assert r"\(\varepsilon_i\)" in normalized
    validate_tex_source(normalized)


def test_normalize_unwraps_math_symbol_from_text_command():
    normalized = normalize_math_spans(r"Получаем $\text{\lambda}=1$.")

    assert r"$\lambda=1$" in normalized
    validate_tex_source(normalized)


def test_tex_gate_rejects_unbalanced_braces():
    with pytest.raises(ValueError, match="unbalanced TeX braces"):
        validate_tex_source(r"\section{Незакрытая секция")


def test_tex_gate_rejects_surviving_serialization_damage():
    with pytest.raises(ValueError, match="raw command-like token"):
        validate_tex_source("Пусть mathbb{C} — поле.")
