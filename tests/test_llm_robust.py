from automatic_lecture_tex.llm_robust import (
    _backend_output_token_ceiling,
    _is_context_overflow_error,
    _next_context_output_budget,
)


def test_vllm_hard_max_tokens_error_is_treated_as_context_budget_rejection():
    error = RuntimeError(
        "max_tokens=24576 cannot be greater than "
        "max_model_len=max_total_tokens=20000. "
        "Please request fewer output tokens."
    )

    assert _is_context_overflow_error(error)
    assert _backend_output_token_ceiling(error) == 20000
    assert (
        _next_context_output_budget(
            error,
            current_max_tokens=24576,
            last_accepted_max_tokens=12288,
        )
        == 12288
    )


def test_context_budget_retry_falls_back_geometrically_without_prior_success():
    error = RuntimeError(
        "max_tokens=24576 cannot be greater than max_total_tokens=20000."
    )

    assert (
        _next_context_output_budget(
            error,
            current_max_tokens=24576,
            last_accepted_max_tokens=None,
        )
        == 12288
    )
