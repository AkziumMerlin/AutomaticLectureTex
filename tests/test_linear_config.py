from pathlib import Path

from automatic_lecture_tex.config import load_config


def test_functional_analysis_config_uses_pre_pr2_linear_baseline():
    config = load_config(Path("configs/functional_analysis_vk_lecture01.yaml"))

    assert config.notes.architecture == "linear"
    assert config.asr.backend == "faster_whisper"
    assert config.asr.model == "large-v3"
    assert "функциональный анализ" in config.asr.hotwords
    assert "слабая сходимость" in config.asr.hotwords
    assert config.notes.chunk_target_seconds == 180
    assert config.notes.visual_rule_selector is True
    assert config.notes.visual_llm_selector is False
    assert config.llm.math_audit is True
    assert config.llm.math_audit_min_equals == 4
    assert config.notes.linear_correction_scan_enabled is True
    assert config.vision.temporal_composite_enabled is False
