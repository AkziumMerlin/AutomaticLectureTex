from pathlib import Path

from automatic_lecture_tex.config import load_config


def test_functional_analysis_config_uses_linear_pipeline():
    config = load_config(Path("configs/functional_analysis_vk_lecture01.yaml"))

    assert config.notes.architecture == "linear"
    assert config.asr.backend == "faster_whisper"
    assert config.asr.model == "large-v3"
    assert config.llm.math_audit is False
    assert config.notes.linear_correction_scan_enabled is True
    assert config.vision.temporal_composite_enabled is False
