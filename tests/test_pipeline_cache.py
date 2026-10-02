from pathlib import Path

from automatic_lecture_tex import graph_revision_pipeline, graph_revision_render, graph_surface_writer
from automatic_lecture_tex.config import load_config
from automatic_lecture_tex.pipeline import Pipeline
from automatic_lecture_tex.schemas import Transcript


def _empty_transcript() -> Transcript:
    return Transcript(lecture_id="lecture_01", language="ru", segments=[])


def test_mutable_graph_runtime_version_invalidates_lecture_ir_cache(monkeypatch) -> None:
    config = load_config(Path("configs/functional_analysis_vk_lecture01_state_20s.yaml"))
    pipeline = Pipeline(config)
    transcript = _empty_transcript()

    before = pipeline._ir_fingerprint(transcript, {})
    monkeypatch.setattr(
        graph_revision_pipeline,
        "GRAPH_REVISION_RUNTIME_VERSION",
        graph_revision_pipeline.GRAPH_REVISION_RUNTIME_VERSION + 1,
    )
    after = pipeline._ir_fingerprint(transcript, {})

    assert before != after


def test_graph_runtime_version_does_not_invalidate_linear_ir_cache(monkeypatch) -> None:
    config = load_config(Path("configs/functional_analysis_vk_lecture01.yaml"))
    pipeline = Pipeline(config)
    transcript = _empty_transcript()

    before = pipeline._ir_fingerprint(transcript, {})
    monkeypatch.setattr(
        graph_revision_pipeline,
        "GRAPH_REVISION_RUNTIME_VERSION",
        graph_revision_pipeline.GRAPH_REVISION_RUNTIME_VERSION + 1,
    )
    after = pipeline._ir_fingerprint(transcript, {})

    assert before == after



def test_surface_versions_invalidate_mutable_graph_lecture_ir_cache(monkeypatch) -> None:
    config = load_config(Path("configs/functional_analysis_vk_lecture01_state_20s.yaml"))
    pipeline = Pipeline(config)
    transcript = _empty_transcript()

    before = pipeline._ir_fingerprint(transcript, {})
    monkeypatch.setattr(
        graph_surface_writer,
        "GRAPH_SURFACE_WRITER_VERSION",
        graph_surface_writer.GRAPH_SURFACE_WRITER_VERSION + 1,
    )
    writer_changed = pipeline._ir_fingerprint(transcript, {})

    monkeypatch.setattr(
        graph_revision_render,
        "GRAPH_SURFACE_RENDER_VERSION",
        graph_revision_render.GRAPH_SURFACE_RENDER_VERSION + 1,
    )
    renderer_changed = pipeline._ir_fingerprint(transcript, {})

    assert before != writer_changed
    assert writer_changed != renderer_changed


def test_surface_versions_do_not_invalidate_linear_ir_cache(monkeypatch) -> None:
    config = load_config(Path("configs/functional_analysis_vk_lecture01.yaml"))
    pipeline = Pipeline(config)
    transcript = _empty_transcript()

    before = pipeline._ir_fingerprint(transcript, {})
    monkeypatch.setattr(
        graph_surface_writer,
        "GRAPH_SURFACE_WRITER_VERSION",
        graph_surface_writer.GRAPH_SURFACE_WRITER_VERSION + 1,
    )
    monkeypatch.setattr(
        graph_revision_render,
        "GRAPH_SURFACE_RENDER_VERSION",
        graph_revision_render.GRAPH_SURFACE_RENDER_VERSION + 1,
    )
    after = pipeline._ir_fingerprint(transcript, {})

    assert before == after
