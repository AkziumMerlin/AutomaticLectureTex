from pathlib import Path

from automatic_lecture_tex.config import load_config
from automatic_lecture_tex.latex import render_block
from automatic_lecture_tex.schemas import (
    BlockType,
    EpisodeKind,
    EpisodeStatus,
    LectureKnowledgeBase,
    LectureObservation,
    ObservationKind,
    OutlineSection,
    SemanticEpisode,
)
from automatic_lecture_tex.state_canonicalization import CanonicalRenderPolicy
from automatic_lecture_tex.state_document_assembly import (
    StateDocumentBlockProposal,
    StateDocumentOmission,
    StateDocumentPlan,
    _validate_and_render_plan,
    build_state_document_section,
)


def _observation(
    observation_id: str,
    *,
    start: float,
    kind: ObservationKind,
    text: str,
    latex: str | None = None,
) -> LectureObservation:
    return LectureObservation(
        id=observation_id,
        window_id="window",
        episode_id="episode_0",
        start=start,
        end=start + 1.0,
        kind=kind,
        text=text,
        latex=latex,
    )


def _fixture():
    observations = [
        _observation(
            "obs_def",
            start=0.0,
            kind=ObservationKind.DEFINITION,
            text="Функционал f на комплексном пространстве X называется комплексно-линейным.",
            latex=r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)",
        ),
        _observation(
            "obs_step",
            start=1.0,
            kind=ObservationKind.PROOF_STEP,
            text="Разложение функционала f на действительную и мнимую части связывает u и v.",
        ),
        _observation(
            "obs_formula",
            start=2.0,
            kind=ObservationKind.PROOF_STEP,
            text="Из комплексной линейности f получается связь частей u и v.",
            latex=r"v(ix)=u(x),\qquad v(x)=-u(ix)",
        ),
        _observation(
            "obs_duplicate",
            start=3.0,
            kind=ObservationKind.REMARK,
            text="Та же связь между u и v повторяется ещё раз.",
        ),
        _observation(
            "obs_intermediate",
            start=4.0,
            kind=ObservationKind.PROOF_STEP,
            text="Промежуточная алгебраическая строка для f.",
            latex=r"f(ix)=u(ix)+iv(ix)",
        ),
    ]
    episode = SemanticEpisode(
        id="episode_0",
        title="Комплексные функционалы",
        kind=EpisodeKind.DERIVATION,
        start=0.0,
        end=5.0,
        status=EpisodeStatus.CLOSED,
        observation_ids=[item.id for item in observations],
    )
    kb = LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=observations,
        episodes=[episode],
    )
    section = OutlineSection(
        id="topic_000",
        title="Комплексные функционалы",
        start=0.0,
        end=5.0,
        episode_ids=["episode_0"],
    )
    return kb, section, observations


def _good_plan() -> StateDocumentPlan:
    return StateDocumentPlan(
        blocks=[
            StateDocumentBlockProposal(
                type="subsection",
                text="Комплексно-линейные функционалы",
                source_observation_ids=["obs_def"],
            ),
            StateDocumentBlockProposal(
                type="definition",
                text=(
                    "Функционал f на комплексном пространстве X называется "
                    "комплексно-линейным."
                ),
                source_observation_ids=["obs_def"],
            ),
            StateDocumentBlockProposal(
                type="formula",
                formula_observation_id="obs_def",
            ),
            StateDocumentBlockProposal(
                type="proof",
                text=(
                    "Разложение функционала f на действительную и мнимую части "
                    "связывает u и v."
                ),
                source_observation_ids=["obs_step", "obs_formula"],
            ),
            StateDocumentBlockProposal(
                type="formula",
                formula_observation_id="obs_formula",
            ),
        ],
        omissions=[
            StateDocumentOmission(
                observation_id="obs_duplicate",
                channel="text",
                reason="duplicate",
            ),
            StateDocumentOmission(
                observation_id="obs_intermediate",
                channel="both",
                reason="intermediate",
            ),
        ],
    )


def test_document_plan_preserves_formula_latex_verbatim():
    kb, section, observations = _fixture()
    notes, issues, stats = _validate_and_render_plan(
        _good_plan(),
        section=section,
        observations=observations,
        render_policy=CanonicalRenderPolicy(),
        output_language="ru",
        max_prose_ratio=0.90,
        max_remarks_fraction=0.25,
        max_blocks=20,
    )

    assert issues == []
    assert notes is not None
    assert [block.type for block in notes.blocks] == [
        BlockType.SUBSECTION,
        BlockType.DEFINITION,
        BlockType.EQUATION,
        BlockType.PROOF,
        BlockType.EQUATION,
    ]
    assert notes.blocks[2].latex == r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)"
    assert notes.blocks[4].latex == r"v(ix)=u(x),\qquad v(x)=-u(ix)"
    assert stats["remarks"] == 0
    assert stats["omitted_formula_channels"] == 1


def test_document_plan_rejects_new_symbol_in_generated_prose():
    _, section, observations = _fixture()
    plan = _good_plan()
    plan.blocks[1].text = "Функционал g на пространстве Y называется комплексно-линейным."

    notes, issues, _ = _validate_and_render_plan(
        plan,
        section=section,
        observations=observations,
        render_policy=CanonicalRenderPolicy(),
        output_language="ru",
        max_prose_ratio=0.90,
        max_remarks_fraction=0.25,
        max_blocks=20,
    )

    assert notes is None
    assert any("new standalone Latin symbols" in issue for issue in issues)


def test_document_plan_requires_channel_accounting():
    _, section, observations = _fixture()
    plan = _good_plan()
    plan.omissions = [
        item for item in plan.omissions if item.observation_id != "obs_duplicate"
    ]

    notes, issues, _ = _validate_and_render_plan(
        plan,
        section=section,
        observations=observations,
        render_policy=CanonicalRenderPolicy(),
        output_language="ru",
        max_prose_ratio=0.90,
        max_remarks_fraction=0.25,
        max_blocks=20,
    )

    assert notes is None
    assert any("obs_duplicate" in issue and "neither used nor omitted" in issue for issue in issues)


def test_document_assembly_is_cached(tmp_path):
    kb, section, _ = _fixture()

    class StubOrchestrator:
        output_language = "ru"

        def __init__(self):
            self.calls = 0

        def _structured(self, prompt, schema, **kwargs):
            self.calls += 1
            assert kwargs["operation"] == "state_document_assembly"
            return _good_plan()

    orchestrator = StubOrchestrator()
    kwargs = dict(
        kb=kb,
        section=section,
        render_policy=CanonicalRenderPolicy(),
        work=tmp_path,
        llm_config={"model": "stub"},
        max_prose_ratio=0.90,
        max_remarks_fraction=0.25,
        max_blocks=20,
        force=False,
    )
    first, first_stats, first_unresolved = build_state_document_section(
        orchestrator,
        **kwargs,
    )
    second, second_stats, second_unresolved = build_state_document_section(
        orchestrator,
        **kwargs,
    )

    assert first is not None
    assert second is not None
    assert first_unresolved == []
    assert second_unresolved == []
    assert orchestrator.calls == 1
    assert first_stats["model_calls"] == 1
    assert second_stats["cache_hits"] == 1


def test_subsection_block_renders_as_heading():
    kb, section, observations = _fixture()
    notes, issues, _ = _validate_and_render_plan(
        _good_plan(),
        section=section,
        observations=observations,
        render_policy=CanonicalRenderPolicy(),
        output_language="ru",
        max_prose_ratio=0.90,
        max_remarks_fraction=0.25,
        max_blocks=20,
    )
    assert issues == []
    assert notes is not None
    rendered = render_block(notes.blocks[0])
    assert rendered == "\\subsection{Комплексно-линейные функционалы}\n"


def test_functional_analysis_20s_config_uses_document_assembly():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "functional_analysis_vk_lecture01_state_20s.yaml"
    )
    config = load_config(config_path)

    assert config.notes.state_section_assembly == "document"
    assert config.notes.state_prose_compression_enabled is False
    assert config.notes.state_document_max_prose_ratio == 0.45
    assert config.notes.state_document_max_remarks_fraction == 0.20
