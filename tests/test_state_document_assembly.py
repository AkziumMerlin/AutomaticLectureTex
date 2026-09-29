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
    return StateDocumentPlan.model_validate(
        {
            "blocks": [
                {
                    "type": "subsection",
                    "text": "Комплексно-линейные функционалы",
                    "source_observation_ids": ["obs_def"],
                },
                {
                    "type": "definition",
                    "text": (
                        "Функционал f на комплексном пространстве X называется "
                        "комплексно-линейным."
                    ),
                    "source_observation_ids": ["obs_def"],
                },
                {
                    "type": "formula",
                    "formula_observation_id": "obs_def",
                },
                {
                    "type": "proof",
                    "text": (
                        "Разложение функционала f на действительную и мнимую части "
                        "связывает u и v."
                    ),
                    "source_observation_ids": ["obs_step", "obs_formula"],
                },
                {
                    "type": "formula",
                    "formula_observation_id": "obs_formula",
                },
            ],
            "omissions": [
                {
                    "observation_id": "obs_duplicate",
                    "channel": "text",
                    "reason": "duplicate",
                },
                {
                    "observation_id": "obs_intermediate",
                    "channel": "both",
                    "reason": "intermediate",
                },
            ],
        }
    )


def test_document_block_json_schema_encodes_type_specific_required_fields():
    schema = StateDocumentPlan.model_json_schema()
    prose = schema["$defs"]["StateDocumentProseBlock"]
    formula = schema["$defs"]["StateDocumentFormulaBlock"]
    block_items = schema["properties"]["blocks"]["items"]

    assert "discriminator" in block_items
    assert set(prose["required"]) >= {"type", "text", "source_observation_ids"}
    assert "formula_observation_id" not in prose["properties"]
    assert set(formula["required"]) == {"type", "formula_observation_id"}
    assert "text" not in formula["properties"]
    assert "title" not in formula["properties"]


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


def test_document_plan_may_implicitly_omit_routine_remark_and_proof_channels():
    _, section, observations = _fixture()
    plan = _good_plan()
    plan.omissions = []

    notes, issues, stats = _validate_and_render_plan(
        plan,
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
    assert stats["omitted_text_channels"] >= 2
    assert stats["omitted_formula_channels"] >= 1


def test_document_plan_keeps_important_definition_fail_closed():
    _, section, observations = _fixture()
    plan = _good_plan()
    plan.blocks = [
        block
        for block in plan.blocks
        if not (
            getattr(block, "formula_observation_id", None) == "obs_def"
            or "obs_def" in getattr(block, "source_observation_ids", [])
        )
    ]
    plan.omissions = [
        item for item in plan.omissions if item.observation_id != "obs_def"
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
    assert any("important text channel for obs_def" in issue for issue in issues)
    assert any("important formula channel for obs_def" in issue for issue in issues)


def test_document_plan_allows_semantic_reordering_within_section():
    _, section, observations = _fixture()
    plan = _good_plan()
    proof = plan.blocks.pop(3)
    formula = plan.blocks.pop(3)
    plan.blocks.extend([formula, proof])

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

    assert issues == []
    assert notes is not None


def test_document_plan_allows_literal_source_equality_in_prose():
    _, section, observations = _fixture()
    observations[1].text = "Для частей функционала используется равенство u = v."
    plan = _good_plan()
    plan.blocks[3].text = "Для частей функционала используется равенство u = v."
    plan.blocks[3].source_observation_ids = ["obs_step"]

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

    assert issues == []
    assert notes is not None


def test_document_plan_may_omit_formula_when_important_observation_is_used_as_prose():
    _, section, observations = _fixture()
    plan = _good_plan()
    plan.blocks = [
        block
        for block in plan.blocks
        if getattr(block, "formula_observation_id", None) != "obs_def"
    ]

    notes, issues, stats = _validate_and_render_plan(
        plan,
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
    assert stats["omitted_formula_channels"] >= 1


def test_document_plan_allows_section_scoped_non_equality_relation_tokens():
    _, section, observations = _fixture()
    observations[0].text = "Рассматривается отображение f: X → X."
    plan = _good_plan()
    plan.blocks[3].text = "Отображение f действует X → X и связывает u и v."
    plan.blocks[3].source_observation_ids = ["obs_step"]

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

    assert issues == []
    assert notes is not None


def test_document_narration_gate_allows_passive_note_style():
    _, section, observations = _fixture()
    plan = _good_plan()
    plan.blocks[3].text = "Равенство записывается в двух эквивалентных формах."
    plan.blocks[3].source_observation_ids = ["obs_step"]

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

    assert issues == []
    assert notes is not None


def test_document_plan_may_omit_redundant_notation_without_explicit_accounting():
    _, section, observations = _fixture()
    observations.append(
        _observation(
            "obs_notation",
            start=5.0,
            kind=ObservationKind.NOTATION,
            text="Обозначение X используется для комплексного пространства.",
            latex=None,
        )
    )
    plan = _good_plan()

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

    assert issues == []
    assert notes is not None


def test_document_plan_may_omit_definition_when_document_covers_its_content():
    _, section, observations = _fixture()
    observations.append(
        _observation(
            "obs_duplicate_definition",
            start=5.0,
            kind=ObservationKind.DEFINITION,
            text=(
                "Функционал f на комплексном пространстве X называется "
                "комплексно-линейным."
            ),
            latex=r"f:X\to\mathbb{C}",
        )
    )
    plan = _good_plan()

    notes, issues, stats = _validate_and_render_plan(
        plan,
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
    assert stats["omitted_text_channels"] >= 1
    assert stats["omitted_formula_channels"] >= 1


def test_document_plan_does_not_hide_uncovered_definition():
    _, section, observations = _fixture()
    observations.append(
        _observation(
            "obs_new_definition",
            start=5.0,
            kind=ObservationKind.DEFINITION,
            text="Новый оператор T называется компактным при выполнении отдельного условия.",
            latex=None,
        )
    )
    plan = _good_plan()

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
    assert any("important text channel for obs_new_definition" in issue for issue in issues)


def test_document_plan_allows_section_level_notation_and_source_relations():
    _, section, observations = _fixture()
    observations[0].text = "Рассматривается пространство X и функционал f: X → ℂ."
    observations[1].text = "Для x∈X выполняется f(x) = 0."
    plan = _good_plan()
    plan.blocks[1].text = "Функционал f: X → ℂ рассматривается на пространстве X."
    plan.blocks[1].source_observation_ids = ["obs_def"]
    plan.blocks[3].text = "Для x∈X используется равенство f(x) = 0."
    plan.blocks[3].source_observation_ids = ["obs_step"]

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

    assert issues == []
    assert notes is not None


def test_document_narration_gate_does_not_reject_mathematical_transition_phrase():
    _, section, observations = _fixture()
    plan = _good_plan()
    plan.blocks[3].text = (
        "Переходя к супремуму, получаем оценку нормы функционала f."
    )
    plan.blocks[3].source_observation_ids = ["obs_step"]

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

    assert issues == []
    assert notes is not None


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
