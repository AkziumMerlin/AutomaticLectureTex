from automatic_lecture_tex.knowledge_pipeline import _assemble_state_section_deterministically
from automatic_lecture_tex.schemas import (
    EpisodeKind,
    LectureKnowledgeBase,
    LectureObservation,
    LectureOutline,
    ObservationKind,
    OutlineSection,
    SemanticEpisode,
)
from automatic_lecture_tex.state_canonicalization import CanonicalRenderPolicy
from automatic_lecture_tex.state_prose_compression import (
    ProseCompressionPlan,
    ProseCompressionSentence,
    ProseRedundancyGroupProposal,
    ProseSummaryGroupProposal,
    _summary_runs,
    _validate_redundancy_group,
    _validate_summary_group,
    run_state_prose_compression,
)


def _obs(
    observation_id: str,
    *,
    start: float,
    kind: ObservationKind,
    text: str,
    latex: str | None = None,
) -> LectureObservation:
    return LectureObservation(
        id=observation_id,
        episode_id="episode_0000",
        start=start,
        end=start + 1.0,
        kind=kind,
        text=text,
        latex=latex,
    )


def _kb(*observations: LectureObservation) -> LectureKnowledgeBase:
    return LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=list(observations),
        episodes=[
            SemanticEpisode(
                id="episode_0000",
                title="Intro",
                kind=EpisodeKind.TOPIC,
                start=0.0,
                end=100.0,
                observation_ids=[item.id for item in observations],
            )
        ],
    )


def _section() -> OutlineSection:
    return OutlineSection(
        id="topic_000",
        title="Введение",
        start=0.0,
        end=100.0,
        episode_ids=["episode_0000"],
    )


def test_summary_runs_bridge_transitions_but_stop_at_formal_content():
    observations = [
        _obs(
            "obs_a",
            start=1.0,
            kind=ObservationKind.REMARK,
            text="Перечисляются темы второй части курса.",
        ),
        _obs(
            "transition",
            start=2.0,
            kind=ObservationKind.TRANSITION,
            text="Лектор переходит к следующей записи.",
        ),
        _obs(
            "obs_b",
            start=3.0,
            kind=ObservationKind.REMARK,
            text="Исходное пространство X рассматривается как комплексное.",
        ),
        _obs(
            "obs_c",
            start=4.0,
            kind=ObservationKind.NOTATION,
            text="X обозначает комплексное пространство.",
        ),
        _obs(
            "obs_math",
            start=5.0,
            kind=ObservationKind.REMARK,
            text="Получаем v(ix) = u(x).",
        ),
        _obs(
            "obs_d",
            start=6.0,
            kind=ObservationKind.REMARK,
            text="Новая чисто текстовая тема.",
        ),
        _obs(
            "obs_e",
            start=7.0,
            kind=ObservationKind.REMARK,
            text="Её повторяют другими словами.",
        ),
        _obs(
            "obs_f",
            start=8.0,
            kind=ObservationKind.NOTATION,
            text="Для неё фиксируется обозначение.",
        ),
    ]

    runs = _summary_runs(observations, min_group_size=3)

    assert [[item.id for item in run] for run in runs] == [
        ["obs_a", "obs_b", "obs_c"],
        ["obs_d", "obs_e", "obs_f"],
    ]


def test_opening_narration_compresses_across_structural_transitions(tmp_path):
    observations = [
        _obs(
            "transition_a",
            start=1.0,
            kind=ObservationKind.TRANSITION,
            text="Начинается вторая часть курса.",
        ),
        _obs(
            "obs_topics",
            start=2.0,
            kind=ObservationKind.REMARK,
            text=(
                "Перечисляются нормированные пространства, сопряжённые пространства, "
                "спектр и спектральная теория операторов."
            ),
        ),
        _obs(
            "transition_b",
            start=3.0,
            kind=ObservationKind.TRANSITION,
            text="Начинается первый вопрос.",
        ),
        _obs(
            "obs_space",
            start=4.0,
            kind=ObservationKind.REMARK,
            text="Исходное пространство X рассматривается как комплексное пространство.",
        ),
        _obs(
            "obs_repeat",
            start=5.0,
            kind=ObservationKind.REMARK,
            text="Рассматриваемое пространство снова называется комплексным.",
        ),
        _obs(
            "obs_notation",
            start=6.0,
            kind=ObservationKind.NOTATION,
            text="X обозначает комплексное пространство.",
        ),
        _obs(
            "obs_definition",
            start=7.0,
            kind=ObservationKind.DEFINITION,
            text="Вводится линейный функционал.",
            latex=r"f:X\to\mathbb{C}",
        ),
    ]
    kb = _kb(*observations)
    outline = LectureOutline(sections=[_section()])
    plan = ProseCompressionPlan(
        summary_groups=[
            ProseSummaryGroupProposal(
                source_observation_ids=[
                    "obs_topics",
                    "obs_space",
                    "obs_repeat",
                    "obs_notation",
                ],
                sentences=[
                    ProseCompressionSentence(
                        text=(
                            "Во второй части курса рассматриваются нормированные и сопряжённые "
                            "пространства, спектр и спектральная теория операторов."
                        ),
                        source_observation_ids=["obs_topics"],
                    ),
                    ProseCompressionSentence(
                        text="Исходное пространство X рассматривается как комплексное.",
                        source_observation_ids=[
                            "obs_space",
                            "obs_repeat",
                            "obs_notation",
                        ],
                    ),
                ],
            )
        ]
    )

    class StubOrchestrator:
        output_language = "ru"

        def __init__(self):
            self.calls = 0

        def _structured(self, *args, **kwargs):
            self.calls += 1
            return plan

    orchestrator = StubOrchestrator()
    policy, stats, unresolved = run_state_prose_compression(
        orchestrator,
        kb=kb,
        outline=outline,
        work=tmp_path,
        llm_config={"model": "stub"},
        pipeline_version=10,
        min_group_size=3,
        max_sentences=2,
        max_ratio=0.90,
        max_summary_chars=700,
        force=False,
    )

    assert unresolved == []
    assert orchestrator.calls == 1
    assert stats["accepted_summary_groups"] == 1
    assert stats["summarized_observations"] == 4

    notes = _assemble_state_section_deterministically(
        kb,
        _section(),
        render_policy=CanonicalRenderPolicy(
            suppress_text_ids=["transition_a", "transition_b"]
        ),
        prose_policy=policy,
    )
    assert notes.blocks[0].latex == (
        "Во второй части курса рассматриваются нормированные и сопряжённые пространства, "
        "спектр и спектральная теория операторов. "
        "Исходное пространство X рассматривается как комплексное."
    )
    assert notes.blocks[1].latex == "Вводится линейный функционал."
    assert notes.blocks[2].latex == r"f:X\to\mathbb{C}"


def test_small_two_observation_summary_is_rejected():
    run = [
        _obs(
            "obs_a",
            start=1.0,
            kind=ObservationKind.REMARK,
            text="Пространство X комплексное.",
        ),
        _obs(
            "obs_b",
            start=2.0,
            kind=ObservationKind.NOTATION,
            text="X обозначает комплексное пространство.",
        ),
        _obs(
            "obs_c",
            start=3.0,
            kind=ObservationKind.REMARK,
            text="Ещё одно повторение.",
        ),
    ]
    proposal = ProseSummaryGroupProposal(
        source_observation_ids=["obs_a", "obs_b"],
        sentences=[
            ProseCompressionSentence(
                text="Пространство X рассматривается как комплексное.",
                source_observation_ids=["obs_a", "obs_b"],
            )
        ],
    )
    group, reason = _validate_summary_group(
        proposal,
        section_id="topic_000",
        runs=[run],
        used_ids=set(),
        min_group_size=3,
        max_sentences=2,
        max_ratio=0.90,
        max_summary_chars=700,
    )

    assert group is None
    assert "fewer than 3" in reason


def test_generated_summary_cannot_introduce_math_or_reconstruction_narration():
    run = [
        _obs(
            "obs_a",
            start=1.0,
            kind=ObservationKind.REMARK,
            text="Пространство X рассматривается как комплексное.",
        ),
        _obs(
            "obs_b",
            start=2.0,
            kind=ObservationKind.NOTATION,
            text="X обозначает комплексное пространство.",
        ),
        _obs(
            "obs_c",
            start=3.0,
            kind=ObservationKind.REMARK,
            text="Рассматриваемое пространство снова называется комплексным.",
        ),
    ]

    for text in [
        r"Получаем \(X=\mathbb{C}\).",
        "По OCR пространство X является комплексным.",
        "Пространство Y является комплексным.",
    ]:
        proposal = ProseSummaryGroupProposal(
            source_observation_ids=["obs_a", "obs_b", "obs_c"],
            sentences=[
                ProseCompressionSentence(
                    text=text,
                    source_observation_ids=["obs_a", "obs_b", "obs_c"],
                )
            ],
        )
        group, _ = _validate_summary_group(
            proposal,
            section_id="topic_000",
            runs=[run],
            used_ids=set(),
            min_group_size=3,
            max_sentences=2,
            max_ratio=0.90,
            max_summary_chars=700,
        )
        assert group is None


def test_selection_only_dedup_keeps_formula_only_in_prose():
    candidates = [
        _obs(
            "obs_rep",
            start=1.0,
            kind=ObservationKind.REMARK,
            text="Записи выражают одну и ту же связь между u и v.",
        ),
        _obs(
            "obs_formula_in_prose",
            start=2.0,
            kind=ObservationKind.REMARK,
            text="Имеем v(ix) = u(x), то есть части связаны.",
        ),
        _obs(
            "obs_with_latex",
            start=3.0,
            kind=ObservationKind.REMARK,
            text="Связь ещё раз поясняется.",
            latex=r"v(ix)=u(x)",
        ),
        _obs(
            "obs_repeat",
            start=4.0,
            kind=ObservationKind.REMARK,
            text="Та же связь снова поясняется устно.",
        ),
    ]
    proposal = ProseRedundancyGroupProposal(
        source_observation_ids=[
            "obs_rep",
            "obs_formula_in_prose",
            "obs_with_latex",
            "obs_repeat",
        ],
        representative_observation_id="obs_rep",
    )

    suppressed, reason = _validate_redundancy_group(
        proposal,
        candidates=candidates,
        used_ids=set(),
    )

    assert reason == ""
    assert suppressed == {"obs_with_latex", "obs_repeat"}
    assert "obs_formula_in_prose" not in suppressed


def test_selection_only_dedup_changes_only_text_channel():
    source = _obs(
        "obs_source",
        start=1.0,
        kind=ObservationKind.REMARK,
        text="Повторное пояснение одной связи.",
        latex=r"v(ix)=u(x)",
    )
    representative = _obs(
        "obs_rep",
        start=2.0,
        kind=ObservationKind.REMARK,
        text="Поясняется та же связь.",
    )
    kb = _kb(source, representative)

    from automatic_lecture_tex.state_prose_compression import ProseCompressionPolicy

    notes = _assemble_state_section_deterministically(
        kb,
        _section(),
        render_policy=CanonicalRenderPolicy(),
        prose_policy=ProseCompressionPolicy(suppress_text_ids=["obs_source"]),
    )

    assert all(block.latex != "Повторное пояснение одной связи." for block in notes.blocks)
    assert any(block.latex == r"v(ix)=u(x)" for block in notes.blocks)
    assert any(block.latex == "Поясняется та же связь." for block in notes.blocks)


def test_cached_plan_avoids_second_model_call(tmp_path):
    observations = [
        _obs(
            "obs_a",
            start=1.0,
            kind=ObservationKind.REMARK,
            text="Пространство X рассматривается как комплексное.",
        ),
        _obs(
            "obs_b",
            start=2.0,
            kind=ObservationKind.NOTATION,
            text="X обозначает комплексное пространство.",
        ),
        _obs(
            "obs_c",
            start=3.0,
            kind=ObservationKind.REMARK,
            text="Рассматриваемое пространство снова называется комплексным.",
        ),
    ]
    kb = _kb(*observations)
    outline = LectureOutline(sections=[_section()])
    plan = ProseCompressionPlan(
        summary_groups=[
            ProseSummaryGroupProposal(
                source_observation_ids=["obs_a", "obs_b", "obs_c"],
                sentences=[
                    ProseCompressionSentence(
                        text="Пространство X рассматривается как комплексное.",
                        source_observation_ids=["obs_a", "obs_b", "obs_c"],
                    )
                ],
            )
        ]
    )

    class StubOrchestrator:
        output_language = "ru"

        def __init__(self):
            self.calls = 0

        def _structured(self, *args, **kwargs):
            self.calls += 1
            return plan

    orchestrator = StubOrchestrator()
    kwargs = dict(
        kb=kb,
        outline=outline,
        work=tmp_path,
        llm_config={"model": "stub"},
        pipeline_version=10,
        min_group_size=3,
        max_sentences=2,
        max_ratio=0.90,
        max_summary_chars=700,
        force=False,
    )
    run_state_prose_compression(orchestrator, **kwargs)
    _, stats, _ = run_state_prose_compression(orchestrator, **kwargs)

    assert orchestrator.calls == 1
    assert stats["cache_hits"] == 1
