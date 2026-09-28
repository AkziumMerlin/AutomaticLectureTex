from automatic_lecture_tex.knowledge_pipeline import _assemble_state_section_deterministically
from automatic_lecture_tex.schemas import (
    EpisodeKind,
    LectureKnowledgeBase,
    LectureObservation,
    ObservationKind,
    OutlineSection,
    SemanticEpisode,
)
from automatic_lecture_tex.state_canonicalization import (
    CanonicalObservationRelation,
    CanonicalRenderPolicy,
    _build_render_policy,
    run_state_canonicalization,
)


def _kb(*observations: LectureObservation) -> LectureKnowledgeBase:
    episode = SemanticEpisode(
        id="episode_0000",
        title="Topic",
        kind=EpisodeKind.TOPIC,
        start=0.0,
        end=100.0,
        observation_ids=[item.id for item in observations],
    )
    return LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=list(observations),
        episodes=[episode],
    )


def _obs(
    observation_id: str,
    *,
    start: float,
    kind: ObservationKind = ObservationKind.EQUATION,
    text: str = "",
    latex: str | None = None,
    target_observation_id: str | None = None,
) -> LectureObservation:
    return LectureObservation(
        id=observation_id,
        episode_id="episode_0000",
        start=start,
        end=start + 1.0,
        kind=kind,
        text=text,
        latex=latex,
        target_observation_id=target_observation_id,
    )


def test_canonicalization_schema_cannot_rewrite_mathematics():
    properties = CanonicalObservationRelation.model_json_schema()["properties"]

    assert "text" not in properties
    assert "latex" not in properties
    assert "replacement_latex" not in properties
    assert "semantic_text" not in properties


def test_exact_duplicate_suppresses_only_formula_channel_and_keeps_state():
    source = _obs(
        "obs_a",
        start=1.0,
        text="Первое доказательное пояснение.",
        latex=r"\|f\|=\sup_{\|x\|\le1}|f(x)|",
    )
    target = _obs(
        "obs_b",
        start=2.0,
        text="Повтор формулы.",
        latex=r"\|f\|=\sup_{\|x\|\le1}|f(x)|",
    )
    kb = _kb(source, target)

    policy, stats, relations = _build_render_policy(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="duplicate",
            )
        ],
    )

    assert [item.id for item in kb.observations] == ["obs_a", "obs_b"]
    assert policy.suppress_latex_ids == ["obs_a"]
    assert policy.suppress_text_ids == []
    assert stats["suppressed_latex_blocks"] == 1
    assert relations[0]["suppressed_channels"] == ["latex"]


def test_incomplete_formula_suppression_preserves_source_prose():
    source = _obs(
        "obs_a",
        start=1.0,
        text="Начинается определение окрестности.",
        latex=r"V(x)=\{y\in X:|\varphi(y)-\varphi(x)|\cdots\}",
    )
    target = _obs(
        "obs_b",
        start=2.0,
        text="Определение дописано.",
        latex=r"V(x)=\{y\in X:|\varphi(y)-\varphi(x)|<\varepsilon\}",
    )
    kb = _kb(source, target)

    policy, _, _ = _build_render_policy(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="intermediate",
            )
        ],
    )

    assert policy.suppress_latex_ids == ["obs_a"]
    assert policy.suppress_text_ids == []


def test_ldots_inside_complete_formula_is_not_an_incomplete_board_marker():
    source = _obs(
        "obs_a",
        start=1.0,
        latex=r"\beta_\Phi=\{V(y_1),\ldots,V(y_m)\}",
    )
    target = _obs(
        "obs_b",
        start=2.0,
        latex=r"\widetilde{\beta}_\Phi=\{V(x,\varphi_1),\ldots,V(x,\varphi_m)\}",
    )
    kb = _kb(source, target)

    policy, stats, relations = _build_render_policy(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="intermediate",
            )
        ],
    )

    assert policy == CanonicalRenderPolicy()
    assert stats["rejected_unsafe_relation"] == 1
    assert relations[0]["host_verified"] is False


def test_incomplete_formula_is_not_hidden_by_another_incomplete_formula():
    source = _obs("obs_a", start=1.0, latex=r"V(x)=\{y:\varphi(y)\cdots\}")
    target = _obs("obs_b", start=2.0, latex=r"V(x)=\{z:\varphi(z)\cdots\}")
    kb = _kb(source, target)

    policy, stats, _ = _build_render_policy(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="intermediate",
            )
        ],
    )

    assert policy == CanonicalRenderPolicy()
    assert stats["rejected_unsafe_relation"] == 1


def test_semantic_supersession_is_audit_only_without_explicit_correction():
    source = _obs("obs_a", start=1.0, text="old", latex="x=1")
    target = _obs("obs_b", start=50.0, text="new", latex="x=2")
    kb = _kb(source, target)

    policy, stats, relations = _build_render_policy(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="supersedes",
            )
        ],
    )

    assert policy == CanonicalRenderPolicy()
    assert stats["flagged_supersedes"] == 1
    assert relations[0]["applied_as"] == "audit_only"


def test_explicit_correction_suppresses_old_render_channels_without_rewriting():
    source = _obs("obs_a", start=1.0, text="Старое утверждение.", latex="x=1")
    correction = _obs(
        "obs_b",
        start=2.0,
        kind=ObservationKind.CORRECTION,
        text="Исправление.",
        latex="x=2",
        target_observation_id="obs_a",
    )
    kb = _kb(source, correction)

    policy, _, relations = _build_render_policy(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="supersedes",
            )
        ],
    )

    assert policy.suppress_text_ids == ["obs_a"]
    assert policy.suppress_latex_ids == ["obs_a"]
    assert relations[0]["host_verified"] is True


def test_transition_is_hidden_from_render_but_remains_structural_evidence():
    transition = _obs(
        "obs_a",
        start=1.0,
        kind=ObservationKind.TRANSITION,
        text="Перейдём к следующему вопросу.",
    )
    kb = _kb(transition)

    policy, _, _ = _build_render_policy(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                relation="meta",
            )
        ],
    )

    assert policy.suppress_text_ids == ["obs_a"]
    assert [item.id for item in kb.observations] == ["obs_a"]
    assert kb.episodes[0].observation_ids == ["obs_a"]


def test_deterministic_renderer_applies_channels_independently():
    source = _obs(
        "obs_a",
        start=1.0,
        text="Полезное пояснение сохраняется.",
        latex="x=1",
    )
    target = _obs("obs_b", start=2.0, text="Формула дописана.", latex="x=1")
    kb = _kb(source, target)
    section = OutlineSection(
        id="topic_000",
        title="Раздел",
        start=0.0,
        end=10.0,
        episode_ids=["episode_0000"],
    )
    policy = CanonicalRenderPolicy(suppress_latex_ids=["obs_a"])

    notes = _assemble_state_section_deterministically(kb, section, render_policy=policy)

    assert any(block.latex == "Полезное пояснение сохраняется." for block in notes.blocks)
    assert sum(block.latex == "x=1" for block in notes.blocks) == 1


def test_normal_run_does_not_call_llm_semantic_audit(tmp_path):
    source = _obs("obs_a", start=1.0, latex="x=1")
    target = _obs("obs_b", start=2.0, latex="x=1")
    kb = _kb(source, target)

    class NoLLM:
        def _structured(self, *args, **kwargs):
            raise AssertionError("semantic audit should be disabled")

    policy, stats, unresolved = run_state_canonicalization(
        NoLLM(),
        kb=kb,
        work=tmp_path,
        llm_config={},
        pipeline_version=10,
        semantic_audit=False,
        force=False,
    )

    assert unresolved == []
    assert policy.suppress_latex_ids == ["obs_a"]
    assert stats["semantic_audit_enabled"] == 0
    assert (tmp_path / "state_canonicalization.json").exists()
