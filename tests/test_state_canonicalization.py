from automatic_lecture_tex.schemas import (
    EpisodeKind,
    LectureKnowledgeBase,
    LectureObservation,
    ObservationKind,
    SemanticEpisode,
)
from automatic_lecture_tex.state_canonicalization import (
    CanonicalObservationRelation,
    _apply_relations,
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


def test_exact_duplicate_suppression_keeps_target_payload_byte_for_byte():
    source = _obs(
        "obs_a",
        start=1.0,
        text="Норма функционала.",
        latex=r"\|f\|=\sup_{\|x\|\le1}|f(x)|",
    )
    target = _obs(
        "obs_b",
        start=2.0,
        text="Норма функционала.",
        latex=r"\|f\|=\sup_{\|x\|\le1}|f(x)|",
    )
    kb = _kb(source, target)

    result, stats, relations = _apply_relations(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="duplicate",
            )
        ],
    )

    assert [item.id for item in result.observations] == ["obs_b"]
    assert result.observations[0].text == target.text
    assert result.observations[0].latex == target.latex
    assert stats["suppressed_duplicate"] == 1
    assert relations[0]["host_verified"] is True


def test_partial_formula_can_only_be_suppressed_by_literal_later_completion():
    source = _obs("obs_a", start=1.0, latex=r"V(x)=\{y\in X:")
    target = _obs(
        "obs_b",
        start=2.0,
        latex=r"V(x)=\{y\in X:|\varphi(y)-\varphi(x)|<\varepsilon\}",
    )
    kb = _kb(source, target)

    result, stats, _ = _apply_relations(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="intermediate",
            )
        ],
    )

    assert [item.id for item in result.observations] == ["obs_b"]
    assert result.observations[0].latex == target.latex
    assert stats["suppressed_intermediate"] == 1


def test_semantic_supersession_is_audit_only_without_explicit_correction():
    source = _obs("obs_a", start=1.0, latex="x=1")
    target = _obs("obs_b", start=50.0, latex="x=2")
    kb = _kb(source, target)

    result, stats, relations = _apply_relations(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="supersedes",
            )
        ],
    )

    assert [item.id for item in result.observations] == ["obs_a", "obs_b"]
    assert stats["flagged_supersedes"] == 1
    assert relations[0]["applied_as"] == "audit_only"
    assert relations[0]["host_verified"] is False


def test_explicit_correction_may_suppress_its_target_without_rewriting_replacement():
    source = _obs("obs_a", start=1.0, latex="x=1")
    correction = _obs(
        "obs_b",
        start=2.0,
        kind=ObservationKind.CORRECTION,
        latex="x=2",
        target_observation_id="obs_a",
    )
    kb = _kb(source, correction)

    result, stats, _ = _apply_relations(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                target_observation_id="obs_b",
                relation="supersedes",
            )
        ],
    )

    assert [item.id for item in result.observations] == ["obs_b"]
    assert result.observations[0].latex == "x=2"
    assert stats["suppressed_explicit_superseded"] == 1


def test_meta_suppression_is_limited_to_explicit_transition_events():
    mathematical_remark = _obs(
        "obs_a",
        start=1.0,
        kind=ObservationKind.REMARK,
        text="Слабая сходимость не влечёт сходимость по норме.",
    )
    transition = _obs(
        "obs_b",
        start=2.0,
        kind=ObservationKind.TRANSITION,
        text="Перейдём к следующему вопросу.",
    )
    kb = _kb(mathematical_remark, transition)

    result, stats, _ = _apply_relations(
        kb,
        [
            CanonicalObservationRelation(
                source_observation_id="obs_a",
                relation="meta",
            ),
            CanonicalObservationRelation(
                source_observation_id="obs_b",
                relation="meta",
            ),
        ],
    )

    assert [item.id for item in result.observations] == ["obs_a"]
    assert stats["suppressed_meta"] == 1
    assert stats["rejected_unsafe_relation"] == 1
