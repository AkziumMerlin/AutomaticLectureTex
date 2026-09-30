from automatic_lecture_tex.claim_compaction import compact_repaired_claims
from automatic_lecture_tex.schemas import (
    ClaimCompactionBatch,
    ClaimStatus,
    EpisodeKind,
    EpisodeStatus,
    KnowledgeClaim,
    LectureKnowledgeBase,
    SemanticEpisode,
    SourceStatus,
)


def _kb() -> LectureKnowledgeBase:
    claims = [
        KnowledgeClaim(
            id="claim_1",
            content="Пусть f — комплексно-линейный функционал.",
            episode_id="episode_1",
            scope="episode_1",
            evidence_ids=["obs_1"],
            introduced_at=1.0,
            source_status=SourceStatus.RECONSTRUCTED,
        ),
        KnowledgeClaim(
            id="claim_2",
            content="Комплексная линейность означает линейность по комплексным скалярам.",
            latex=r"f(\alpha x)=\alpha f(x)",
            episode_id="episode_1",
            scope="episode_1",
            evidence_ids=["obs_2"],
            introduced_at=2.0,
            source_status=SourceStatus.OBSERVED,
        ),
        KnowledgeClaim(
            id="claim_3",
            content="То же условие комплексной линейности повторяется ещё раз.",
            latex=r"f(\alpha x)=\alpha f(x)",
            episode_id="episode_1",
            scope="episode_1",
            evidence_ids=["obs_3"],
            introduced_at=3.0,
            source_status=SourceStatus.RECONSTRUCTED,
        ),
    ]
    episode = SemanticEpisode(
        id="episode_1",
        title="Комплексная линейность",
        kind=EpisodeKind.PROOF,
        start=1.0,
        end=4.0,
        status=EpisodeStatus.CLOSED,
        observation_ids=["obs_1", "obs_2", "obs_3"],
        claim_ids=[item.id for item in claims],
    )
    return LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        claims=claims,
        episodes=[episode],
    )


class FakeOrchestrator:
    output_language = "ru"

    def __init__(self):
        self.calls = 0

    def _structured(self, prompt, schema, **kwargs):
        self.calls += 1
        assert schema is ClaimCompactionBatch
        assert kwargs["operation"] == "repaired_claim_compaction"
        assert "final NoteBlocks" in prompt
        return ClaimCompactionBatch.model_validate(
            {
                "episodes": [
                    {
                        "episode_id": "episode_1",
                        "items": [
                            {
                                "type": "prose",
                                "kind": "claim",
                                "content": (
                                    "Для комплексно-линейного функционала линейность "
                                    "понимается по комплексным скалярам."
                                ),
                                "source_claim_ids": ["claim_1", "claim_2"],
                            },
                            {
                                "type": "formula",
                                "source_claim_id": "claim_2",
                            },
                        ],
                    }
                ]
            }
        )


def test_claim_compaction_supersedes_raw_claims_and_preserves_formula(tmp_path):
    kb = _kb()
    orchestrator = FakeOrchestrator()

    compacted, stats, unresolved = compact_repaired_claims(
        orchestrator,
        kb=kb,
        work=tmp_path,
        llm_config={"model": "stub"},
        force=False,
    )

    assert unresolved == []
    assert stats["episodes_compacted"] == 1
    assert stats["claims_before"] == 3
    assert stats["claims_after"] == 2
    assert orchestrator.calls == 1

    source = {
        item.id: item
        for item in compacted.claims
        if item.id in {"claim_1", "claim_2", "claim_3"}
    }
    assert all(item.status == ClaimStatus.SUPERSEDED for item in source.values())

    episode = compacted.episodes[0]
    assert episode.claim_ids == [
        "claim_compact_episode_1_000",
        "claim_compact_episode_1_001",
    ]
    derived = {item.id: item for item in compacted.claims if item.id in episode.claim_ids}
    assert derived["claim_compact_episode_1_000"].evidence_ids == ["obs_1", "obs_2"]
    assert derived["claim_compact_episode_1_001"].content == ""
    assert derived["claim_compact_episode_1_001"].latex == r"f(\alpha x)=\alpha f(x)"


def test_claim_compaction_cache_reuses_compacted_graph(tmp_path):
    orchestrator = FakeOrchestrator()
    kwargs = {
        "work": tmp_path,
        "llm_config": {"model": "stub"},
        "force": False,
    }

    first, first_stats, _ = compact_repaired_claims(orchestrator, kb=_kb(), **kwargs)
    second, second_stats, _ = compact_repaired_claims(orchestrator, kb=_kb(), **kwargs)

    assert orchestrator.calls == 1
    assert first_stats["model_calls"] == 1
    assert second_stats["cache_hits"] == 1
    assert second_stats["model_calls"] == 0
    assert second.episodes[0].claim_ids == first.episodes[0].claim_ids


def test_invalid_episode_plan_keeps_original_claims_active(tmp_path):
    class InvalidOrchestrator:
        output_language = "ru"

        def _structured(self, prompt, schema, **kwargs):
            del prompt, schema, kwargs
            return ClaimCompactionBatch.model_validate(
                {
                    "episodes": [
                        {
                            "episode_id": "episode_1",
                            "items": [
                                {
                                    "type": "formula",
                                    "source_claim_id": "missing_claim",
                                }
                            ],
                        }
                    ]
                }
            )

    compacted, stats, unresolved = compact_repaired_claims(
        InvalidOrchestrator(),
        kb=_kb(),
        work=tmp_path,
        llm_config={"model": "stub"},
        force=False,
    )

    assert stats["fallback_episodes"] == 1
    assert unresolved
    assert compacted.episodes[0].claim_ids == ["claim_1", "claim_2", "claim_3"]
    assert all(item.status == ClaimStatus.ACTIVE for item in compacted.claims)
