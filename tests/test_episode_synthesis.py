import json

from automatic_lecture_tex.config import NotesConfig
from automatic_lecture_tex.episode_synthesis import (
    assemble_outline_sections,
    episode_evidence_batches,
    plan_episode_hierarchy_bounded,
)
from automatic_lecture_tex.schemas import (
    ChunkNotes,
    EpisodeHierarchyPlan,
    EpisodeKind,
    EpisodeStatus,
    KnowledgeClaim,
    LectureKnowledgeBase,
    LectureObservation,
    NoteBlock,
    ObservationKind,
    OutlineSection,
    SemanticEpisode,
    Transcript,
    TranscriptSegment,
)


def _kb_with_episodes(count: int, *, text_size: int = 20) -> LectureKnowledgeBase:
    kb = LectureKnowledgeBase(lecture_id="lecture", title="Lecture")
    for index in range(count):
        observation_id = f"obs_{index:03d}"
        claim_id = f"claim_{index:03d}"
        episode_id = f"episode_{index:04d}"
        observation = LectureObservation(
            id=observation_id,
            start=float(index * 10),
            end=float(index * 10 + 5),
            kind=ObservationKind.CLAIM,
            text="x" * text_size,
            confidence=1.0,
            episode_id=episode_id,
        )
        claim = KnowledgeClaim(
            id=claim_id,
            kind=ObservationKind.CLAIM,
            content="x" * text_size,
            episode_id=episode_id,
            scope=episode_id,
            evidence_ids=[observation_id],
            introduced_at=observation.start,
        )
        episode = SemanticEpisode(
            id=episode_id,
            title=f"Episode {index}",
            kind=EpisodeKind.TOPIC,
            start=observation.start,
            end=observation.end,
            status=EpisodeStatus.CLOSED,
            observation_ids=[observation_id],
            claim_ids=[claim_id],
        )
        kb.observations.append(observation)
        kb.claims.append(claim)
        kb.episodes.append(episode)
    return kb


def test_episode_evidence_batches_are_bounded():
    kb = _kb_with_episodes(1)
    episode = kb.episodes[0]
    # Make one episode contain enough evidence to require several synthesis calls.
    for index in range(1, 8):
        observation_id = f"obs_extra_{index:03d}"
        claim_id = f"claim_extra_{index:03d}"
        observation = LectureObservation(
            id=observation_id,
            start=float(index * 10),
            end=float(index * 10 + 5),
            kind=ObservationKind.PROOF_STEP,
            text="y" * 900,
            confidence=1.0,
            episode_id=episode.id,
        )
        claim = KnowledgeClaim(
            id=claim_id,
            kind=ObservationKind.PROOF_STEP,
            content="y" * 900,
            episode_id=episode.id,
            scope=episode.id,
            evidence_ids=[observation_id],
            introduced_at=observation.start,
        )
        kb.observations.append(observation)
        kb.claims.append(claim)
        episode.observation_ids.append(observation_id)
        episode.claim_ids.append(claim_id)
        episode.end = observation.end

    config = NotesConfig(episode_synthesis_max_evidence_chars=4500)
    transcript = Transcript(
        lecture_id="lecture",
        segments=[
            TranscriptSegment(
                id=f"s{index}",
                start=float(index * 10),
                end=float(index * 10 + 6),
                text="raw " + "z" * 120,
            )
            for index in range(8)
        ],
    )
    batches = episode_evidence_batches(kb, episode, config, transcript)

    assert len(batches) > 1
    for payload in batches:
        assert len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) <= 4500


class _HierarchyOrchestrator:
    def __init__(self):
        self.config = NotesConfig(hierarchy_batch_episodes=2)
        self.output_language = "ru"
        self.prompts = []

    def _structured(self, prompt, schema, *, operation, max_tokens=None):
        self.prompts.append((operation, prompt, max_tokens))
        return EpisodeHierarchyPlan()


def test_hierarchy_planning_is_batched():
    kb = _kb_with_episodes(5)
    orchestrator = _HierarchyOrchestrator()

    plan = plan_episode_hierarchy_bounded(orchestrator, kb)

    assert plan.boundaries == []
    assert len(orchestrator.prompts) == 3
    assert all(operation == "episode_hierarchy" for operation, _, _ in orchestrator.prompts)


def test_outline_sections_are_assembled_without_llm_rewrite():
    section = OutlineSection(
        id="topic_000",
        title="Topic",
        start=0,
        end=20,
        episode_ids=["episode_0000", "episode_0001"],
    )
    episode_notes = {
        "episode_0000": ChunkNotes(
            chunk_id="episode_0000",
            section_title="E0",
            blocks=[NoteBlock(type="paragraph", latex="first", source_evidence_ids=["o0"])],
        ),
        "episode_0001": ChunkNotes(
            chunk_id="episode_0001",
            section_title="E1",
            blocks=[NoteBlock(type="paragraph", latex="second", source_evidence_ids=["o1"])],
        ),
    }

    sections = assemble_outline_sections([section], episode_notes)

    assert [block.latex for block in sections[0].blocks] == ["first", "second"]
    assert sections[0].section_title == "Topic"
