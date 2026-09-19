from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from automatic_lecture_tex.config import NotesConfig
from automatic_lecture_tex.episode_synthesis_resilient import (
    episode_evidence_batches,
    merge_episode_batches,
    reset_synthesis_stats,
    synthesis_stats_snapshot,
    write_episode_batch,
)
from automatic_lecture_tex.knowledge_integrity import (
    GeneratedLectureObservation,
    IntegrityKnowledgeOrchestrator,
)
from automatic_lecture_tex.latex import render_block, render_lecture
from automatic_lecture_tex.schemas import (
    ChunkNotes,
    KnowledgeClaim,
    LectureChunk,
    LectureIR,
    LectureKnowledgeBase,
    LectureObservation,
    NoteBlock,
    ObservationKind,
    SemanticEpisode,
    Transcript,
    TranscriptSegment,
)


class _ExtractionLLM:
    def _structured(self, prompt, schema, *, operation, max_tokens=None):
        assert operation == "knowledge_extract"
        assert "seg_002" in prompt
        return schema.model_validate(
            {
                "observations": [
                    {
                        "kind": "definition",
                        "text": "Определяется комплексный линейный функционал.",
                        "latex": r"f:X\to\mathbb{C}",
                        "confidence": 0.95,
                        "source_status": "observed",
                        "source_segment_ids": ["seg_002"],
                    }
                ],
                "unresolved": [],
            }
        )


def test_generated_observation_rejects_empty_text_and_requires_provenance():
    with pytest.raises(ValidationError):
        GeneratedLectureObservation(
            kind="claim",
            text="",
            confidence=1.0,
            source_status="observed",
            source_segment_ids=["seg_001"],
        )
    with pytest.raises(ValidationError):
        GeneratedLectureObservation(
            kind="claim",
            text="claim",
            confidence=1.0,
            source_status="observed",
            source_segment_ids=[],
        )


def test_observation_times_are_derived_from_asr_segment_ids():
    transcript = Transcript(
        lecture_id="lecture",
        segments=[
            TranscriptSegment(id="seg_001", start=220.0, end=230.0, text="before"),
            TranscriptSegment(id="seg_002", start=236.46, end=247.04, text="definition"),
        ],
    )
    chunk = LectureChunk(
        id="window_0000",
        start=0.0,
        end=478.68,
        segment_ids=["seg_001", "seg_002"],
        text="before\ndefinition",
    )
    orchestrator = IntegrityKnowledgeOrchestrator(
        _ExtractionLLM(),
        NotesConfig(),
        "ru",
        transcript=transcript,
    )

    result = orchestrator.extract_observations(
        chunk,
        [],
        LectureKnowledgeBase(lecture_id="lecture", title="Lecture"),
    )

    assert len(result.observations) == 1
    assert result.observations[0].start == pytest.approx(236.46)
    assert result.observations[0].end == pytest.approx(247.04)
    assert result.observations[0].evidence_refs == ["seg_002"]


def _episode_evidence(count: int) -> dict:
    return {
        "episode": {
            "id": "episode_0001",
            "title": "Episode",
            "kind": "topic",
            "start": 0,
            "end": 20,
        },
        "observations": [
            {
                "id": f"obs_{index}",
                "start": float(index * 10),
                "end": float(index * 10 + 5),
                "kind": "proof_step",
                "text": f"step {index}",
                "latex": None,
                "target_observation_id": None,
                "confidence": 1.0,
                "source_status": "observed",
            }
            for index in range(count)
        ],
        "claims": [
            {
                "id": f"claim_{index}",
                "kind": "proof_step",
                "content": f"step {index}",
                "latex": None,
                "math_status": "unchecked",
                "source_status": "observed",
                "evidence_ids": [f"obs_{index}"],
                "supersedes": [],
                "introduced_at": float(index * 10),
            }
            for index in range(count)
        ],
        "symbols": [],
        "transcript": [],
        "batch": {"index": 0, "count": 1},
    }


def test_missing_substantive_coverage_triggers_recursive_split(monkeypatch):
    from automatic_lecture_tex import episode_synthesis_resilient as resilient

    reset_synthesis_stats()
    calls = []

    def fake_write(orchestrator, episode, evidence, previous_context):
        ids = [item["id"] for item in evidence["observations"]]
        calls.append(ids)
        selected = evidence["observations"] if len(ids) == 1 else evidence["observations"][:1]
        return ChunkNotes(
            section_title="Episode",
            blocks=[
                NoteBlock(
                    type="proof",
                    latex=item["text"],
                    source_evidence_ids=[item["id"]],
                )
                for item in selected
            ],
        )

    monkeypatch.setattr(resilient, "_write_once", fake_write)
    episode = SemanticEpisode(id="episode_0001", title="Episode", start=0, end=20)
    orchestrator = SimpleNamespace(
        config=SimpleNamespace(global_validation=False, global_validation_apply_threshold=0.85)
    )

    notes = write_episode_batch(orchestrator, episode, _episode_evidence(2), [])

    assert calls == [["obs_0", "obs_1"], ["obs_0"], ["obs_1"]]
    assert {item for block in notes.blocks for item in block.source_evidence_ids} == {
        "obs_0",
        "obs_1",
    }
    assert synthesis_stats_snapshot()["coverage_splits"] == 1


def test_episode_batches_include_bounded_local_transcript_context():
    kb = LectureKnowledgeBase(lecture_id="lecture", title="Lecture")
    observation = LectureObservation(
        id="obs_0",
        window_id="window_0",
        start=10.0,
        end=12.0,
        kind=ObservationKind.CLAIM,
        text="claim",
        confidence=1.0,
        evidence_refs=["seg_0"],
    )
    claim = KnowledgeClaim(
        id="claim_0",
        content="claim",
        evidence_ids=["obs_0"],
        introduced_at=10.0,
    )
    episode = SemanticEpisode(
        id="episode_0",
        title="Episode",
        start=10.0,
        end=12.0,
        observation_ids=["obs_0"],
        claim_ids=["claim_0"],
    )
    kb.observations.append(observation)
    kb.claims.append(claim)
    kb.episodes.append(episode)
    transcript = Transcript(
        lecture_id="lecture",
        segments=[
            TranscriptSegment(id="seg_before", start=0, end=1, text="near context"),
            TranscriptSegment(id="seg_0", start=9, end=13, text="local source"),
            TranscriptSegment(id="seg_after", start=100, end=101, text="far away"),
        ],
    )

    batches = episode_evidence_batches(kb, episode, NotesConfig(), transcript=transcript)
    ids = [item["id"] for item in batches[0]["transcript"]]

    assert "seg_0" in ids
    assert "seg_after" not in ids


def test_proof_fragments_are_stitched_and_exact_duplicates_are_removed():
    episode = SemanticEpisode(id="episode_0", title="Proof", start=0, end=10)
    batches = [
        ChunkNotes(
            section_title="Proof",
            blocks=[
                NoteBlock(type="proof", latex="Шаг один.", source_evidence_ids=["obs_1"]),
            ],
        ),
        ChunkNotes(
            section_title="Proof",
            blocks=[
                NoteBlock(
                    type="proof",
                    title="Продолжение доказательства",
                    latex="Шаг два.",
                    source_evidence_ids=["obs_2"],
                ),
                NoteBlock(type="remark", latex="Итог.", source_evidence_ids=["obs_3"]),
                NoteBlock(type="remark", latex="Итог.", source_evidence_ids=["obs_3"]),
            ],
        ),
    ]

    notes = merge_episode_batches(episode, batches)

    assert [block.type for block in notes.blocks] == ["proof", "remark"]
    assert "Шаг один." in notes.blocks[0].latex
    assert "Шаг два." in notes.blocks[0].latex
    assert notes.blocks[0].source_evidence_ids == ["obs_1", "obs_2"]


def test_renderer_preserves_math_in_titles_and_does_not_double_wrap_prose_equations():
    bad_old_equation = NoteBlock(
        type="equation",
        latex="Записано соотношение $x ∈ X$.",
        source_evidence_ids=["obs_1"],
    )
    rendered_block = render_block(bad_old_equation)
    assert not rendered_block.startswith("\\[")
    assert r"$x \in  X$" in rendered_block

    ir = LectureIR(
        lecture_id="l1",
        title="Лекция 1",
        chunks=[
            ChunkNotes(
                section_title=r"Слабая топология $\tau_{weak}$",
                blocks=[
                    NoteBlock(
                        type="remark",
                        title=r"Критерий $\tau_{weak}$",
                        latex="Текст с управляющим символом \x7f и формулой $x ∈ X$.",
                    )
                ],
            )
        ],
    )

    tex = render_lecture(ir)
    assert r"\section{Слабая топология $\tau_{weak}$}" in tex
    assert r"[Критерий $\tau_{weak}$]" in tex
    assert "\x7f" not in tex
    assert r"$x \in  X$" in tex



def test_context_overflow_parser_uses_backend_reported_budget():
    from automatic_lecture_tex.llm_robust import _context_overflow_output_ceiling

    error = RuntimeError(
        "This model's maximum context length is 20000 tokens. However, you requested 16384 "
        "output tokens and your prompt contains at least 3617 input tokens, for a total of at "
        "least 20001 tokens. (parameter=input_tokens, value=3617)"
    )

    assert _context_overflow_output_ceiling(error) == 15871


def test_structured_retry_recovers_from_vllm_context_overflow(monkeypatch):
    from automatic_lecture_tex import llm_robust

    class Payload(BaseModel):
        value: int

    class FakeBadRequest(Exception):
        pass

    class FakeCompletions:
        def __init__(self):
            self.max_tokens_seen = []

        def create(self, **kwargs):
            budget = kwargs["max_tokens"]
            self.max_tokens_seen.append(budget)
            if budget == 16384:
                raise FakeBadRequest(
                    "This model's maximum context length is 20000 tokens. However, you requested "
                    "16384 output tokens and your prompt contains at least 3617 input tokens, "
                    "for a total of at least 20001 tokens. "
                    "(parameter=input_tokens, value=3617)"
                )
            if budget in {4096, 8192}:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"value":'),
                            finish_reason="length",
                        )
                    ],
                    usage=SimpleNamespace(completion_tokens=budget),
                )
            assert budget == 15871
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"value":7}'),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=8),
            )

    fake_completions = FakeCompletions()
    client = object.__new__(llm_robust.LectureModelClient)
    client.config = SimpleNamespace(
        model="test",
        temperature=0.0,
        max_tokens=4096,
        max_retries=2,
    )
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_completions)
    )
    client._extra_body = lambda: {}
    client._response_format = lambda schema: {"type": "json_object"}
    client._record_usage = lambda operation, response: None

    monkeypatch.setattr(llm_robust, "BadRequestError", FakeBadRequest)

    result = client._structured("prompt", Payload, operation="math_audit")

    assert result.value == 7
    assert fake_completions.max_tokens_seen == [4096, 8192, 16384, 15871]
