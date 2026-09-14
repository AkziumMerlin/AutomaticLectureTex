import json
from types import SimpleNamespace

from automatic_lecture_tex import episode_synthesis_resilient as resilient
from automatic_lecture_tex.schemas import ChunkNotes, MathAudit, NoteBlock, SemanticEpisode


def _episode() -> SemanticEpisode:
    return SemanticEpisode(
        id="episode_0001",
        title="Episode",
        start=0.0,
        end=20.0,
        observation_ids=["obs_0", "obs_1"],
    )


def _evidence(count: int) -> dict:
    observations = [
        {
            "id": f"obs_{index}",
            "start": float(index * 10),
            "end": float(index * 10 + 5),
            "kind": "claim",
            "text": f"fact {index}",
            "latex": None,
            "target_observation_id": None,
            "confidence": 1.0,
            "source_status": "observed",
        }
        for index in range(count)
    ]
    return {
        "episode": {
            "id": "episode_0001",
            "title": "Episode",
            "kind": "topic",
            "start": 0,
            "end": 20,
        },
        "observations": observations,
        "claims": [
            {
                "id": f"claim_{index}",
                "kind": "claim",
                "content": f"fact {index}",
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


def _orchestrator(*, validate: bool = False, retries: int | None = None):
    kwargs = {
        "config": SimpleNamespace(
            global_validation=validate,
            global_validation_apply_threshold=0.85,
        )
    }
    if retries is not None:
        kwargs["llm"] = SimpleNamespace(config=SimpleNamespace(max_retries=retries))
    return SimpleNamespace(**kwargs)


def test_episode_write_failure_recursively_splits_and_merges(monkeypatch) -> None:
    calls = []

    def fake_write(orchestrator, episode, evidence, previous_context):
        calls.append([item["id"] for item in evidence["observations"]])
        if len(evidence["observations"]) > 1:
            raise json.JSONDecodeError("Unterminated string", '"cut', 0)
        observation = evidence["observations"][0]
        return ChunkNotes(
            chunk_id="leaf",
            section_title="Episode",
            blocks=[
                NoteBlock(
                    type="paragraph",
                    latex=observation["text"],
                    source_evidence_ids=[observation["id"]],
                )
            ],
        )

    monkeypatch.setattr(resilient, "_write_once", fake_write)

    notes = resilient.write_episode_batch(_orchestrator(), _episode(), _evidence(2), [])

    assert calls == [["obs_0", "obs_1"], ["obs_0"], ["obs_1"]]
    assert [block.latex for block in notes.blocks] == ["fact 0", "fact 1"]


def test_multi_atom_call_disables_internal_retries_and_restores_them(monkeypatch) -> None:
    orchestrator = _orchestrator(retries=2)
    seen_retries = []

    def fake_write(orchestrator, episode, evidence, previous_context):
        seen_retries.append(orchestrator.llm.config.max_retries)
        blocks = [
            NoteBlock(
                type="paragraph",
                latex=item["text"],
                source_evidence_ids=[item["id"]],
            )
            for item in evidence["observations"]
        ]
        return ChunkNotes(chunk_id="leaf", section_title="Episode", blocks=blocks)

    monkeypatch.setattr(resilient, "_write_once", fake_write)

    resilient.write_episode_batch(orchestrator, _episode(), _evidence(2), [])

    assert seen_retries == [0]
    assert orchestrator.llm.config.max_retries == 2


def test_validation_failure_resynthesizes_smaller_children(monkeypatch) -> None:
    write_calls = []
    validation_calls = []

    def fake_write(orchestrator, episode, evidence, previous_context):
        write_calls.append(len(evidence["observations"]))
        blocks = [
            NoteBlock(
                type="paragraph",
                latex=item["text"],
                source_evidence_ids=[item["id"]],
            )
            for item in evidence["observations"]
        ]
        return ChunkNotes(chunk_id="leaf", section_title="Episode", blocks=blocks)

    def fake_validate(orchestrator, evidence, notes):
        validation_calls.append(len(evidence["observations"]))
        if len(evidence["observations"]) > 1:
            raise json.JSONDecodeError("Unterminated string", '"cut', 0)
        return MathAudit()

    monkeypatch.setattr(resilient, "_write_once", fake_write)
    monkeypatch.setattr(resilient, "_validate_once", fake_validate)

    notes = resilient.write_episode_batch(
        _orchestrator(validate=True),
        _episode(),
        _evidence(2),
        [],
    )

    assert write_calls == [2, 1, 1]
    assert validation_calls == [2, 1, 1]
    assert [block.latex for block in notes.blocks] == ["fact 0", "fact 1"]


def test_indivisible_synthesis_failure_is_recorded_not_fatal(monkeypatch) -> None:
    def fake_write(orchestrator, episode, evidence, previous_context):
        raise json.JSONDecodeError("Unterminated string", '"cut', 0)

    monkeypatch.setattr(resilient, "_write_once", fake_write)

    notes = resilient.write_episode_batch(_orchestrator(), _episode(), _evidence(1), [])

    assert notes.blocks == []
    assert "indivisible evidence" in notes.unresolved[0]


def test_proactive_split_caps_observation_count(monkeypatch) -> None:
    payload = _evidence(14)
    monkeypatch.setattr(resilient, "_base_evidence_batches", lambda *args, **kwargs: [payload])

    batches = resilient.episode_evidence_batches(None, None, None)

    assert len(batches) >= 3
    assert max(len(batch["observations"]) for batch in batches) <= 6
    assert [batch["batch"]["index"] for batch in batches] == list(range(len(batches)))
