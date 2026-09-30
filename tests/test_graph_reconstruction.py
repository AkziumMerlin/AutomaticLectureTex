from automatic_lecture_tex.graph_reconstruction import (
    CandidateHypothesis,
    CandidateSet,
    GraphEdge,
    PairwisePotential,
    beam_map_inference,
    build_sparse_edges,
)
from automatic_lecture_tex.schemas import ObservationKind


def _candidate(
    candidate_id: str,
    observation_id: str,
    *,
    score: float,
    text: str,
    latex: str | None = None,
    source: str = "model",
) -> CandidateHypothesis:
    return CandidateHypothesis(
        id=candidate_id,
        observation_id=observation_id,
        kind=ObservationKind.CLAIM,
        text=text,
        latex=latex,
        unary_score=score,
        source=source,
    )


def test_global_pairwise_factor_can_override_local_best() -> None:
    left = CandidateSet(
        observation_id="o1",
        start=0.0,
        end=1.0,
        candidates=[
            _candidate("o1:local", "o1", score=1.0, text="A"),
            _candidate("o1:global", "o1", score=0.6, text="B"),
        ],
    )
    right = CandidateSet(
        observation_id="o2",
        start=2.0,
        end=3.0,
        candidates=[
            _candidate("o2:local", "o2", score=1.0, text="C"),
            _candidate("o2:global", "o2", score=0.6, text="D"),
        ],
    )
    edge = GraphEdge(
        left_observation_id="o1",
        right_observation_id="o2",
        reasons=["temporal"],
        potentials=[
            PairwisePotential(
                left_candidate_id="o1:local",
                right_candidate_id="o2:local",
                score=-2.0,
                relation="contradicts",
            ),
            PairwisePotential(
                left_candidate_id="o1:local",
                right_candidate_id="o2:global",
                score=-1.0,
                relation="contradicts",
            ),
            PairwisePotential(
                left_candidate_id="o1:global",
                right_candidate_id="o2:local",
                score=-1.0,
                relation="contradicts",
            ),
            PairwisePotential(
                left_candidate_id="o1:global",
                right_candidate_id="o2:global",
                score=2.5,
                relation="continuation",
            ),
        ],
    )

    hypotheses = beam_map_inference(
        [left, right],
        [edge],
        beam_width=8,
        top_k=4,
        pairwise_weight=1.0,
    )

    assert hypotheses[0].assignments == {
        "o1": "o1:global",
        "o2": "o2:global",
    }
    assert hypotheses[0].score > hypotheses[1].score


def test_beam_inference_returns_competing_global_hypotheses() -> None:
    sets = [
        CandidateSet(
            observation_id=f"o{index}",
            start=float(index),
            end=float(index) + 0.5,
            candidates=[
                _candidate(
                    f"o{index}:a",
                    f"o{index}",
                    score=0.5,
                    text=f"A{index}",
                ),
                _candidate(
                    f"o{index}:b",
                    f"o{index}",
                    score=0.4,
                    text=f"B{index}",
                ),
            ],
        )
        for index in range(3)
    ]

    hypotheses = beam_map_inference(
        sets,
        [],
        beam_width=8,
        top_k=4,
        pairwise_weight=1.0,
    )

    assert len(hypotheses) == 4
    assert hypotheses[0].score >= hypotheses[1].score
    assert hypotheses[0].assignments != hypotheses[1].assignments


def test_sparse_edges_include_shared_symbol_nonlocal_link() -> None:
    sets = [
        CandidateSet(
            observation_id="o1",
            start=0.0,
            end=1.0,
            candidates=[
                _candidate(
                    "o1:a",
                    "o1",
                    score=0.5,
                    text="",
                    latex=r"f(x)=\\varphi(x)",
                    source="source",
                )
            ],
        ),
        CandidateSet(
            observation_id="o2",
            start=20.0,
            end=21.0,
            candidates=[
                _candidate("o2:a", "o2", score=0.5, text="unrelated")
            ],
        ),
        CandidateSet(
            observation_id="o3",
            start=40.0,
            end=41.0,
            candidates=[
                _candidate(
                    "o3:a",
                    "o3",
                    score=0.5,
                    text="",
                    latex=r"\\varphi(y)=0",
                    source="source",
                )
            ],
        ),
    ]

    edges = build_sparse_edges(
        sets,
        neighbor_span=1,
        max_gap_seconds=30.0,
        symbol_gap_seconds=60.0,
    )

    edge_by_pair = {
        (edge.left_observation_id, edge.right_observation_id): edge
        for edge in edges
    }
    assert ("o1", "o3") in edge_by_pair
    assert "shared_symbol" in edge_by_pair[("o1", "o3")].reasons
