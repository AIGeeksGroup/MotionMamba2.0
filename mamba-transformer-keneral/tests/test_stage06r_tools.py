from __future__ import annotations

import sys
from pathlib import Path

import torch


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from evaluate_ardy_history_oracle import (  # noqa: E402
    _aggregate,
    _resolve_diffusion_indices,
)
from evaluate_ardy_learned_routing import _subset_result  # noqa: E402
from train_ardy_history_router import (  # noqa: E402
    HistoryRouter,
    _recent_metrics,
)


def _metric(loss: float, chunks=(), distances=()):
    return {
        "loss": loss,
        "root_loss": loss / 2,
        "body_loss": loss / 2,
        "chunks": list(chunks),
        "distances": list(distances),
    }


def test_oracle_helpers_resolve_aliases_and_aggregate_paired_gains():
    assert _resolve_diffusion_indices("last,first,middle,5", 10) == [0, 5, 9]
    cases = [
        {
            "no_history": _metric(3),
            "recent": _metric(2),
            "oldest": _metric(2),
            "full": _metric(1.5),
            "oracle_top1": _metric(1, (0,), (2,)),
            "oracle_top2": _metric(1.2, (0, 1), (2, 1)),
        },
        {
            "no_history": _metric(4),
            "recent": _metric(3),
            "oldest": _metric(3),
            "full": _metric(2.5),
            "oracle_top1": _metric(2, (1,), (1,)),
            "oracle_top2": None,
        },
    ]
    result = _aggregate(cases)
    comparison = result["paired_comparisons"]["oracle_top1_vs_full"]
    assert comparison["fraction_strictly_better"] == 1.0
    assert result["oracle_top1_recent_fraction"] == 0.5
    assert result["oracle_top1_distance_counts"] == {"1": 1, "2": 1}


def test_subset_lookup_preserves_chunk_boundaries():
    case = {
        "available_chunks": [0, 1, 2],
        "no_history": _metric(3),
        "full": _metric(1, (0, 1, 2)),
        "singleton_candidates": [
            _metric(2.0, (0,)),
            _metric(2.1, (1,)),
            _metric(2.2, (2,)),
        ],
        "pair_candidates": [
            _metric(1.2, (0, 1)),
            _metric(1.3, (0, 2)),
            _metric(1.4, (1, 2)),
        ],
    }
    assert _subset_result(case, (2, 0))["loss"] == 1.3
    assert _subset_result(case, (0, 1, 2))["loss"] == 1
    assert _subset_result(case, ())["loss"] == 3


def test_router_masks_padding_and_recent_ranking_handles_distance_three():
    router = HistoryRouter(8, 12, 4, 0.1)
    query = torch.randn(3, 8)
    candidates = torch.randn(3, 3, 12)
    valid = torch.tensor([[True, True, False], [True, True, True], [True, True, True]])
    scores = router(query, candidates, valid)
    assert scores.shape == (3, 3)
    assert torch.isneginf(scores[0, 2])

    examples = [
        {"candidate_chunks": torch.tensor([0, 1]), "label": 1},
        {"candidate_chunks": torch.tensor([0, 1, 2]), "label": 0},
    ]
    metrics = _recent_metrics(examples)
    assert metrics["hit_at_1"] == 0.5
    assert metrics["hit_at_2"] == 0.5
