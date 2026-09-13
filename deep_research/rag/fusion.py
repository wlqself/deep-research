"""Rank-fusion helpers for production retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def reciprocal_rank_fusion_scores(
    rankings: Iterable[Sequence[str]],
    *,
    rrf_k: int = 60,
) -> tuple[dict[str, float], dict[str, int]]:
    """Return RRF scores and deterministic first-seen tie-break positions."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)
            first_seen.setdefault(chunk_id, order)
            order += 1
    return scores, first_seen


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[str]:
    """Fuse ranked ID lists without comparing incompatible raw scores."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scores, first_seen = reciprocal_rank_fusion_scores(rankings, rrf_k=rrf_k)
    return [
        chunk_id
        for chunk_id, _score in sorted(
            scores.items(),
            key=lambda item: (-item[1], first_seen[item[0]], item[0]),
        )[:top_k]
    ]
