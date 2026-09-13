import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RetrievalCase:
    query_id: str
    ranked_chunk_ids: Sequence[str]
    relevance: Mapping[str, int]


def _relevant_ids(relevance: Mapping[str, int]) -> set[str]:
    return {
        chunk_id
        for chunk_id, grade in relevance.items()
        if grade > 0
    }


def hit_rate_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = _relevant_ids(relevance)
    return float(
        any(chunk_id in relevant_ids for chunk_id in ranked_chunk_ids[:k])
    )


def recall_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = _relevant_ids(relevance)
    if not relevant_ids:
        return 0.0
    retrieved_relevant = sum(
        chunk_id in relevant_ids
        for chunk_id in ranked_chunk_ids[:k]
    )
    return retrieved_relevant / len(relevant_ids)


def precision_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = _relevant_ids(relevance)
    retrieved_relevant = sum(
        chunk_id in relevant_ids
        for chunk_id in ranked_chunk_ids[:k]
    )
    return retrieved_relevant / k


def reciprocal_rank_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = _relevant_ids(relevance)
    for rank, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1 / rank
    return 0.0


def average_precision_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = _relevant_ids(relevance)
    if not relevant_ids:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        if chunk_id not in relevant_ids:
            continue
        hit_count += 1
        precision_sum += hit_count / rank

    return precision_sum / len(relevant_ids)


def _dcg_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    return sum(
        (2 ** relevance.get(chunk_id, 0) - 1) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_chunk_ids[:k], start=1)
        if relevance.get(chunk_id, 0) > 0
    )


def ndcg_at_k(
    ranked_chunk_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    ideal_ranking = sorted(
        relevance,
        key=lambda chunk_id: relevance[chunk_id],
        reverse=True,
    )
    ideal_dcg = _dcg_at_k(ideal_ranking, relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return _dcg_at_k(ranked_chunk_ids, relevance, k) / ideal_dcg


def evaluate_retrieval(
    cases: Iterable[RetrievalCase],
    *,
    k_values: Sequence[int] = (1, 3, 5, 8),
) -> dict[str, object]:
    normalized_k_values = tuple(k_values)
    if not normalized_k_values or any(k <= 0 for k in normalized_k_values):
        raise ValueError("k_values must contain positive values")

    case_list = list(cases)
    answerable_cases = [
        case for case in case_list if _relevant_ids(case.relevance)
    ]
    per_query: list[dict[str, object]] = []

    for case in case_list:
        metrics: dict[str, float] = {}
        for k in normalized_k_values:
            metrics[f"hit_rate@{k}"] = hit_rate_at_k(
                case.ranked_chunk_ids,
                case.relevance,
                k,
            )
            metrics[f"recall@{k}"] = recall_at_k(
                case.ranked_chunk_ids,
                case.relevance,
                k,
            )
            metrics[f"precision@{k}"] = precision_at_k(
                case.ranked_chunk_ids,
                case.relevance,
                k,
            )
            metrics[f"mrr@{k}"] = reciprocal_rank_at_k(
                case.ranked_chunk_ids,
                case.relevance,
                k,
            )
            metrics[f"map@{k}"] = average_precision_at_k(
                case.ranked_chunk_ids,
                case.relevance,
                k,
            )
            metrics[f"ndcg@{k}"] = ndcg_at_k(
                case.ranked_chunk_ids,
                case.relevance,
                k,
            )

        per_query.append(
            {
                "query_id": case.query_id,
                "metrics": metrics,
            }
        )

    aggregate: dict[str, float] = {}
    for k in normalized_k_values:
        for metric_name in (
            "hit_rate",
            "recall",
            "precision",
            "mrr",
            "map",
            "ndcg",
        ):
            key = f"{metric_name}@{k}"
            values = [
                query["metrics"][key]
                for query in per_query
                if query["query_id"] in {
                    case.query_id for case in answerable_cases
                }
            ]
            aggregate[key] = (
                sum(values) / len(values) if values else 0.0
            )

    return {
        "query_count": len(case_list),
        "answerable_query_count": len(answerable_cases),
        "unanswerable_query_count": (
            len(case_list) - len(answerable_cases)
        ),
        "k_values": list(normalized_k_values),
        "aggregate": aggregate,
        "per_query": per_query,
    }
