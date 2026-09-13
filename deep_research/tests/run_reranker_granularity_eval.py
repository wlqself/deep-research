"""Compare Child, micro-Parent, and Parent reranker input granularities."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from deep_research.config import settings
from deep_research.rag.lexical import SqliteBm25Index
from deep_research.evaluation.parent_child import (
    build_metadata_weighted_text,
    build_parent_child_view,
    collapse_child_rankings,
)
from deep_research.rag.query_rewrite import (
    build_rewritten_lexical_query,
)
from deep_research.evaluation.rag_runner import (
    evaluate_retriever,
    load_eval_queries,
    load_qrels,
)
from deep_research.rag.fusion import reciprocal_rank_fusion
from deep_research.rag.factory import create_embeddings, ensure_collection
from deep_research.rag.service import RagService
from deep_research.tests.run_hybrid_rrf_eval import load_corpus_rows
from deep_research.tests.run_reranker_eval import SiliconFlowReranker
from deep_research.tests.run_rewritten_parent_child_eval import (
    map_qrels_to_parents,
)


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
FULL_COMBO_PATH = SOURCE_DIR / "rewritten_parent_child_eval.json"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "reranker_granularity_eval_v1.json"
EVAL_COLLECTION = "rag_eval_reranker_granularity_v1"
GRANULARITIES = (1, 2, 4)


def _load_rewritten_terms(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_terms = payload.get("rewritten_terms", {})
    if not isinstance(raw_terms, Mapping):
        return {}

    return {
        query_id: [
            term.strip()
            for term in terms
            if isinstance(term, str) and term.strip()
        ]
        for query_id, terms in raw_terms.items()
        if isinstance(query_id, str) and isinstance(terms, list)
    }


def _build_granules(
    children: Iterable[Mapping[str, Any]],
    *,
    granularity: int,
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Build source-order context windows without changing searchable Children."""
    if granularity <= 0:
        raise ValueError("granularity must be positive")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_child in children:
        child = dict(raw_child)
        metadata = child.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Child must contain metadata")
        document_id = metadata.get("document_id")
        child_id = child.get("_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Child metadata must contain document_id")
        if not isinstance(child_id, str) or not child_id:
            raise ValueError("Child must contain a non-empty _id")
        grouped[document_id].append(child)

    granules: dict[str, dict[str, object]] = {}
    child_to_granule: dict[str, str] = {}
    for document_id, document_rows in grouped.items():
        document_rows.sort(
            key=lambda row: (
                int(row.get("metadata", {}).get("chunk_index", 0)),
                str(row["_id"]),
            )
        )
        for start in range(0, len(document_rows), granularity):
            window = document_rows[start:start + granularity]
            granule_index = start // granularity
            granule_id = (
                f"{document_id}:granule:{granularity}:{granule_index}"
            )
            child_ids = [str(row["_id"]) for row in window]
            granules[granule_id] = {
                "_id": granule_id,
                "document_id": document_id,
                "child_ids": child_ids,
                "text": "\n\n".join(str(row["text"]) for row in window),
            }
            for child_id in child_ids:
                child_to_granule[child_id] = granule_id

    return granules, child_to_granule


def _aggregate_query_group(
    result: Mapping[str, Any],
    query_ids: set[str],
) -> dict[str, object]:
    per_query = [
        row
        for row in result["per_query"]
        if row["query_id"] in query_ids
    ]
    aggregate: dict[str, float] = {}
    for key in result["aggregate"]:
        values = [row["metrics"][key] for row in per_query]
        aggregate[key] = sum(values) / len(values) if values else 0.0
    return {
        "query_count": len(per_query),
        "aggregate": aggregate,
    }


def _rank_map(parent_ids: list[str]) -> dict[str, int]:
    return {
        parent_id: rank
        for rank, parent_id in enumerate(parent_ids, start=1)
    }


def _input_stats(
    granules: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    lengths = [
        len(str(granule["text"]))
        for granule in granules.values()
    ]
    if not lengths:
        return {
            "granule_count": 0,
            "char_count": {},
        }
    return {
        "granule_count": len(lengths),
        "char_count": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
        },
    }


def run_reranker_granularity_eval(
    *,
    corpus_path: str | Path = CORPUS_PATH,
    queries_path: str | Path = QUERIES_PATH,
    qrels_path: str | Path = QRELS_PATH,
    full_combo_path: str | Path = FULL_COMBO_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    top_k: int = 8,
    candidate_k: int = 32,
    rrf_k: int = 60,
    parent_size: int = 4,
    metadata_weight: int = 2,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
) -> dict[str, object]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")

    rows = load_corpus_rows(corpus_path)
    children, _parents, child_to_parent = build_parent_child_view(
        rows,
        parent_size=parent_size,
    )
    children_by_id = {
        str(child["_id"]): child
        for child in children
    }
    queries = load_eval_queries(queries_path)
    qrels = load_qrels(qrels_path)
    parent_qrels = map_qrels_to_parents(qrels, child_to_parent)
    query_id_by_text = {query["text"]: query["_id"] for query in queries}
    if len(query_id_by_text) != len(queries):
        raise ValueError("Evaluation queries must have unique text")
    rewritten_terms = _load_rewritten_terms(full_combo_path)

    granule_views = {
        granularity: _build_granules(
            children,
            granularity=granularity,
        )
        for granularity in GRANULARITIES
    }

    bm25 = SqliteBm25Index()
    client = QdrantClient(location=":memory:")
    try:
        bm25.upsert_chunks(children)
        embeddings = create_embeddings(settings)
        ensure_collection(
            client=client,
            collection_name=EVAL_COLLECTION,
            dimensions=settings.embedding_dimensions,
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=EVAL_COLLECTION,
            embedding=embeddings,
            validate_collection_config=False,
        )
        rag_service = RagService(
            client=client,
            embeddings=embeddings,
            vector_store=vector_store,
            collection_name=EVAL_COLLECTION,
            metadata_weight=metadata_weight,
        )

        documents = []
        for child in children:
            metadata = dict(child["metadata"])
            metadata["chunk_id"] = child["_id"]
            metadata["status"] = "indexed"
            documents.append(
                Document(
                    page_content=build_metadata_weighted_text(
                        child,
                        metadata_weight=metadata_weight,
                    ),
                    metadata=metadata,
                )
            )

        print("indexing reranker granularity candidates", flush=True)
        rag_service.index_chunks(documents)
        reranker = SiliconFlowReranker(
            model=reranker_model,
            top_n=candidate_k,
        )

        candidate_cache: dict[str, dict[str, list[str]]] = {}
        ranking_cache: dict[int, dict[str, dict[str, object]]] = {
            granularity: {}
            for granularity in GRANULARITIES
        }

        def retrieve_candidates(query: str) -> dict[str, list[str]]:
            cached = candidate_cache.get(query)
            if cached is not None:
                return cached

            query_id = query_id_by_text[query]
            dense_results = rag_service.search_chunks(
                query,
                top_k=candidate_k,
            )
            dense_ids = [
                document.metadata["chunk_id"]
                for document, _score in dense_results
                if isinstance(document.metadata.get("chunk_id"), str)
            ]
            lexical_query = build_rewritten_lexical_query(
                query,
                rewritten_terms.get(query_id, []),
            )
            bm25_ids = bm25.search_ids(
                lexical_query,
                top_k=candidate_k,
            )
            fused_ids = reciprocal_rank_fusion(
                [dense_ids, bm25_ids],
                top_k=candidate_k,
                rrf_k=rrf_k,
            )
            child_ids = [
                child_id
                for child_id in fused_ids
                if child_id in children_by_id
            ]
            cached = {
                "rrf_child_ids": child_ids,
                "rrf_parent_ids": collapse_child_rankings(
                    child_ids,
                    child_to_parent,
                    top_k=candidate_k,
                ),
            }
            candidate_cache[query] = cached
            return cached

        def baseline_search(query: str, requested_top_k: int) -> list[str]:
            return retrieve_candidates(query)["rrf_parent_ids"][:requested_top_k]

        def get_granularity_ranking(
            query: str,
            granularity: int,
        ) -> dict[str, object]:
            cached = ranking_cache[granularity].get(query)
            if cached is not None:
                return cached

            query_id = query_id_by_text[query]
            candidates = retrieve_candidates(query)
            granules, child_to_granule = granule_views[granularity]
            candidate_granule_ids: list[str] = []
            for child_id in candidates["rrf_child_ids"]:
                granule_id = child_to_granule[child_id]
                if granule_id not in candidate_granule_ids:
                    candidate_granule_ids.append(granule_id)
            reranker_documents = [
                str(granules[granule_id]["text"])
                for granule_id in candidate_granule_ids
            ]

            reranked = reranker.rerank(query, reranker_documents)
            reranked_granule_ids: list[str] = []
            reranker_scores: dict[str, float] = {}
            for index, score in reranked:
                if not 0 <= index < len(candidate_granule_ids):
                    continue
                granule_id = candidate_granule_ids[index]
                if granule_id in reranker_scores:
                    continue
                reranked_granule_ids.append(granule_id)
                reranker_scores[granule_id] = score

            # Keep omitted granules in the original RRF order.
            for granule_id in candidate_granule_ids:
                if granule_id not in reranker_scores:
                    reranked_granule_ids.append(granule_id)

            reranked_child_ids: list[str] = []
            for granule_id in reranked_granule_ids:
                reranked_child_ids.extend(
                    str(child_id)
                    for child_id in granules[granule_id]["child_ids"]
                )
            reranked_parent_ids = collapse_child_rankings(
                reranked_child_ids,
                child_to_parent,
                top_k=candidate_k,
            )
            cached = {
                "query_id": query_id,
                "reranker_query": query,
                "candidate_granule_ids": candidate_granule_ids,
                "reranked_granule_ids": reranked_granule_ids,
                "reranker_scores": reranker_scores,
                "reranked_parent_ids": reranked_parent_ids,
            }
            ranking_cache[granularity][query] = cached
            return cached

        print("evaluating RRF baseline", flush=True)
        baseline = evaluate_retriever(
            queries,
            parent_qrels,
            baseline_search,
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="rewritten-metadata-dense-bm25-rrf-child-parent-collapse",
        )

        reranked_results: dict[int, dict[str, object]] = {}
        for granularity in GRANULARITIES:
            print(
                f"evaluating reranker granularity={granularity}",
                flush=True,
            )

            def search(
                query: str,
                requested_top_k: int,
                *,
                current_granularity: int = granularity,
            ) -> list[str]:
                ranking = get_granularity_ranking(
                    query,
                    current_granularity,
                )
                return list(ranking["reranked_parent_ids"])[
                    :requested_top_k
                ]

            reranked_results[granularity] = evaluate_retriever(
                queries,
                parent_qrels,
                search,
                top_k=top_k,
                k_values=(1, 3, 5, 8),
                retriever_name=(
                    f"reranker-granularity-{granularity}-child-windows"
                ),
            )
    finally:
        bm25.close()
        client.close()

    answerable_query_ids = {
        query_id
        for query_id, relevance in parent_qrels.items()
        if any(grade > 0 for grade in relevance.values())
    }
    query_groups: dict[str, set[str]] = {}
    for query in queries:
        metadata = query.get("metadata")
        query_type = metadata.get("type") if isinstance(metadata, Mapping) else None
        if (
            isinstance(query_type, str)
            and query_type in {"exact-term", "semantic"}
            and query["_id"] in answerable_query_ids
        ):
            query_groups.setdefault(query_type, set()).add(query["_id"])

    query_diagnostics: dict[str, dict[str, object]] = {}
    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]
        candidates = candidate_cache[query_text]
        baseline_parent_ids = candidates["rrf_parent_ids"]
        baseline_ranks = _rank_map(baseline_parent_ids)
        per_granularity: dict[str, object] = {}
        for granularity in GRANULARITIES:
            ranking = ranking_cache[granularity][query_text]
            reranked_parent_ids = list(ranking["reranked_parent_ids"])
            reranked_ranks = _rank_map(reranked_parent_ids)
            relevant_parent_rankings = []
            for parent_id, relevance in sorted(
                parent_qrels.get(query_id, {}).items(),
                key=lambda item: (
                    baseline_ranks.get(item[0], 10**9),
                    item[0],
                ),
            ):
                rrf_rank = baseline_ranks.get(parent_id)
                reranker_rank = reranked_ranks.get(parent_id)
                relevant_parent_rankings.append(
                    {
                        "parent_id": parent_id,
                        "relevance": relevance,
                        "rrf_rank": rrf_rank,
                        "reranker_rank": reranker_rank,
                        "rank_delta": (
                            reranker_rank - rrf_rank
                            if rrf_rank is not None and reranker_rank is not None
                            else None
                        ),
                    }
                )
            per_granularity[str(granularity)] = {
                "candidate_granule_count": len(
                    ranking["candidate_granule_ids"]
                ),
                "relevant_parent_rankings": relevant_parent_rankings,
            }
        query_diagnostics[query_id] = {
            "query_type": (
                query.get("metadata", {}).get("type")
                if isinstance(query.get("metadata"), Mapping)
                else None
            ),
            "candidate_child_count": len(candidates["rrf_child_ids"]),
            "candidate_parent_count": len(baseline_parent_ids),
            "granularities": per_granularity,
        }

    result = {
        "experiment_name": "reranker_granularity_eval_v1",
        "query_mode": "original",
        "retrieval_query_mode": "rewritten_terms_for_bm25_only",
        "reranker_level": "child_window_granularity",
        "reranker_model": reranker_model,
        "granularities": list(GRANULARITIES),
        "candidate_k": candidate_k,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "parent_size": parent_size,
        "metadata_weight": metadata_weight,
        "fusion_method": "pure_reranker",
        "collapse_method": "best_child",
        "granularity_definition": (
            "Source-order windows of 1, 2, or 4 adjacent Children; "
            "selected windows are mapped back to the default Parent "
            "before best-child collapse."
        ),
        "dataset": {
            "corpus_path": str(corpus_path),
            "queries_path": str(queries_path),
            "qrels_path": str(qrels_path),
            "rewrite_terms_source": str(full_combo_path),
            "query_count": len(queries),
            "answerable_query_count": len(answerable_query_ids),
        },
        "input_stats": {
            str(granularity): _input_stats(granule_views[granularity][0])
            for granularity in GRANULARITIES
        },
        "metrics": {
            "baseline": baseline["aggregate"],
            **{
                f"granularity_{granularity}": reranked_results[granularity][
                    "aggregate"
                ]
                for granularity in GRANULARITIES
            },
        },
        "group_metrics": {
            query_type: {
                "baseline": _aggregate_query_group(
                    baseline,
                    query_ids,
                ),
                **{
                    f"granularity_{granularity}": _aggregate_query_group(
                        reranked_results[granularity],
                        query_ids,
                    )
                    for granularity in GRANULARITIES
                },
            }
            for query_type, query_ids in query_groups.items()
        },
        "baseline": baseline,
        "experiments": {
            str(granularity): reranked_results[granularity]
            for granularity in GRANULARITIES
        },
        "ranking_traces": {
            str(granularity): {
                query_id_by_text[query["text"]]: ranking_cache[granularity][
                    query["text"]
                ]
                for query in queries
            }
            for granularity in GRANULARITIES
        },
        "query_diagnostics": query_diagnostics,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--parent-size", type=int, default=4)
    parser.add_argument("--metadata-weight", type=int, default=2)
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )
    args = parser.parse_args()
    result = run_reranker_granularity_eval(
        output_path=args.output,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        parent_size=args.parent_size,
        metadata_weight=args.metadata_weight,
        reranker_model=args.reranker_model,
    )
    print(
        json.dumps(
            {
                "metrics": result["metrics"],
                "group_metrics": result["group_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
