"""Compare Child reranking quality at several RRF candidate widths.

This is an isolated evaluation. It keeps the production retrieval pipeline
unchanged and varies only ``candidate_k`` for the Child reranker experiment.
The reranker receives the original query and Child text only. Parent collapse
uses the existing best-child rule after reranking.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping
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
from deep_research.tests.run_child_reranker_eval import _load_rewritten_terms
from deep_research.tests.run_reranker_eval import SiliconFlowReranker
from deep_research.tests.run_rewritten_parent_child_eval import (
    map_qrels_to_parents,
)


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
FULL_COMBO_PATH = SOURCE_DIR / "rewritten_parent_child_eval.json"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "reranker_candidate_k_eval_v1.json"
EVAL_COLLECTION = "rag_eval_reranker_candidate_k_v1"
DEFAULT_CANDIDATE_KS = (32, 64, 100)
K_VALUES = (1, 3, 5, 8)


def _rank_map(parent_ids: list[str]) -> dict[str, int]:
    return {
        parent_id: rank
        for rank, parent_id in enumerate(parent_ids, start=1)
    }


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


def _stats(values: list[int]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _build_query_groups(
    queries: list[dict[str, object]],
    parent_qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, set[str]]:
    answerable_query_ids = {
        query_id
        for query_id, relevance in parent_qrels.items()
        if any(grade > 0 for grade in relevance.values())
    }
    groups: dict[str, set[str]] = {}
    for query in queries:
        metadata = query.get("metadata")
        query_type = metadata.get("type") if isinstance(metadata, Mapping) else None
        if (
            isinstance(query_type, str)
            and query_type in {"exact-term", "semantic"}
            and query["_id"] in answerable_query_ids
        ):
            groups.setdefault(query_type, set()).add(str(query["_id"]))
    return groups


def run_reranker_candidate_k_eval(
    *,
    corpus_path: str | Path = CORPUS_PATH,
    queries_path: str | Path = QUERIES_PATH,
    qrels_path: str | Path = QRELS_PATH,
    full_combo_path: str | Path = FULL_COMBO_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    top_k: int = 8,
    candidate_ks: tuple[int, ...] = DEFAULT_CANDIDATE_KS,
    rrf_k: int = 60,
    parent_size: int = 4,
    metadata_weight: int = 2,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
) -> dict[str, object]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not candidate_ks or any(candidate_k < top_k for candidate_k in candidate_ks):
        raise ValueError("every candidate_k must be at least top_k")
    if tuple(sorted(set(candidate_ks))) != candidate_ks:
        raise ValueError("candidate_ks must be sorted and unique")

    rows = load_corpus_rows(corpus_path)
    children, _parents, child_to_parent = build_parent_child_view(
        rows,
        parent_size=parent_size,
    )
    children_by_id = {str(child["_id"]): child for child in children}
    queries = load_eval_queries(queries_path)
    qrels = load_qrels(qrels_path)
    parent_qrels = map_qrels_to_parents(qrels, child_to_parent)
    query_id_by_text = {str(query["text"]): str(query["_id"]) for query in queries}
    if len(query_id_by_text) != len(queries):
        raise ValueError("Evaluation queries must have unique text")
    rewritten_terms = _load_rewritten_terms(full_combo_path)
    query_groups = _build_query_groups(queries, parent_qrels)

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

        print("indexing candidate-k evaluation corpus", flush=True)
        rag_service.index_chunks(documents)

        candidate_cache: dict[int, dict[str, dict[str, object]]] = {
            candidate_k: {}
            for candidate_k in candidate_ks
        }
        ranking_cache: dict[int, dict[str, dict[str, object]]] = {
            candidate_k: {}
            for candidate_k in candidate_ks
        }
        rerankers = {
            candidate_k: SiliconFlowReranker(
                model=reranker_model,
                top_n=candidate_k,
            )
            for candidate_k in candidate_ks
        }

        def retrieve_candidates(
            query: str,
            candidate_k: int,
        ) -> dict[str, object]:
            cached = candidate_cache[candidate_k].get(query)
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
                "query_id": query_id,
                "rrf_child_ids": child_ids,
                "rrf_parent_ids": collapse_child_rankings(
                    child_ids,
                    child_to_parent,
                    top_k=candidate_k,
                ),
            }
            candidate_cache[candidate_k][query] = cached
            return cached

        def get_reranker_ranking(
            query: str,
            candidate_k: int,
        ) -> dict[str, object]:
            cached = ranking_cache[candidate_k].get(query)
            if cached is not None:
                return cached

            candidates = retrieve_candidates(query, candidate_k)
            child_ids = list(candidates["rrf_child_ids"])
            child_documents = [
                str(children_by_id[child_id]["text"])
                for child_id in child_ids
            ]
            reranked = rerankers[candidate_k].rerank(
                query,
                child_documents,
            )
            reranked_child_ids: list[str] = []
            reranker_scores: dict[str, float] = {}
            for index, score in reranked:
                if not 0 <= index < len(child_ids):
                    continue
                child_id = child_ids[index]
                if child_id in reranker_scores:
                    continue
                reranked_child_ids.append(child_id)
                reranker_scores[child_id] = score

            # Preserve candidates omitted by the rerank service in RRF order.
            for child_id in child_ids:
                if child_id not in reranker_scores:
                    reranked_child_ids.append(child_id)

            reranked_parent_ids = collapse_child_rankings(
                reranked_child_ids,
                child_to_parent,
                top_k=candidate_k,
            )
            cached = {
                "query_id": query_id_by_text[query],
                "reranker_query": query,
                "candidate_child_ids": child_ids,
                "rrf_parent_ids": list(candidates["rrf_parent_ids"]),
                "reranker_scores": reranker_scores,
                "reranked_child_ids": reranked_child_ids,
                "reranked_parent_ids": reranked_parent_ids,
                "input_diagnostics": {
                    "query_char_count": len(query),
                    "document_count": len(child_documents),
                    "document_char_count": _stats(
                        [len(document) for document in child_documents]
                    ),
                    "total_document_char_count": sum(
                        len(document) for document in child_documents
                    ),
                    "token_count": None,
                    "truncation_detected": None,
                    "truncation_note": (
                        "The remote rerank API does not expose tokenizer or "
                        "truncation diagnostics; character lengths are recorded "
                        "for comparison."
                    ),
                },
            }
            ranking_cache[candidate_k][query] = cached
            return cached

        print("evaluating candidate-k baselines and rerankers", flush=True)
        baselines: dict[str, dict[str, object]] = {}
        reranked_results: dict[str, dict[str, object]] = {}
        for candidate_k in candidate_ks:
            def baseline_search(
                query: str,
                requested_top_k: int,
                *,
                current_candidate_k: int = candidate_k,
            ) -> list[str]:
                ranking = retrieve_candidates(query, current_candidate_k)
                return list(ranking["rrf_parent_ids"])[:requested_top_k]

            def reranked_search(
                query: str,
                requested_top_k: int,
                *,
                current_candidate_k: int = candidate_k,
            ) -> list[str]:
                ranking = get_reranker_ranking(query, current_candidate_k)
                return list(ranking["reranked_parent_ids"])[:requested_top_k]

            key = str(candidate_k)
            baselines[key] = evaluate_retriever(
                queries,
                parent_qrels,
                baseline_search,
                top_k=top_k,
                k_values=K_VALUES,
                retriever_name=(
                    f"rewritten-metadata-dense-bm25-rrf-child-parent-collapse-"
                    f"candidate-k-{candidate_k}"
                ),
            )
            reranked_results[key] = evaluate_retriever(
                queries,
                parent_qrels,
                reranked_search,
                top_k=top_k,
                k_values=K_VALUES,
                retriever_name=(
                    f"child-reranker-original-query-candidate-k-{candidate_k}"
                ),
            )
    finally:
        bm25.close()
        client.close()

    query_diagnostics: dict[str, dict[str, object]] = {}
    for query in queries:
        query_id = str(query["_id"])
        query_text = str(query["text"])
        per_candidate_k: dict[str, object] = {}
        for candidate_k in candidate_ks:
            key = str(candidate_k)
            candidates = candidate_cache[candidate_k][query_text]
            ranking = ranking_cache[candidate_k][query_text]
            rrf_ranks = _rank_map(list(candidates["rrf_parent_ids"]))
            reranked_ranks = _rank_map(list(ranking["reranked_parent_ids"]))
            relevant_parent_rankings = []
            for parent_id, relevance in sorted(
                parent_qrels.get(query_id, {}).items(),
                key=lambda item: (
                    rrf_ranks.get(item[0], 10**9),
                    item[0],
                ),
            ):
                rrf_rank = rrf_ranks.get(parent_id)
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
            per_candidate_k[key] = {
                "candidate_child_count": len(candidates["rrf_child_ids"]),
                "candidate_parent_count": len(candidates["rrf_parent_ids"]),
                "relevant_parent_rankings": relevant_parent_rankings,
                "input_diagnostics": ranking["input_diagnostics"],
            }
        query_diagnostics[query_id] = {
            "query_type": query.get("metadata", {}).get("type")
            if isinstance(query.get("metadata"), Mapping)
            else None,
            "candidate_k": per_candidate_k,
        }

    result = {
        "experiment_name": "reranker_candidate_k_eval_v1",
        "query_mode": "original",
        "retrieval_query_mode": "rewritten_terms_for_bm25_only",
        "reranker_level": "child",
        "reranker_model": reranker_model,
        "candidate_ks": list(candidate_ks),
        "top_k": top_k,
        "metric_k_values": list(K_VALUES),
        "rrf_k": rrf_k,
        "parent_size": parent_size,
        "metadata_weight": metadata_weight,
        "reranker_granularity": 1,
        "fusion_method": "pure_reranker_after_rrf",
        "collapse_method": "best_child",
        "production_code_modified": False,
        "candidate_k_is_only_variable": True,
        "dataset": {
            "corpus_path": str(corpus_path),
            "queries_path": str(queries_path),
            "qrels_path": str(qrels_path),
            "rewrite_terms_source": str(full_combo_path),
            "query_count": len(queries),
            "answerable_query_count": sum(
                1
                for query_id, relevance in parent_qrels.items()
                if any(grade > 0 for grade in relevance.values())
            ),
        },
        "metrics": {
            "baseline": baselines,
            "child_reranker": reranked_results,
        },
        "group_metrics": {
            query_type: {
                "baseline": {
                    key: _aggregate_query_group(result, query_ids)
                    for key, result in baselines.items()
                },
                "child_reranker": {
                    key: _aggregate_query_group(result, query_ids)
                    for key, result in reranked_results.items()
                },
            }
            for query_type, query_ids in query_groups.items()
        },
        "baseline": baselines,
        "experiments": reranked_results,
        "ranking_traces": {
            key: {
                query_id_by_text[query["text"]]: ranking_cache[int(key)][
                    query["text"]
                ]
                for query in queries
            }
            for key in (str(candidate_k) for candidate_k in candidate_ks)
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
    parser.add_argument(
        "--candidate-ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_KS),
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--parent-size", type=int, default=4)
    parser.add_argument("--metadata-weight", type=int, default=2)
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )
    args = parser.parse_args()
    candidate_ks = tuple(args.candidate_ks)
    result = run_reranker_candidate_k_eval(
        output_path=args.output,
        top_k=args.top_k,
        candidate_ks=candidate_ks,
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
