"""Evaluate Child-level reranking without changing the production RAG path."""

from __future__ import annotations

import argparse
import json
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
from deep_research.tests.run_reranker_eval import SiliconFlowReranker
from deep_research.tests.run_rewritten_parent_child_eval import (
    map_qrels_to_parents,
)


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
FULL_COMBO_PATH = SOURCE_DIR / "rewritten_parent_child_eval.json"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "child_reranker_original_query_v1.json"
EVAL_COLLECTION = "rag_eval_child_reranker_v1"


def _load_rewritten_terms(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_terms = payload.get("rewritten_terms", {})
    if not isinstance(raw_terms, Mapping):
        return {}

    rewritten_terms: dict[str, list[str]] = {}
    for query_id, terms in raw_terms.items():
        if not isinstance(query_id, str) or not isinstance(terms, list):
            continue
        rewritten_terms[query_id] = [
            term.strip()
            for term in terms
            if isinstance(term, str) and term.strip()
        ]
    return rewritten_terms


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


def run_child_reranker_eval(
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
        str(row["_id"]): row
        for row in children
    }
    queries = load_eval_queries(queries_path)
    qrels = load_qrels(qrels_path)
    parent_qrels = map_qrels_to_parents(qrels, child_to_parent)
    query_id_by_text = {query["text"]: query["_id"] for query in queries}
    if len(query_id_by_text) != len(queries):
        raise ValueError("Evaluation queries must have unique text")
    rewritten_terms = _load_rewritten_terms(full_combo_path)

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
        for row in children:
            metadata = dict(row["metadata"])
            metadata["chunk_id"] = row["_id"]
            metadata["status"] = "indexed"
            documents.append(
                Document(
                    page_content=build_metadata_weighted_text(
                        row,
                        metadata_weight=metadata_weight,
                    ),
                    metadata=metadata,
                )
            )

        print("indexing Child reranker candidates", flush=True)
        rag_service.index_chunks(documents)

        reranker = SiliconFlowReranker(
            model=reranker_model,
            top_n=candidate_k,
        )
        candidate_cache: dict[str, dict[str, list[str]]] = {}
        reranker_cache: dict[str, dict[str, object]] = {}

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
            terms = rewritten_terms.get(query_id, [])
            lexical_query = build_rewritten_lexical_query(query, terms)
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
            candidates = retrieve_candidates(query)
            return candidates["rrf_parent_ids"][:requested_top_k]

        def get_reranker_ranking(query: str) -> dict[str, object]:
            cached = reranker_cache.get(query)
            if cached is not None:
                return cached

            query_id = query_id_by_text[query]
            candidates = retrieve_candidates(query)
            child_ids = candidates["rrf_child_ids"]
            child_documents = [
                str(children_by_id[child_id]["text"])
                for child_id in child_ids
            ]

            # Deliberately use the original query and Child text only.
            reranked = reranker.rerank(query, child_documents)
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

            # Preserve any candidates omitted by the service in RRF order.
            for child_id in child_ids:
                if child_id not in reranker_scores:
                    reranked_child_ids.append(child_id)

            reranked_parent_ids = collapse_child_rankings(
                reranked_child_ids,
                child_to_parent,
                top_k=candidate_k,
            )
            cached = {
                "query_id": query_id,
                "reranker_query": query,
                "candidate_child_ids": child_ids,
                "rrf_parent_ids": candidates["rrf_parent_ids"],
                "reranker_scores": reranker_scores,
                "reranked_child_ids": reranked_child_ids,
                "reranked_parent_ids": reranked_parent_ids,
            }
            reranker_cache[query] = cached
            return cached

        def reranked_search(query: str, requested_top_k: int) -> list[str]:
            ranking = get_reranker_ranking(query)
            return list(ranking["reranked_parent_ids"])[:requested_top_k]

        print("evaluating RRF Child baseline", flush=True)
        baseline = evaluate_retriever(
            queries,
            parent_qrels,
            baseline_search,
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="rewritten-metadata-dense-bm25-rrf-child-parent-collapse",
        )
        print("evaluating Child reranker", flush=True)
        reranked = evaluate_retriever(
            queries,
            parent_qrels,
            reranked_search,
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="child-reranker-original-query-parent-collapse",
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
        query_type = query.get("metadata", {}).get("type")
        if isinstance(query_type, str) and query_type in {"exact-term", "semantic"}:
            if query["_id"] in answerable_query_ids:
                query_groups.setdefault(query_type, set()).add(query["_id"])

    query_diagnostics: dict[str, dict[str, object]] = {}
    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]
        ranking = reranker_cache.get(query_text)
        if ranking is None:
            ranking = get_reranker_ranking(query_text)
        rrf_parent_ids = list(ranking["rrf_parent_ids"])
        reranked_parent_ids = list(ranking["reranked_parent_ids"])
        rrf_ranks = _rank_map(rrf_parent_ids)
        reranked_ranks = _rank_map(reranked_parent_ids)

        relevant_parent_ids = sorted(
            parent_qrels.get(query_id, {}),
            key=lambda parent_id: (
                rrf_ranks.get(parent_id, 10**9),
                parent_id,
            ),
        )
        relevant_parent_rankings = []
        for parent_id in relevant_parent_ids:
            rrf_rank = rrf_ranks.get(parent_id)
            reranked_rank = reranked_ranks.get(parent_id)
            relevant_parent_rankings.append(
                {
                    "parent_id": parent_id,
                    "relevance": parent_qrels[query_id][parent_id],
                    "rrf_rank": rrf_rank,
                    "child_reranker_rank": reranked_rank,
                    "rank_delta": (
                        reranked_rank - rrf_rank
                        if rrf_rank is not None and reranked_rank is not None
                        else None
                    ),
                }
            )

        query_diagnostics[query_id] = {
            "query_type": query.get("metadata", {}).get("type"),
            "candidate_child_count": len(ranking["candidate_child_ids"]),
            "candidate_parent_count": len(rrf_parent_ids),
            "relevant_parent_rankings": relevant_parent_rankings,
        }

    result = {
        "experiment_name": "child_reranker_original_query_v1",
        "query_mode": "original",
        "retrieval_query_mode": "rewritten_terms_for_bm25_only",
        "reranker_level": "child",
        "reranker_model": reranker_model,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "parent_size": parent_size,
        "metadata_weight": metadata_weight,
        "fusion_method": "pure_reranker",
        "collapse_method": "best_child",
        "dataset": {
            "corpus_path": str(corpus_path),
            "queries_path": str(queries_path),
            "qrels_path": str(qrels_path),
            "rewrite_terms_source": str(full_combo_path),
            "query_count": len(queries),
            "answerable_query_count": len(answerable_query_ids),
        },
        "metrics": {
            "baseline": baseline["aggregate"],
            "child_reranker": reranked["aggregate"],
        },
        "group_metrics": {
            query_type: {
                "baseline": _aggregate_query_group(
                    baseline,
                    query_ids,
                ),
                "child_reranker": _aggregate_query_group(
                    reranked,
                    query_ids,
                ),
            }
            for query_type, query_ids in query_groups.items()
        },
        "baseline": baseline,
        "child_reranker": reranked,
        "ranking_traces": {
            query_id_by_text[query["text"]]: reranker_cache[query["text"]]
            for query in queries
            if query["text"] in reranker_cache
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
    result = run_child_reranker_eval(
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
                "baseline": result["metrics"]["baseline"],
                "child_reranker": result["metrics"]["child_reranker"],
                "group_metrics": result["group_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
