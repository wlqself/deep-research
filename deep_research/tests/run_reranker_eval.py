"""Evaluate a SiliconFlow reranker on the full hybrid parent-child pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from deep_research.config import settings
from deep_research.rag.lexical import SqliteBm25Index
from deep_research.evaluation.parent_child import (
    build_metadata_weighted_text,
    build_parent_child_view,
    build_reranker_text,
    collapse_child_rankings,
)
from deep_research.evaluation.rag_runner import (
    evaluate_retriever,
    load_eval_queries,
    load_qrels,
)
from deep_research.rag.fusion import (
    reciprocal_rank_fusion,
)
from deep_research.rag.query_rewrite import (
    build_rewritten_lexical_query,
)
from deep_research.evaluation.reranker import build_reranker_query
from deep_research.rag.factory import create_embeddings, ensure_collection
from deep_research.rag.service import RagService
from deep_research.tests.run_hybrid_rrf_eval import load_corpus_rows
from deep_research.tests.run_rewritten_parent_child_eval import map_qrels_to_parents


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
FULL_COMBO_PATH = SOURCE_DIR / "rewritten_parent_child_eval.json"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "reranker_eval.json"
EVAL_COLLECTION = "rag_eval_reranker_v1"


class SiliconFlowReranker:
    def __init__(self, *, model: str, top_n: int) -> None:
        base_url = (
            settings.embedding_base_url
            or settings.model_base_url
            or ""
        ).rstrip("/")
        self.url = f"{base_url}/rerank"
        self.api_key = settings.embedding_api_key or settings.model_api_key
        self.model = model
        self.top_n = top_n

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": min(self.top_n, len(documents)),
                "return_documents": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("rerank response has no results list")
        ranked: list[tuple[int, float]] = []
        for item in results:
            index = item.get("index") if isinstance(item, dict) else None
            score = item.get("relevance_score") if isinstance(item, dict) else None
            if (
                isinstance(index, int)
                and 0 <= index < len(documents)
                and isinstance(score, (int, float))
            ):
                ranked.append((index, float(score)))
        if not ranked:
            raise ValueError("rerank response returned no valid indices")
        return ranked


def run_reranker_eval(
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
    reranker_weight: float = 0.5,
) -> dict[str, object]:
    rows = load_corpus_rows(corpus_path)
    children, parents, child_to_parent = build_parent_child_view(
        rows,
        parent_size=parent_size,
    )
    queries = load_eval_queries(queries_path)
    parent_qrels = map_qrels_to_parents(
        load_qrels(qrels_path),
        child_to_parent,
    )
    full_combo = json.loads(Path(full_combo_path).read_text(encoding="utf-8"))
    rewritten_terms = full_combo.get("rewritten_terms", {})
    if not isinstance(rewritten_terms, dict):
        rewritten_terms = {}
    query_id_by_text = {query["text"]: query["_id"] for query in queries}

    if not 0 <= reranker_weight <= 1:
        raise ValueError("reranker_weight must be between 0 and 1")

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
        print("indexing reranker candidates", flush=True)
        rag_service.index_chunks(documents)

        reranker = SiliconFlowReranker(
            model=reranker_model,
            top_n=candidate_k,
        )
        candidate_cache: dict[str, tuple[list[str], dict[str, float]]] = {}
        reranker_cache: dict[str, dict[str, object]] = {}

        def retrieve_candidates(query: str) -> tuple[list[str], dict[str, float]]:
            cached = candidate_cache.get(query)
            if cached is not None:
                return cached
            query_id = query_id_by_text[query]
            dense_results = rag_service.search_chunks(query, top_k=candidate_k)
            dense_ids = [
                document.metadata["chunk_id"]
                for document, _score in dense_results
            ]
            terms = rewritten_terms.get(query_id, [])
            if not isinstance(terms, list):
                terms = []
            lexical_query = build_rewritten_lexical_query(query, terms)
            bm25_ids = bm25.search_ids(lexical_query, top_k=candidate_k)
            fused_ids = reciprocal_rank_fusion(
                [dense_ids, bm25_ids],
                top_k=candidate_k,
                rrf_k=rrf_k,
            )
            # Collapse before reranking: one parent gets one candidate slot.
            parent_ids = collapse_child_rankings(
                fused_ids,
                child_to_parent,
                top_k=candidate_k,
            )
            # Use the collapsed RRF order as the parent-level RRF signal.
            # This guarantees that blend weight 0 reproduces the baseline.
            parent_rrf_scores = {
                parent_id: 1 / (rrf_k + rank)
                for rank, parent_id in enumerate(parent_ids, start=1)
            }
            candidate_cache[query] = (parent_ids, parent_rrf_scores)
            return candidate_cache[query]

        def baseline_search(query: str, requested_top_k: int) -> list[str]:
            parent_ids, _scores = retrieve_candidates(query)
            return parent_ids[:requested_top_k]

        def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return {}
            minimum = min(scores.values())
            maximum = max(scores.values())
            if maximum == minimum:
                return {key: 1.0 for key in scores}
            return {
                key: (value - minimum) / (maximum - minimum)
                for key, value in scores.items()
            }

        def get_reranker_ranking(query: str) -> dict[str, object]:
            cached = reranker_cache.get(query)
            if cached is not None:
                return cached

            query_id = query_id_by_text[query]
            parent_ids, parent_rrf_scores = retrieve_candidates(query)
            candidate_ids = [parent_id for parent_id in parent_ids if parent_id in parents]
            candidate_texts = [
                build_reranker_text(parents[parent_id])
                for parent_id in candidate_ids
            ]
            terms = rewritten_terms.get(query_id, [])
            if not isinstance(terms, list):
                terms = []
            rerank_query = build_reranker_query(query, terms)
            reranked = reranker.rerank(rerank_query, candidate_texts)

            reranked_ids: list[str] = []
            reranker_scores: dict[str, float] = {}
            for index, score in reranked:
                if index >= len(candidate_ids):
                    continue
                parent_id = candidate_ids[index]
                if parent_id in reranker_scores:
                    continue
                reranked_ids.append(parent_id)
                reranker_scores[parent_id] = score

            # Preserve candidates if a service returns fewer than requested.
            for parent_id in candidate_ids:
                if parent_id not in reranker_scores:
                    reranked_ids.append(parent_id)

            normalized_reranker = _normalize_scores(reranker_scores)
            normalized_rrf = _normalize_scores({
                parent_id: parent_rrf_scores.get(parent_id, 0.0)
                for parent_id in candidate_ids
            })
            blended_scores = {
                parent_id: (
                    reranker_weight * normalized_reranker.get(parent_id, 0.0)
                    + (1 - reranker_weight) * normalized_rrf.get(parent_id, 0.0)
                )
                for parent_id in candidate_ids
            }
            blended_ids = [
                parent_id
                for parent_id, _score in sorted(
                    blended_scores.items(),
                    key=lambda item: (-item[1], candidate_ids.index(item[0])),
                )
            ]
            cached = {
                "query_id": query_id,
                "reranker_query": rerank_query,
                "candidate_parent_ids": candidate_ids,
                "rrf_scores": parent_rrf_scores,
                "reranker_scores": reranker_scores,
                "reranked_parent_ids": reranked_ids,
                "blended_parent_ids": blended_ids,
            }
            reranker_cache[query] = cached
            return cached

        def reranked_search(query: str, requested_top_k: int) -> list[str]:
            ranking = get_reranker_ranking(query)
            return list(ranking["reranked_parent_ids"])[:requested_top_k]

        def blended_search(query: str, requested_top_k: int) -> list[str]:
            ranking = get_reranker_ranking(query)
            return list(ranking["blended_parent_ids"])[:requested_top_k]

        print("evaluating baseline", flush=True)
        baseline = evaluate_retriever(
            queries,
            parent_qrels,
            baseline_search,
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="full-combo-without-reranker",
        )
        print("evaluating reranker", flush=True)
        reranked = evaluate_retriever(
            queries,
            parent_qrels,
            reranked_search,
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="full-combo-with-bge-reranker-v2-m3",
        )
        print("evaluating RRF/reranker score blend", flush=True)
        blended = evaluate_retriever(
            queries,
            parent_qrels,
            blended_search,
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="full-combo-with-bge-reranker-v2-m3-rrf-blend",
        )
    finally:
        bm25.close()
        client.close()

    result = {
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "parent_size": parent_size,
        "metadata_weight": metadata_weight,
        "reranker_model": reranker_model,
        "reranker_weight": reranker_weight,
        "baseline": baseline,
        "reranked": reranked,
        "blended": blended,
        "ranking_traces": {
            query_id_by_text[query["text"]]: trace
            for query in queries
            for trace in [reranker_cache.get(query["text"])]
            if trace is not None
        },
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
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--reranker-weight", type=float, default=0.5)
    args = parser.parse_args()
    result = run_reranker_eval(
        output_path=args.output,
        candidate_k=args.candidate_k,
        reranker_model=args.reranker_model,
        reranker_weight=args.reranker_weight,
    )
    print(json.dumps({
        "baseline": result["baseline"]["aggregate"],
        "reranked": result["reranked"]["aggregate"],
        "blended": result["blended"]["aggregate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
