"""Evaluate rewritten Dense + BM25/RRF retrieval with parent-child collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from deep_research.config import settings
from deep_research.evaluation.parent_child import build_parent_child_view
from deep_research.rag.query_rewrite import QueryRewriter
from deep_research.evaluation.rag_runner import (
    evaluate_retriever,
    load_eval_queries,
    load_qrels,
)
from deep_research.rag.lexical import SqliteBm25Index
from deep_research.rag.factory import create_embeddings, ensure_collection
from deep_research.rag.service import RagService
from deep_research.tests.run_hybrid_rrf_eval import load_corpus_rows


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "rewritten_parent_child_eval.json"
EVAL_COLLECTION = "rag_eval_rewritten_parent_child_v1"


def build_eval_model() -> ChatOpenAI:
    kwargs = {
        "model": settings.model_name,
        "api_key": settings.model_api_key,
        "temperature": 0,
        "timeout": 20,
        "max_retries": 0,
    }
    if settings.model_base_url:
        kwargs["base_url"] = settings.model_base_url
    return ChatOpenAI(**kwargs)


class RecordingQueryRewriter:
    """Record model rewrites while delegating retrieval to RagService."""

    def __init__(self, delegate, query_ids, output) -> None:
        self.delegate = delegate
        self.query_ids = query_ids
        self.output = output

    def rewrite(self, query: str):
        result = self.delegate.rewrite(query)
        query_id = self.query_ids.get(query)
        if query_id is not None:
            self.output[query_id] = list(result.terms)
        return result


def map_qrels_to_parents(
    qrels: dict[str, dict[str, int]],
    child_to_parent: dict[str, str],
) -> dict[str, dict[str, int]]:
    mapped: dict[str, dict[str, int]] = {}
    for query_id, child_grades in qrels.items():
        for child_id, grade in child_grades.items():
            parent_id = child_to_parent.get(child_id)
            if parent_id is None:
                continue
            mapped.setdefault(query_id, {})[parent_id] = max(
                grade,
                mapped.setdefault(query_id, {}).get(parent_id, 0),
            )
    return mapped


def run_rewritten_parent_child_eval(
    *,
    corpus_path: str | Path = CORPUS_PATH,
    queries_path: str | Path = QUERIES_PATH,
    qrels_path: str | Path = QRELS_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    top_k: int = 8,
    candidate_k: int = 32,
    rrf_k: int = 60,
    parent_size: int = 4,
    metadata_weight: int = 2,
    rewriter: QueryRewriter | None = None,
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
    queries = load_eval_queries(queries_path)
    parent_qrels = map_qrels_to_parents(
        load_qrels(qrels_path),
        child_to_parent,
    )
    query_id_by_text = {query["text"]: query["_id"] for query in queries}
    query_rewriter = rewriter or QueryRewriter(
        build_eval_model(),
        structured_output=False,
    )
    rewritten_terms: dict[str, list[str]] = {}
    query_rewriter = RecordingQueryRewriter(
        query_rewriter,
        query_id_by_text,
        rewritten_terms,
    )

    client = QdrantClient(location=":memory:")
    bm25 = SqliteBm25Index()
    rag_service = None
    try:
        for child in children:
            metadata = child["metadata"]
            parent_id = metadata.get("parent_id")
            if isinstance(parent_id, str) and parent_id in _parents:
                metadata["parent_child_ids"] = list(
                    _parents[parent_id]["metadata"]["child_ids"]
                )
        embeddings = create_embeddings(settings)
        collection_name = EVAL_COLLECTION
        ensure_collection(
            client=client,
            collection_name=collection_name,
            dimensions=settings.embedding_dimensions,
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            validate_collection_config=False,
        )
        rag_service = RagService(
            client=client,
            embeddings=embeddings,
            vector_store=vector_store,
            collection_name=collection_name,
            metadata_weight=metadata_weight,
            bm25_index=bm25,
            query_rewriter=query_rewriter,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )
        documents = []
        for row in children:
            metadata = dict(row["metadata"])
            metadata["chunk_id"] = row["_id"]
            metadata["status"] = "indexed"
            documents.append(
                Document(
                    page_content=str(row["text"]),
                    metadata=metadata,
                )
            )

        print("indexing full combination", flush=True)
        rag_service.index_chunks(documents)

        def search(query: str, requested_top_k: int) -> list[str]:
            query_id = query_id_by_text[query]
            print(f"searching {query_id}", flush=True)
            contexts = rag_service.search_contexts(
                query,
                top_k=requested_top_k,
            )
            return [
                str(document.metadata["parent_id"])
                for document, _score in contexts
                if isinstance(document.metadata.get("parent_id"), str)
            ]

        result = evaluate_retriever(
            queries,
            parent_qrels,
            search,
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="rewritten-dense-metadata-bm25-rrf-parent-child",
        )
    finally:
        if rag_service is not None:
            rag_service.close()
        else:
            bm25.close()

    result["candidate_k"] = candidate_k
    result["rrf_k"] = rrf_k
    result["parent_size"] = parent_size
    result["metadata_weight"] = metadata_weight
    result["rewritten_terms"] = rewritten_terms
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
    args = parser.parse_args()
    result = run_rewritten_parent_child_eval(
        output_path=args.output,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        parent_size=args.parent_size,
        metadata_weight=args.metadata_weight,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
