"""Evaluate Dense + LLM-rewritten BM25 + RRF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_openai import ChatOpenAI

from deep_research.config import settings
from deep_research.rag.lexical import SqliteBm25Index
from deep_research.rag.query_rewrite import (
    QueryRewriter,
    RetrievalQueryRewrite,
    build_rewritten_lexical_query,
)
from deep_research.evaluation.rag_runner import (
    evaluate_retriever,
    load_eval_queries,
    load_qrels,
)
from deep_research.rag.fusion import reciprocal_rank_fusion
from deep_research.tests.run_hybrid_rrf_eval import (
    load_corpus_rows,
    load_dense_rankings,
)


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
DENSE_RESULT_PATH = SOURCE_DIR / "dense_depth32.json"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "hybrid_rewritten_rrf_depth32.json"


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


def run_rewritten_hybrid_eval(
    *,
    corpus_path: str | Path = CORPUS_PATH,
    queries_path: str | Path = QUERIES_PATH,
    qrels_path: str | Path = QRELS_PATH,
    dense_result_path: str | Path = DENSE_RESULT_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    top_k: int = 8,
    candidate_k: int = 32,
    rrf_k: int = 60,
    rewriter: QueryRewriter | None = None,
) -> dict[str, object]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")

    queries = load_eval_queries(queries_path)
    qrels = load_qrels(qrels_path)
    dense_rankings = load_dense_rankings(dense_result_path)
    query_id_by_text = {query["text"]: query["_id"] for query in queries}
    query_rewriter = rewriter or QueryRewriter(
        build_eval_model(),
        structured_output=False,
    )
    rewritten_terms: dict[str, list[str]] = {}

    index = SqliteBm25Index()
    try:
        index.upsert_chunks(load_corpus_rows(corpus_path))

        def search(query: str, requested_top_k: int) -> list[str]:
            query_id = query_id_by_text[query]
            dense_ids = dense_rankings[query_id]
            if len(dense_ids) < candidate_k:
                raise ValueError(
                    f"Dense result for {query_id} has only "
                    f"{len(dense_ids)} candidates"
                )
            print(f"rewriting {query_id}", flush=True)
            rewrite: RetrievalQueryRewrite = query_rewriter.rewrite(query)
            rewritten_terms[query_id] = rewrite.terms
            lexical_query = build_rewritten_lexical_query(
                query,
                rewrite.terms,
            )
            bm25_ids = index.search_ids(
                lexical_query,
                top_k=candidate_k,
            )
            return reciprocal_rank_fusion(
                [dense_ids[:candidate_k], bm25_ids],
                top_k=requested_top_k,
                rrf_k=rrf_k,
            )

        result = evaluate_retriever(
            queries,
            qrels,
            search,
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="hybrid-dense-llm-rewritten-bm25-rrf",
        )
    finally:
        index.close()

    result["candidate_k"] = candidate_k
    result["rrf_k"] = rrf_k
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
    parser.add_argument("--dense-result", type=Path, default=DENSE_RESULT_PATH)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()
    result = run_rewritten_hybrid_eval(
        dense_result_path=args.dense_result,
        output_path=args.output,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
