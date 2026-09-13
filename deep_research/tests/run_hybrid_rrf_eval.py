"""Evaluate BM25 + existing Dense rankings with Reciprocal Rank Fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deep_research.rag.lexical import (
    SqliteBm25Index,
    build_lexical_query,
)
from deep_research.evaluation.rag_runner import (
    evaluate_retriever,
    load_eval_queries,
    load_qrels,
)
from deep_research.rag.fusion import reciprocal_rank_fusion


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
DENSE_RESULT_PATH = SOURCE_DIR / "dense_baseline.json"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "hybrid_rrf.json"


def load_corpus_rows(path: str | Path = CORPUS_PATH) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def load_dense_rankings(
    path: str | Path = DENSE_RESULT_PATH,
) -> dict[str, list[str]]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        row["query_id"]: list(row["retrieved_chunk_ids"])
        for row in result["per_query"]
    }


def run_hybrid_eval(
    *,
    corpus_path: str | Path = CORPUS_PATH,
    queries_path: str | Path = QUERIES_PATH,
    qrels_path: str | Path = QRELS_PATH,
    dense_result_path: str | Path = DENSE_RESULT_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    top_k: int = 8,
    candidate_k: int | None = None,
    rrf_k: int = 60,
    keyword_query: bool = False,
) -> dict[str, object]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    effective_candidate_k = top_k if candidate_k is None else candidate_k
    if effective_candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")

    queries = load_eval_queries(queries_path)
    qrels = load_qrels(qrels_path)
    dense_rankings = load_dense_rankings(dense_result_path)
    query_id_by_text = {query["text"]: query["_id"] for query in queries}
    if len(query_id_by_text) != len(queries):
        raise ValueError("Evaluation queries must have unique text")

    index = SqliteBm25Index()
    try:
        index.upsert_chunks(load_corpus_rows(corpus_path))

        def search(query: str, requested_top_k: int) -> list[str]:
            query_id = query_id_by_text[query]
            dense_ids = dense_rankings.get(query_id, [])
            if len(dense_ids) < effective_candidate_k:
                raise ValueError(
                    f"Dense result for {query_id} has only "
                    f"{len(dense_ids)} results; rerun Dense with at least "
                    f"{effective_candidate_k} candidates"
                )
            bm25_ids = index.search_ids(
                build_lexical_query(query) if keyword_query else query,
                top_k=effective_candidate_k,
            )
            return reciprocal_rank_fusion(
                [
                    dense_ids[:effective_candidate_k],
                    bm25_ids,
                ],
                top_k=requested_top_k,
                rrf_k=rrf_k,
            )

        result = evaluate_retriever(
            queries,
            qrels,
            search,
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="hybrid-dense-bm25-rrf",
        )
    finally:
        index.close()

    result["candidate_k"] = effective_candidate_k
    result["rrf_k"] = rrf_k
    result["bm25_query_mode"] = (
        "keywords" if keyword_query else "original"
    )
    result["dense_result_path"] = str(dense_result_path)
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
    parser.add_argument(
        "--dense-result",
        type=Path,
        default=DENSE_RESULT_PATH,
        help="Dense result JSON containing the candidate rankings.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidates per retriever; must be available in Dense result.",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--keyword-query",
        action="store_true",
        help="Use the rule-based lexical keyword query for BM25.",
    )
    args = parser.parse_args()
    result = run_hybrid_eval(
        dense_result_path=args.dense_result,
        output_path=args.output,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        keyword_query=args.keyword_query,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
