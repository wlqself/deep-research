"""Evaluate the SQLite FTS5 BM25 retriever on the generated eval corpus."""

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
from deep_research.tests.run_hybrid_rrf_eval import load_corpus_rows


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "bm25_baseline.json"


def run_bm25_eval(
    *,
    corpus_path: str | Path = CORPUS_PATH,
    queries_path: str | Path = QUERIES_PATH,
    qrels_path: str | Path = QRELS_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    top_k: int = 8,
    keyword_query: bool = False,
) -> dict[str, object]:
    queries = load_eval_queries(queries_path)
    qrels = load_qrels(qrels_path)
    index = SqliteBm25Index()
    try:
        index.upsert_chunks(load_corpus_rows(corpus_path))
        result = evaluate_retriever(
            queries,
            qrels,
            lambda query, requested_top_k: index.search_ids(
                build_lexical_query(query) if keyword_query else query,
                top_k=requested_top_k,
            ),
            top_k=top_k,
            k_values=(1, 3, 5, 8),
            retriever_name="bm25-baseline",
        )
    finally:
        index.close()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    result["query_mode"] = "keywords" if keyword_query else "original"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--keyword-query",
        action="store_true",
        help="Use the rule-based lexical keyword query for BM25.",
    )
    args = parser.parse_args()
    result = run_bm25_eval(
        output_path=args.output,
        top_k=args.top_k,
        keyword_query=args.keyword_query,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
