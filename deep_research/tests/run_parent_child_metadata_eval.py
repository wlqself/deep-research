"""Evaluate parent-child retrieval and metadata-weighted Dense retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from deep_research.config import settings
from deep_research.evaluation.parent_child import (
    build_metadata_weighted_text,
    build_parent_child_view,
    collapse_child_rankings,
)
from deep_research.evaluation.rag_runner import (
    evaluate_retriever,
    load_eval_queries,
    load_qrels,
)
from deep_research.rag.factory import create_embeddings, ensure_collection
from deep_research.rag.service import RagService
from deep_research.tests.run_hybrid_rrf_eval import (
    load_corpus_rows,
    load_dense_rankings,
)


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
DENSE_RESULT_PATH = SOURCE_DIR / "dense_depth32.json"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "parent_child_metadata_eval.json"
EVAL_COLLECTION = "rag_eval_parent_child_v1"


def map_qrels_to_parents(
    qrels: dict[str, dict[str, int]],
    child_to_parent: dict[str, str],
) -> dict[str, dict[str, int]]:
    mapped: dict[str, dict[str, int]] = {}
    for query_id, chunk_grades in qrels.items():
        for child_id, grade in chunk_grades.items():
            parent_id = child_to_parent.get(child_id)
            if parent_id is None:
                continue
            mapped.setdefault(query_id, {})[parent_id] = max(
                grade,
                mapped.setdefault(query_id, {}).get(parent_id, 0),
            )
    return mapped


def _load_parent_view(
    corpus_path: str | Path,
    qrels_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, int]]]:
    rows = load_corpus_rows(corpus_path)
    children, parents, child_to_parent = build_parent_child_view(rows, parent_size=4)
    qrels = load_qrels(qrels_path)
    return children, parents, child_to_parent, map_qrels_to_parents(qrels, child_to_parent)


def run_parent_child_metadata_eval(
    *,
    corpus_path: str | Path = CORPUS_PATH,
    queries_path: str | Path = QUERIES_PATH,
    qrels_path: str | Path = QRELS_PATH,
    dense_result_path: str | Path = DENSE_RESULT_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    top_k: int = 8,
    candidate_k: int = 32,
    metadata_weight: int = 2,
    variant: str = "all",
) -> dict[str, object]:
    variant_specs = {
        "metadata_weighted": False,
        "parent_context_metadata_weighted": True,
    }
    if variant != "all" and variant not in variant_specs:
        raise ValueError(f"Unknown variant: {variant}")
    selected_variants = (
        variant_specs
        if variant == "all"
        else {variant: variant_specs[variant]}
    )

    children, parents, child_to_parent, parent_qrels = _load_parent_view(
        corpus_path,
        qrels_path,
    )
    queries = load_eval_queries(queries_path)
    dense_rankings = load_dense_rankings(dense_result_path)
    query_id_by_text = {query["text"]: query["_id"] for query in queries}

    def baseline_search(query: str, requested_top_k: int) -> list[str]:
        query_id = query_id_by_text[query]
        return collapse_child_rankings(
            dense_rankings[query_id][:candidate_k],
            child_to_parent,
            top_k=requested_top_k,
        )

    baseline = evaluate_retriever(
        queries,
        parent_qrels,
        baseline_search,
        top_k=top_k,
        k_values=(1, 3, 5, 8),
        retriever_name="dense-parent-collapsed-baseline",
    )

    embeddings = create_embeddings(settings)
    client = QdrantClient(location=":memory:")
    variant_results: dict[str, object] = {}
    try:
        for variant_name, include_parent in selected_variants.items():
            print(f"indexing {variant_name}", flush=True)
            collection_name = f"{EVAL_COLLECTION}_{variant_name}"
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
            )
            documents = []
            for row in children:
                parent_id = row["metadata"]["parent_id"]
                parent_text = parents[parent_id]["text"] if include_parent else None
                metadata = dict(row["metadata"])
                metadata["chunk_id"] = row["_id"]
                metadata["status"] = "indexed"
                documents.append(
                    Document(
                        page_content=build_metadata_weighted_text(
                            row,
                            metadata_weight=metadata_weight,
                            parent_text=parent_text,
                        ),
                        metadata=metadata,
                    )
                )
            rag_service.index_chunks(documents)
            print(f"searching {variant_name}", flush=True)

            def search(query: str, requested_top_k: int) -> list[str]:
                query_id = query_id_by_text[query]
                child_results = rag_service.search_chunks(
                    query,
                    top_k=candidate_k,
                )
                child_ids = [document.metadata["chunk_id"] for document, _score in child_results]
                return collapse_child_rankings(
                    child_ids,
                    child_to_parent,
                    top_k=requested_top_k,
                )

            variant_results[variant_name] = evaluate_retriever(
                queries,
                parent_qrels,
                search,
                top_k=top_k,
                k_values=(1, 3, 5, 8),
                retriever_name=variant_name,
            )
    finally:
        client.close()

    result = {
        "parent_size": 4,
        "metadata_weight": metadata_weight,
        "baseline": baseline,
        "variants": variant_results,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-weight", type=int, default=2)
    parser.add_argument(
        "--variant",
        choices=(
            "all",
            "metadata_weighted",
            "parent_context_metadata_weighted",
        ),
        default="all",
    )
    args = parser.parse_args()
    output_path = args.output
    if args.variant != "all" and output_path == DEFAULT_OUTPUT_PATH:
        output_path = DEFAULT_OUTPUT_PATH.with_name(
            f"parent_child_metadata_{args.variant}.json"
        )
    result = run_parent_child_metadata_eval(
        output_path=output_path,
        metadata_weight=args.metadata_weight,
        variant=args.variant,
    )
    print(json.dumps({
        "baseline": result["baseline"]["aggregate"],
        "variants": {name: value["aggregate"] for name, value in result["variants"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
