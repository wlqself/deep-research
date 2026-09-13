"""Run the current Dense/Qdrant retriever against the generated eval corpus.

This module uses an in-memory Qdrant client, so it cannot modify the app's
normal Qdrant path or collection. Running it does call the configured
embedding API for the evaluation corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from deep_research.config import settings
from deep_research.evaluation.rag_runner import (
    evaluate_retriever,
    load_eval_queries,
    load_qrels,
)
from deep_research.rag.factory import create_embeddings, ensure_collection
from deep_research.rag.service import RagService


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "dense_baseline.json"
EVAL_COLLECTION = "rag_eval_dense_v1"


def load_corpus_documents(path: str | Path = CORPUS_PATH) -> list[Document]:
    documents: list[Document] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Corpus row {line_number} is not an object")

            chunk_id = row.get("_id")
            text = row.get("text")
            metadata = row.get("metadata")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Corpus row {line_number} has an invalid _id")
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"Corpus row {line_number} has invalid chunk text"
                )
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Corpus row {line_number} has invalid metadata"
                )

            normalized_metadata = dict(metadata)
            normalized_metadata["chunk_id"] = chunk_id
            normalized_metadata["status"] = "indexed"
            documents.append(
                Document(
                    page_content=text,
                    metadata=normalized_metadata,
                )
            )
    return documents


def run_dense_eval(
    *,
    corpus_path: str | Path = CORPUS_PATH,
    queries_path: str | Path = QUERIES_PATH,
    qrels_path: str | Path = QRELS_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    top_k: int | None = None,
    k_values: tuple[int, ...] = (1, 3, 5, 8),
) -> dict[str, object]:
    """Index the eval corpus in memory and evaluate current Dense retrieval."""
    documents = load_corpus_documents(corpus_path)
    queries = load_eval_queries(queries_path)
    qrels = load_qrels(qrels_path)
    effective_top_k = settings.rag_top_k if top_k is None else top_k

    client = QdrantClient(location=":memory:")
    try:
        ensure_collection(
            client=client,
            collection_name=EVAL_COLLECTION,
            dimensions=settings.embedding_dimensions,
        )
        embeddings = create_embeddings(settings)
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
        )
        rag_service.index_chunks(documents)

        def search(query: str, requested_top_k: int) -> list[str]:
            results = rag_service.search_chunks(
                query,
                top_k=requested_top_k,
            )
            return [
                chunk.metadata["chunk_id"]
                for chunk, _score in results
            ]

        result = evaluate_retriever(
            queries,
            qrels,
            search,
            top_k=effective_top_k,
            k_values=k_values,
            retriever_name="dense-qdrant-baseline",
        )
    finally:
        client.close()

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
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the JSON result; use --no-output to skip it.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write a result file.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    result = run_dense_eval(
        output_path=None if args.no_output else args.output,
        top_k=args.top_k,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
