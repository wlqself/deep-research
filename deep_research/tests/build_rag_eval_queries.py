import csv
import json
from pathlib import Path


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
SEEDS_PATH = SOURCE_DIR / "query_seeds.json"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
REFERENCES_PATH = SOURCE_DIR / "references.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"


def _load_corpus_index() -> dict[tuple[str, int], str]:
    index: dict[tuple[str, int], str] = {}
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row["metadata"]
        key = (
            metadata["document_id"],
            metadata["chunk_index"],
        )
        if key in index:
            raise ValueError(f"duplicate corpus position: {key}")
        index[key] = row["_id"]
    return index


def build_queries(
    *,
    seeds_path: Path = SEEDS_PATH,
    queries_path: Path = QUERIES_PATH,
    references_path: Path = REFERENCES_PATH,
    qrels_path: Path = QRELS_PATH,
) -> int:
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    corpus_index = _load_corpus_index()
    queries = []
    references = []
    qrels = []
    query_ids: set[str] = set()

    for seed in seeds["queries"]:
        query_id = seed["query_id"]
        if query_id in query_ids:
            raise ValueError(f"duplicate query ID: {query_id}")
        query_ids.add(query_id)

        resolved_relevance = []
        for relevance in seed["relevant_chunks"]:
            key = (
                relevance["document_id"],
                relevance["chunk_index"],
            )
            try:
                chunk_id = corpus_index[key]
            except KeyError as exc:
                raise ValueError(
                    f"unknown corpus position for {query_id}: {key}"
                ) from exc

            grade = relevance["grade"]
            if grade not in {1, 2}:
                raise ValueError(f"unsupported relevance grade: {grade}")

            resolved_relevance.append(
                {"chunk_id": chunk_id, "grade": grade}
            )
            qrels.append((query_id, chunk_id, grade))

        queries.append(
            {
                "_id": query_id,
                "text": seed["query"],
                "metadata": {
                    "type": seed["type"],
                    "difficulty": seed["difficulty"],
                    "answerable": seed["answerable"],
                    "tags": seed["tags"],
                },
            }
        )
        references.append(
            {
                "query_id": query_id,
                "expected_answer": seed["expected_answer"],
                "expected_facts": seed["expected_facts"],
                "answerable": seed["answerable"],
            }
        )

    queries_path.write_text(
        "".join(
            json.dumps(query, ensure_ascii=False) + "\n"
            for query in queries
        ),
        encoding="utf-8",
    )
    references_path.write_text(
        "".join(
            json.dumps(reference, ensure_ascii=False) + "\n"
            for reference in references
        ),
        encoding="utf-8",
    )

    with qrels_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(qrels)

    return len(queries)


if __name__ == "__main__":
    count = build_queries()
    print(f"Generated {count} queries at {QUERIES_PATH}")
