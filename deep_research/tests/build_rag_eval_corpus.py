import hashlib
import json
from pathlib import Path

from deep_research.rag.chunking import chunk_documents
from deep_research.rag.parsers import parse_document


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
MANIFEST_PATH = SOURCE_DIR / "pdf_sources.json"
OUTPUT_PATH = SOURCE_DIR / "corpus.jsonl"


def build_corpus(
    *,
    source_dir: Path = SOURCE_DIR,
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = OUTPUT_PATH,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict] = []

    for source in manifest["documents"]:
        source_path = source_dir / source["filename"]
        document_hash = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()

        documents, _ = parse_document(
            source_path,
            document_id=source["document_id"],
            collection_id=manifest["collection_id"],
            filename=source["filename"],
            mime_type=source["mime_type"],
        )
        chunks = chunk_documents(
            documents,
            document_hash=document_hash,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.pop("created_at", None)
            metadata["category"] = source["category"]
            rows.append(
                {
                    "_id": metadata["chunk_id"],
                    "title": metadata["title"],
                    "text": chunk.page_content,
                    "metadata": metadata,
                }
            )

    chunk_ids = [row["_id"] for row in rows]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("generated chunk IDs are not unique")

    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return len(rows)


if __name__ == "__main__":
    count = build_corpus()
    print(f"Generated {count} chunks at {OUTPUT_PATH}")
