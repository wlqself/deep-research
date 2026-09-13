import hashlib
import json
import unittest
from pathlib import Path

from deep_research.rag.chunking import chunk_documents
from deep_research.rag.parsers import parse_document


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
MANIFEST_PATH = SOURCE_DIR / "sources.json"


class RagEvaluationSourceChunkingTests(unittest.TestCase):
    def test_sources_are_chunked_by_the_production_pipeline(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        all_chunks = []

        for source in manifest["documents"]:
            source_path = SOURCE_DIR / source["filename"]
            document_hash = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()

            documents, page_count = parse_document(
                source_path,
                document_id=source["document_id"],
                collection_id=manifest["collection_id"],
                filename=source["filename"],
                mime_type=source["mime_type"],
            )
            chunks = chunk_documents(
                documents,
                document_hash=document_hash,
                chunk_size=1200,
                chunk_overlap=200,
            )

            with self.subTest(document_id=source["document_id"]):
                self.assertGreaterEqual(page_count, 1)
                self.assertGreaterEqual(len(documents), 1)
                self.assertGreaterEqual(len(chunks), 2)
                self.assertTrue(
                    all(
                        chunk.metadata["document_id"] == source["document_id"]
                        for chunk in chunks
                    )
                )
                self.assertTrue(
                    all(chunk.metadata["content_hash"] for chunk in chunks)
                )
                self.assertTrue(
                    all(chunk.metadata["section_title"] for chunk in chunks)
                )

                repeated_chunks = chunk_documents(
                    documents,
                    document_hash=document_hash,
                    chunk_size=1200,
                    chunk_overlap=200,
                )
                self.assertEqual(
                    [chunk.metadata["chunk_id"] for chunk in chunks],
                    [chunk.metadata["chunk_id"] for chunk in repeated_chunks],
                )

            all_chunks.extend(chunks)

        self.assertEqual(len(manifest["documents"]), 6)
        self.assertGreaterEqual(len(all_chunks), 12)
        self.assertEqual(
            len({chunk.metadata["chunk_id"] for chunk in all_chunks}),
            len(all_chunks),
        )


if __name__ == "__main__":
    unittest.main()
