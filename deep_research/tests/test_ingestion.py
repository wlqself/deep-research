import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import deep_research.rag.ingestion as ingestion_module
from deep_research.rag.ingestion import (
    DuplicateDocumentError,
    index_document,
    ingest_upload,
    prepare_document_chunks,
    reindex_document,
)
from deep_research.rag.registry import (
    DocumentRecord,
    DocumentRegistry,
)


class FakeUpload:
    def __init__(
        self,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self.content = content
        self.offset = 0

    async def read(self, size: int) -> bytes:
        chunk = self.content[
            self.offset:self.offset + size
        ]
        self.offset += len(chunk)
        return chunk


class FakeRagService:
    def __init__(
        self,
        *,
        written_count: int | None = None,
        mark_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.written_count = written_count
        self.mark_error = mark_error
        self.delete_error = delete_error
        self.index_calls: list[int] = []
        self.pending_calls: list[str] = []
        self.mark_calls: list[str] = []
        self.delete_calls: list[str] = []

    def mark_document_chunks_pending(
        self,
        document_id: str,
    ) -> None:
        self.pending_calls.append(document_id)

    def index_chunks(self, chunks) -> int:
        self.index_calls.append(len(chunks))
        if self.written_count is not None:
            return self.written_count
        return len(chunks)

    def mark_document_chunks_indexed(
        self,
        document_id: str,
    ) -> None:
        self.mark_calls.append(document_id)
        if self.mark_error is not None:
            raise self.mark_error

    def delete_document_chunks(
        self,
        document_id: str,
    ) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.delete_calls.append(document_id)


class IngestionTests(
    unittest.IsolatedAsyncioTestCase
):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.documents_root = self.root / "documents"
        self.temp_root = self.root / "tmp"
        self.registry = DocumentRegistry(
            str(self.root / "registry.sqlite")
        )
        self.registry.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def ingest(
        self,
        content: bytes,
        *,
        filename: str = "guide.md",
        content_type: str = "text/markdown",
        max_bytes: int = 1000,
    ):
        return await ingest_upload(
            FakeUpload(
                filename,
                content_type,
                content,
            ),
            registry=self.registry,
            documents_root=str(self.documents_root),
            temp_dir=str(self.temp_root),
            collection_id="deep_research_documents",
            max_bytes=max_bytes,
        )

    async def test_success_creates_pending_record_and_moves_file(self):
        content = b"knowledge base content"

        record = await self.ingest(content)

        self.assertEqual(record["status"], "pending")
        self.assertEqual(record["filename"], "guide.md")
        self.assertEqual(record["mime_type"], "text/markdown")

        saved = self.registry.get_document(
            record["document_id"]
        )
        self.assertEqual(saved, record)

        final_path = (
            self.documents_root
            / record["document_id"]
            / record["safe_filename"]
        )
        self.assertEqual(final_path.read_bytes(), content)
        self.assertEqual(list(self.temp_root.iterdir()), [])

    async def test_duplicate_sha256_is_rejected_and_temp_is_cleaned(self):
        content = b"same content"
        first = await self.ingest(content)

        with self.assertRaises(DuplicateDocumentError) as context:
            await self.ingest(content)

        self.assertEqual(
            context.exception.existing_document_id,
            first["document_id"],
        )
        self.assertEqual(
            len(self.registry.list_documents()),
            1,
        )
        self.assertEqual(list(self.temp_root.iterdir()), [])

    async def test_size_limit_rejects_upload_without_record(self):
        with self.assertRaises(ValueError):
            await self.ingest(
                b"123456",
                max_bytes=5,
            )

        self.assertEqual(
            self.registry.list_documents(),
            [],
        )
        self.assertEqual(list(self.temp_root.iterdir()), [])

    async def test_finalize_failure_marks_record_failed(self):
        with patch.object(
            ingestion_module,
            "finalize_document_file",
            side_effect=OSError("simulated move failure"),
        ):
            with self.assertRaises(OSError):
                await self.ingest(b"content")

        records = self.registry.list_documents()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(records[0]["error"], "storage_failed")
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def make_record(
        self,
        *,
        filename: str,
        mime_type: str,
    ) -> DocumentRecord:
        return {
            "document_id": "doc-1",
            "collection_id": "deep_research_documents",
            "filename": filename,
            "safe_filename": f"doc-1{Path(filename).suffix}",
            "mime_type": mime_type,
            "size_bytes": 100,
            "sha256": "document-hash",
            "status": "pending",
            "page_count": 0,
            "chunk_count": 0,
            "error": "",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }

    def register_record(
        self,
        record: DocumentRecord,
    ) -> None:
        self.registry.create_document(record)

    async def test_index_document_marks_indexed(self):
        path = self.root / "guide.md"
        path.write_text(
            "# 安装\n\n安装依赖。" * 10,
            encoding="utf-8",
        )
        record = self.make_record(
            filename="guide.md",
            mime_type="text/markdown",
        )
        self.register_record(record)
        rag_service = FakeRagService()

        result = await index_document(
            path=path,
            record=record,
            registry=self.registry,
            rag_service=rag_service,
            chunk_size=30,
            chunk_overlap=5,
        )

        self.assertEqual(result["status"], "indexed")
        self.assertEqual(result["page_count"], 1)
        self.assertGreater(result["chunk_count"], 0)
        self.assertEqual(
            rag_service.mark_calls,
            ["doc-1"],
        )
        self.assertEqual(rag_service.delete_calls, [])
        saved = self.registry.get_document("doc-1")
        self.assertEqual(saved["status"], "indexed")
        self.assertEqual(
            saved["chunk_count"],
            result["chunk_count"],
        )

    async def test_parse_failure_marks_failed_and_cleans_vectors(self):
        path = self.root / "invalid.txt"
        path.write_bytes(b"\xff\xfe\xfd")
        record = self.make_record(
            filename="invalid.txt",
            mime_type="text/plain",
        )
        self.register_record(record)
        rag_service = FakeRagService()

        with self.assertRaises(UnicodeDecodeError):
            await index_document(
                path=path,
                record=record,
                registry=self.registry,
                rag_service=rag_service,
                chunk_size=30,
                chunk_overlap=5,
            )

        saved = self.registry.get_document("doc-1")
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["error"], "indexing_failed")
        self.assertEqual(rag_service.delete_calls, ["doc-1"])

    async def test_count_mismatch_marks_failed(self):
        path = self.root / "guide.txt"
        path.write_text("可检索内容。" * 10, encoding="utf-8")
        record = self.make_record(
            filename="guide.txt",
            mime_type="text/plain",
        )
        self.register_record(record)
        rag_service = FakeRagService(written_count=0)

        with self.assertRaises(RuntimeError):
            await index_document(
                path=path,
                record=record,
                registry=self.registry,
                rag_service=rag_service,
                chunk_size=20,
                chunk_overlap=4,
            )

        saved = self.registry.get_document("doc-1")
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(rag_service.delete_calls, ["doc-1"])

    async def test_qdrant_status_failure_marks_failed(self):
        path = self.root / "guide.md"
        path.write_text("可检索内容。", encoding="utf-8")
        record = self.make_record(
            filename="guide.md",
            mime_type="text/markdown",
        )
        self.register_record(record)
        rag_service = FakeRagService(
            mark_error=RuntimeError("status update failed")
        )

        with self.assertRaises(RuntimeError):
            await index_document(
                path=path,
                record=record,
                registry=self.registry,
                rag_service=rag_service,
                chunk_size=20,
                chunk_overlap=4,
            )

        saved = self.registry.get_document("doc-1")
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(rag_service.delete_calls, ["doc-1"])

    async def test_reindex_replaces_old_chunks_and_marks_indexed(self):
        record = await self.ingest(
            b"# Guide\n\nOriginal content"
        )
        path = (
            self.documents_root
            / record["document_id"]
            / record["safe_filename"]
        )
        rag_service = FakeRagService()

        result = await reindex_document(
            path=path,
            record=record,
            registry=self.registry,
            rag_service=rag_service,
            chunk_size=30,
            chunk_overlap=5,
        )

        self.assertEqual(result["status"], "indexed")
        self.assertEqual(
            rag_service.pending_calls,
            [record["document_id"]],
        )
        self.assertEqual(
            rag_service.delete_calls,
            [record["document_id"]],
        )
        self.assertEqual(
            rag_service.mark_calls,
            [record["document_id"]],
        )
        self.assertTrue(path.exists())

        saved = self.registry.get_document(
            record["document_id"]
        )
        self.assertEqual(saved["status"], "indexed")

    async def test_reindex_delete_failure_marks_failed_and_keeps_file(self):
        record = await self.ingest(
            b"# Guide\n\nOriginal content"
        )
        path = (
            self.documents_root
            / record["document_id"]
            / record["safe_filename"]
        )
        rag_service = FakeRagService(
            delete_error=RuntimeError(
                "qdrant unavailable"
            )
        )

        with self.assertRaises(RuntimeError):
            await reindex_document(
                path=path,
                record=record,
                registry=self.registry,
                rag_service=rag_service,
                chunk_size=30,
                chunk_overlap=5,
            )

        saved = self.registry.get_document(
            record["document_id"]
        )
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["error"], "reindex_failed")
        self.assertTrue(path.exists())

    def test_prepare_markdown_document_chunks(self):
        path = self.root / "guide.md"
        path.write_text(
            "# 安装\n\n" + "安装依赖。" * 20,
            encoding="utf-8",
        )
        record = self.make_record(
            filename="guide.md",
            mime_type="text/markdown",
        )

        chunks, page_count = prepare_document_chunks(
            path,
            record=record,
            chunk_size=20,
            chunk_overlap=4,
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual(page_count, 1)
        self.assertTrue(
            all(
                chunk.metadata["document_id"] == "doc-1"
                for chunk in chunks
            )
        )
        self.assertTrue(
            all(
                chunk.metadata["document_hash"]
                == "document-hash"
                for chunk in chunks
            )
        )

    def test_prepare_txt_document_returns_page_count(self):
        path = self.root / "guide.txt"
        path.write_text(
            "TXT 文档内容。",
            encoding="utf-8",
        )
        record = self.make_record(
            filename="guide.txt",
            mime_type="text/plain",
        )

        chunks, page_count = prepare_document_chunks(
            path,
            record=record,
            chunk_size=100,
            chunk_overlap=10,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(page_count, 1)
        self.assertEqual(
            chunks[0].metadata["filename"],
            "guide.txt",
        )

    def test_prepare_empty_document_returns_no_chunks(self):
        path = self.root / "empty.md"
        path.write_text("", encoding="utf-8")
        record = self.make_record(
            filename="empty.md",
            mime_type="text/markdown",
        )

        chunks, page_count = prepare_document_chunks(
            path,
            record=record,
            chunk_size=100,
            chunk_overlap=10,
        )

        self.assertEqual(chunks, [])
        self.assertEqual(page_count, 0)


if __name__ == "__main__":
    unittest.main()
