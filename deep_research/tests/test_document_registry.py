import sqlite3
import tempfile
import unittest
from pathlib import Path

from deep_research.rag.registry import (
    DocumentRegistry,
    DocumentRecord,
)


def make_record(
    *,
    document_id: str = "doc-1",
    sha256: str = "hash-1",
) -> DocumentRecord:
    return {
        "document_id": document_id,
        "collection_id": "deep_research_documents",
        "filename": "guide.md",
        "safe_filename": "doc-1.md",
        "mime_type": "text/markdown",
        "size_bytes": 128,
        "sha256": sha256,
        "status": "pending",
        "page_count": 0,
        "chunk_count": 0,
        "error": "",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


class DocumentRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DocumentRegistry(
            str(Path(self.temp_dir.name) / "registry.sqlite")
        )
        self.registry.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_create_and_get_document(self):
        record = make_record()

        self.registry.create_document(record)

        self.assertEqual(
            self.registry.get_document("doc-1"),
            record,
        )

    def test_get_missing_document_returns_none(self):
        self.assertIsNone(
            self.registry.get_document("missing")
        )

    def test_get_document_by_sha256(self):
        record = make_record()
        self.registry.create_document(record)

        self.assertEqual(
            self.registry.get_document_by_sha256("hash-1"),
            record,
        )
        self.assertIsNone(
            self.registry.get_document_by_sha256("missing-hash")
        )

    def test_duplicate_sha256_is_rejected(self):
        self.registry.create_document(make_record())

        with self.assertRaises(sqlite3.IntegrityError):
            self.registry.create_document(
                make_record(
                    document_id="doc-2",
                    sha256="hash-1",
                )
            )

    def test_document_status_and_counts_can_be_updated(self):
        self.registry.create_document(make_record())

        self.registry.update_document_status(
            "doc-1",
            status="processing",
            updated_at="2026-01-01T00:01:00Z",
        )
        processing = self.registry.get_document("doc-1")

        self.assertIsNotNone(processing)
        self.assertEqual(processing["status"], "processing")
        self.assertEqual(processing["page_count"], 0)

        self.registry.update_document_status(
            "doc-1",
            status="indexed",
            page_count=5,
            chunk_count=18,
            updated_at="2026-01-01T00:02:00Z",
        )
        indexed = self.registry.get_document("doc-1")

        self.assertIsNotNone(indexed)
        self.assertEqual(indexed["status"], "indexed")
        self.assertEqual(indexed["page_count"], 5)
        self.assertEqual(indexed["chunk_count"], 18)
        self.assertEqual(indexed["error"], "")

    def test_failed_status_stores_safe_error_code(self):
        self.registry.create_document(make_record())

        self.registry.update_document_status(
            "doc-1",
            status="failed",
            error="parse_failed",
            updated_at="2026-01-01T00:03:00Z",
        )
        failed = self.registry.get_document("doc-1")

        self.assertIsNotNone(failed)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"], "parse_failed")

    def test_get_document_ignores_deleted_record(self):
        self.registry.create_document(make_record())
        self.registry.update_document_status(
            "doc-1",
            status="deleted",
            updated_at="2026-01-01T00:02:00Z",
        )

        self.assertIsNone(
            self.registry.get_document("doc-1")
        )

    def test_updating_missing_document_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.registry.update_document_status(
                "missing",
                status="processing",
                updated_at="2026-01-01T00:01:00Z",
            )

    def test_archive_and_delete_removes_current_record_and_keeps_audit(self):
        record = make_record()
        self.registry.create_document(record)

        self.registry.archive_and_delete_document(
            record,
            event_type="deleted",
            created_at="2026-01-01T00:02:00Z",
        )

        self.assertIsNone(
            self.registry.get_document("doc-1")
        )

        connection = sqlite3.connect(
            str(self.registry.db_path)
        )
        try:
            audit = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    sha256,
                    event_type,
                    status_before,
                    status_after,
                    error
                FROM document_audit
                WHERE document_id = ?
                """,
                ("doc-1",),
            ).fetchone()
        finally:
            connection.close()

        self.assertEqual(
            audit,
            (
                "doc-1",
                "guide.md",
                "hash-1",
                "deleted",
                "pending",
                "deleted",
                "",
            ),
        )

    def test_deleted_sha256_can_be_registered_again(self):
        record = make_record()
        self.registry.create_document(record)
        self.registry.archive_and_delete_document(
            record,
            event_type="deleted",
            created_at="2026-01-01T00:02:00Z",
        )

        replacement = make_record(
            document_id="doc-2",
            sha256="hash-1",
        )
        self.registry.create_document(replacement)

        self.assertEqual(
            self.registry.get_document("doc-2"),
            replacement,
        )

    def test_initialize_migrates_legacy_deleted_record_to_audit(self):
        record = make_record()
        self.registry.create_document(record)
        self.registry.update_document_status(
            "doc-1",
            status="deleted",
            updated_at="2026-01-01T00:02:00Z",
        )

        reopened = DocumentRegistry(str(self.registry.db_path))
        reopened.initialize()

        self.assertIsNone(reopened.get_document("doc-1"))
        self.assertEqual(reopened.list_documents(), [])
        self.assertEqual(
            reopened.get_statistics()["document_count"],
            0,
        )

        connection = sqlite3.connect(str(reopened.db_path))
        try:
            audit = connection.execute(
                """
                SELECT event_type, status_before, status_after
                FROM document_audit
                WHERE document_id = ?
                """,
                ("doc-1",),
            ).fetchone()
        finally:
            connection.close()

        self.assertEqual(
            audit,
            ("legacy_deleted_cleanup", "deleted", "deleted"),
        )

    def test_list_documents_returns_empty_list_for_empty_registry(self):
        self.assertEqual(
            self.registry.list_documents(),
            [],
        )

    def test_list_documents_filters_status_and_orders_by_update_time(self):
        first = make_record()
        second = make_record(
            document_id="doc-2",
            sha256="hash-2",
        )
        self.registry.create_document(first)
        self.registry.create_document(second)

        self.registry.update_document_status(
            "doc-1",
            status="processing",
            updated_at="2026-01-01T00:01:00Z",
        )
        self.registry.update_document_status(
            "doc-2",
            status="indexed",
            updated_at="2026-01-01T00:02:00Z",
        )

        documents = self.registry.list_documents()
        self.assertEqual(
            [document["document_id"] for document in documents],
            ["doc-2", "doc-1"],
        )

        indexed = self.registry.list_documents(
            status="indexed"
        )
        self.assertEqual(
            [document["document_id"] for document in indexed],
            ["doc-2"],
        )

    def test_list_documents_binds_status_as_sql_parameter(self):
        self.registry.create_document(make_record())

        self.assertEqual(
            self.registry.list_documents(
                status="pending' OR 1=1 --"  # type: ignore[arg-type]
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
