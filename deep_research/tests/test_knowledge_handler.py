import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import deep_research.agent as agent_module
import deep_research.main as main_module
from deep_research.config import settings


class FakeAgent:
    pass


class FakeRagService:
    def __init__(self) -> None:
        self.closed = False
        self.indexed_document_ids: list[str] = []
        self.pending_document_ids: list[str] = []
        self.marked_document_ids: list[str] = []
        self.deleted_document_ids: list[str] = []
        self.delete_error: Exception | None = None

    def close(self) -> None:
        self.closed = True

    def index_chunks(self, chunks) -> int:
        if chunks:
            self.indexed_document_ids.append(
                chunks[0].metadata["document_id"]
            )
        return len(chunks)

    def mark_document_chunks_pending(
        self,
        document_id: str,
    ) -> None:
        self.pending_document_ids.append(document_id)

    def mark_document_chunks_indexed(
        self,
        document_id: str,
    ) -> None:
        self.marked_document_ids.append(document_id)

    def delete_document_chunks(
        self,
        document_id: str,
    ) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted_document_ids.append(document_id)


class KnowledgeHandlerTests(unittest.TestCase):
    def test_upload_indexes_document_and_hides_safe_filename(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        response = client.post(
                            "/knowledge/documents",
                            files={
                                "file": (
                                    "guide.md",
                                    "# 安装\n\n安装依赖。".encode(
                                        "utf-8"
                                    ),
                                    "text/markdown",
                                )
                            },
                        )

                self.assertEqual(response.status_code, 201)
                payload = response.json()
                self.assertEqual(payload["filename"], "guide.md")
                self.assertEqual(payload["status"], "indexed")
                self.assertGreater(payload["chunk_count"], 0)
                self.assertNotIn("safe_filename", payload)
                self.assertNotIn("documents_path", payload)
                self.assertEqual(
                    len(fake_rag.marked_document_ids),
                    1,
                )
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_duplicate_upload_returns_conflict(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )
            file_data = (
                "same.md",
                b"same content",
                "text/markdown",
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        first = client.post(
                            "/knowledge/documents",
                            files={"file": file_data},
                        )
                        second = client.post(
                            "/knowledge/documents",
                            files={"file": file_data},
                        )

                self.assertEqual(first.status_code, 201)
                self.assertEqual(second.status_code, 409)
                self.assertEqual(
                    second.json()["detail"]["error"],
                    "duplicate_document",
                )
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_invalid_upload_type_returns_bad_request(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        response = client.post(
                            "/knowledge/documents",
                            files={
                                "file": (
                                    "guide.pdf",
                                    b"not a pdf",
                                    "text/plain",
                                )
                            },
                        )

                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    fake_rag.indexed_document_ids,
                    [],
                )
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_list_documents_supports_status_filter(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        empty = client.get(
                            "/knowledge/documents"
                        )
                        self.assertEqual(empty.status_code, 200)
                        self.assertEqual(empty.json(), [])

                        uploaded = client.post(
                            "/knowledge/documents",
                            files={
                                "file": (
                                    "guide.md",
                                    b"# Guide\n\nContent",
                                    "text/markdown",
                                )
                            },
                        )
                        self.assertEqual(uploaded.status_code, 201)

                        all_documents = client.get(
                            "/knowledge/documents"
                        )
                        indexed_documents = client.get(
                            "/knowledge/documents",
                            params={"status": "indexed"},
                        )
                        failed_documents = client.get(
                            "/knowledge/documents",
                            params={"status": "failed"},
                        )
                        statistics = client.get(
                            "/knowledge/stats"
                        )

                self.assertEqual(all_documents.status_code, 200)
                self.assertEqual(
                    len(all_documents.json()),
                    1,
                )
                self.assertEqual(
                    len(indexed_documents.json()),
                    1,
                )
                self.assertEqual(
                    indexed_documents.json()[0]["status"],
                    "indexed",
                )
                self.assertEqual(
                    failed_documents.json(),
                    [],
                )
                self.assertEqual(statistics.status_code, 200)
                self.assertEqual(
                    statistics.json()["document_count"],
                    1,
                )
                self.assertGreater(
                    statistics.json()["chunk_count"],
                    0,
                )
                self.assertEqual(
                    statistics.json()["indexed_count"],
                    1,
                )
                self.assertEqual(
                    statistics.json()["failed_count"],
                    0,
                )
                self.assertNotIn(
                    "safe_filename",
                    all_documents.json()[0],
                )
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_get_and_download_document_are_safe_and_threadless(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        uploaded = client.post(
                            "/knowledge/documents",
                            files={
                                "file": (
                                    "guide.md",
                                    b"# Guide\n\nContent",
                                    "text/markdown",
                                )
                            },
                        )
                        self.assertEqual(uploaded.status_code, 201)
                        document_id = uploaded.json()["document_id"]

                        detail = client.get(
                            f"/knowledge/documents/{document_id}"
                        )
                        download = client.get(
                            f"/knowledge/documents/"
                            f"{document_id}/download"
                        )
                        missing = client.get(
                            "/knowledge/documents/missing"
                        )

                        registry = (
                            main_module.app.state.document_registry
                        )
                        registry.update_document_status(
                            document_id,
                            status="deleted",
                            updated_at="2026-08-26T00:00:00Z",
                        )

                        deleted_detail = client.get(
                            f"/knowledge/documents/{document_id}"
                        )
                        deleted_download = client.get(
                            f"/knowledge/documents/"
                            f"{document_id}/download"
                        )

                self.assertEqual(detail.status_code, 200)
                self.assertEqual(
                    detail.json()["document_id"],
                    document_id,
                )
                self.assertNotIn(
                    "safe_filename",
                    detail.json(),
                )
                self.assertEqual(download.status_code, 200)
                self.assertEqual(
                    download.content,
                    b"# Guide\n\nContent",
                )
                self.assertTrue(
                    download.headers["content-type"].startswith(
                        "text/markdown"
                    )
                )
                self.assertIn(
                    "guide.md",
                    download.headers["content-disposition"],
                )
                self.assertEqual(missing.status_code, 404)
                self.assertEqual(deleted_detail.status_code, 404)
                self.assertEqual(deleted_download.status_code, 404)
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_delete_document_removes_file_and_is_idempotent(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        uploaded = client.post(
                            "/knowledge/documents",
                            files={
                                "file": (
                                    "guide.md",
                                    b"# Guide\n\nContent",
                                    "text/markdown",
                                )
                            },
                        )
                        document_id = uploaded.json()["document_id"]
                        delete_response = client.delete(
                            f"/knowledge/documents/{document_id}"
                        )
                        second_delete = client.delete(
                            f"/knowledge/documents/{document_id}"
                        )
                        missing = client.delete(
                            "/knowledge/documents/missing"
                        )

                        registry = (
                            main_module.app.state.document_registry
                        )
                        saved = registry.get_document(document_id)

                self.assertEqual(delete_response.status_code, 200)
                self.assertEqual(
                    delete_response.json()["status"],
                    "deleted",
                )
                self.assertEqual(second_delete.status_code, 404)
                self.assertEqual(missing.status_code, 404)
                self.assertEqual(
                    fake_rag.deleted_document_ids,
                    [document_id],
                )
                self.assertIsNone(saved)
                connection = sqlite3.connect(
                    str(registry.db_path)
                )
                try:
                    audit = connection.execute(
                        """
                        SELECT event_type, status_after
                        FROM document_audit
                        WHERE document_id = ?
                        """,
                        (document_id,),
                    ).fetchone()
                finally:
                    connection.close()

                self.assertEqual(
                    audit,
                    ("deleted", "deleted"),
                )
                self.assertEqual(
                    [
                        path
                        for path in (
                            root / "documents"
                        ).rglob("*")
                        if path.is_file()
                    ],
                    [],
                )
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_delete_failure_leaves_deleting_status(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        uploaded = client.post(
                            "/knowledge/documents",
                            files={
                                "file": (
                                    "guide.md",
                                    b"# Guide\n\nContent",
                                    "text/markdown",
                                )
                            },
                        )
                        document_id = uploaded.json()["document_id"]
                        fake_rag.delete_error = RuntimeError(
                            "qdrant unavailable"
                        )

                        response = client.delete(
                            f"/knowledge/documents/{document_id}"
                        )
                        registry = (
                            main_module.app.state.document_registry
                        )
                        saved = registry.get_document(document_id)

                self.assertEqual(response.status_code, 500)
                self.assertEqual(saved["status"], "deleting")
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_reindex_document_succeeds(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        uploaded = client.post(
                            "/knowledge/documents",
                            files={
                                "file": (
                                    "guide.md",
                                    b"# Guide\n\nContent",
                                    "text/markdown",
                                )
                            },
                        )
                        document_id = uploaded.json()["document_id"]

                        response = client.post(
                            f"/knowledge/documents/"
                            f"{document_id}/reindex"
                        )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json()["status"],
                    "indexed",
                )
                self.assertEqual(
                    fake_rag.pending_document_ids,
                    [document_id],
                )
                self.assertEqual(
                    fake_rag.deleted_document_ids,
                    [document_id],
                )
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_reindex_unknown_document_returns_not_found(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        response = client.post(
                            "/knowledge/documents/unknown/reindex"
                        )

                self.assertEqual(response.status_code, 404)
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_reindex_deleted_document_returns_not_found(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        uploaded = client.post(
                            "/knowledge/documents",
                            files={
                                "file": (
                                    "guide.md",
                                    b"# Guide\n\nContent",
                                    "text/markdown",
                                )
                            },
                        )
                        document_id = uploaded.json()["document_id"]
                        registry = (
                            main_module.app.state.document_registry
                        )
                        registry.update_document_status(
                            document_id,
                            status="deleted",
                            updated_at="2026-08-26T00:00:00Z",
                        )

                        response = client.post(
                            f"/knowledge/documents/"
                            f"{document_id}/reindex"
                        )

                self.assertEqual(response.status_code, 404)
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent

    def test_reindex_missing_file_returns_not_found(self):
        original_checkpoint_path = settings.checkpoint_db_path
        original_registry_path = settings.rag_registry_db_path
        original_documents_path = settings.rag_documents_path
        original_agent = agent_module.agent
        fake_rag = FakeRagService()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings.checkpoint_db_path = str(
                root / "checkpoints.sqlite"
            )
            settings.rag_registry_db_path = str(
                root / "registry.sqlite"
            )
            settings.rag_documents_path = str(
                root / "documents"
            )

            try:
                with patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=fake_rag,
                ), patch.object(
                    agent_module,
                    "build_agent",
                    return_value=FakeAgent(),
                ):
                    with TestClient(main_module.app) as client:
                        uploaded = client.post(
                            "/knowledge/documents",
                            files={
                                "file": (
                                    "guide.md",
                                    b"# Guide\n\nContent",
                                    "text/markdown",
                                )
                            },
                        )
                        document_id = uploaded.json()["document_id"]
                        document_files = [
                            path
                            for path in (
                                root / "documents"
                            ).rglob("*")
                            if path.is_file()
                        ]
                        self.assertEqual(len(document_files), 1)
                        document_files[0].unlink()

                        response = client.post(
                            f"/knowledge/documents/"
                            f"{document_id}/reindex"
                        )

                self.assertEqual(response.status_code, 404)
            finally:
                settings.checkpoint_db_path = (
                    original_checkpoint_path
                )
                settings.rag_registry_db_path = (
                    original_registry_path
                )
                settings.rag_documents_path = (
                    original_documents_path
                )
                agent_module.agent = original_agent


if __name__ == "__main__":
    unittest.main()
