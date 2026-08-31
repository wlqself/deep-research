import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.embeddings import Embeddings

from deep_research.config import settings
from deep_research.memory.service import MemoryService
from deep_research.memory.type import SummaryArchive
from deep_research.persistence.memory import sqlite_memory_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0, 0.0]


def make_archive(
    *,
    archive_id: str = "archive-1",
    thread_id: str = "thread-1",
    summary_hash: str = "hash-1",
) -> SummaryArchive:
    return SummaryArchive.model_validate(
        {
            "id": archive_id,
            "thread_id": thread_id,
            "summary_hash": summary_hash,
            "original_summary": "Original summary.",
            "retrieval_summary": "The user prefers concise answers.",
            "topics": ["user preference"],
            "memory_keys": ["user:global:user.answer_style.conciseness"],
            "recallable": True,
            "version": 1,
            "archived_at": datetime.now(timezone.utc),
        }
    )


class SummaryArchiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_thread_and_hash_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    archive = make_archive()

                    self.assertTrue(
                        await service.put_summary_archive(archive)
                    )
                    revision_after_first = service.recall_revision

                    self.assertFalse(
                        await service.put_summary_archive(
                            make_archive(archive_id="different-id")
                        )
                    )
                    self.assertEqual(
                        service.recall_revision,
                        revision_after_first,
                    )

                    loaded = await service.get_summary_archive(
                        "thread-1",
                        "hash-1",
                    )
                    self.assertIsNotNone(loaded)
                    self.assertEqual(loaded.id, "archive-1")

    async def test_thread_id_is_part_of_archive_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")

                    self.assertTrue(
                        await service.put_summary_archive(make_archive())
                    )
                    self.assertTrue(
                        await service.put_summary_archive(
                            make_archive(
                                archive_id="archive-2",
                                thread_id="thread-2",
                            )
                        )
                    )

                    first = await service.get_summary_archive(
                        "thread-1",
                        "hash-1",
                    )
                    second = await service.get_summary_archive(
                        "thread-2",
                        "hash-1",
                    )

                    self.assertEqual(first.id, "archive-1")
                    self.assertEqual(second.id, "archive-2")

    async def test_invalid_thread_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")

                    with self.assertRaises(ValueError):
                        await service.put_summary_archive(
                            make_archive(thread_id="   ")
                        )

                    with self.assertRaises(ValueError):
                        await service.put_summary_archive(
                            make_archive(thread_id="thread.with.dot")
                        )

    async def test_new_archive_rebuilds_but_duplicate_does_not(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    archive = make_archive()

                    with patch.object(
                        service,
                        "rebuild_projections",
                        new_callable=AsyncMock,
                    ) as rebuild:
                        self.assertTrue(
                            await service.put_summary_archive(archive)
                        )
                        self.assertFalse(
                            await service.put_summary_archive(archive)
                        )

                    rebuild.assert_awaited_once()

    async def test_archive_survives_projection_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")

                    with patch.object(
                        service,
                        "rebuild_projections",
                        new_callable=AsyncMock,
                        side_effect=OSError("projection failed"),
                    ):
                        self.assertTrue(
                            await service.put_summary_archive(make_archive())
                        )

                    self.assertIsNotNone(
                        await service.get_summary_archive(
                            "thread-1",
                            "hash-1",
                        )
                    )


if __name__ == "__main__":
    unittest.main()
