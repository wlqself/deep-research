import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from deep_research.config import settings
from deep_research.memory.projection import summary_archive_path
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
    archive_id: str,
    thread_id: str,
    summary_hash: str = "hash-1",
) -> SummaryArchive:
    return SummaryArchive.model_validate(
        {
            "id": archive_id,
            "thread_id": thread_id,
            "summary_hash": summary_hash,
            "original_summary": f"Original {archive_id}.",
            "retrieval_summary": f"Retrieval {archive_id}.",
            "topics": ["memory"],
            "version": 1,
            "archived_at": datetime.now(timezone.utc),
        }
    )


class SummaryArchiveRebuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_rebuild_writes_distinct_files_for_same_hash_across_threads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"
            projection_path = Path(temp_dir) / "projection"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "memory_projection_path", str(projection_path)
            ), patch.object(settings, "embedding_dimensions", 4):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    first = make_archive(
                        archive_id="archive-1",
                        thread_id="thread-1",
                    )
                    second = make_archive(
                        archive_id="archive-2",
                        thread_id="thread-2",
                    )

                    self.assertTrue(
                        await service.put_summary_archive(first)
                    )
                    self.assertTrue(
                        await service.put_summary_archive(second)
                    )

                    await service.rebuild_projections()

                    first_path = summary_archive_path(first)
                    second_path = summary_archive_path(second)

                    self.assertNotEqual(first_path, second_path)
                    self.assertIn(
                        "Retrieval archive-1.",
                        first_path.read_text(encoding="utf-8"),
                    )
                    self.assertIn(
                        "Retrieval archive-2.",
                        second_path.read_text(encoding="utf-8"),
                    )

    async def test_rebuild_removes_archive_projection_deleted_from_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"
            projection_path = Path(temp_dir) / "projection"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "memory_projection_path", str(projection_path)
            ), patch.object(settings, "embedding_dimensions", 4):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    archive = make_archive(
                        archive_id="archive-1",
                        thread_id="thread-1",
                    )

                    await service.put_summary_archive(archive)
                    await service.rebuild_projections()
                    archive_path = summary_archive_path(archive)
                    self.assertTrue(archive_path.exists())

                    await store.adelete(
                        (
                            "memory-internal",
                            "local-user",
                            "summary-archive",
                            "thread-1",
                        ),
                        "hash-1",
                    )
                    await service.rebuild_projections()

                    self.assertFalse(archive_path.exists())


if __name__ == "__main__":
    unittest.main()
