import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from deep_research.config import settings
from deep_research.memory.service import MemoryService
from deep_research.memory.type import MemoryEntry
from deep_research.persistence.memory import sqlite_memory_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0, 0.0]


def make_entry(
    entry_id: str,
    *,
    user_id: str = "local-user",
    kind: str = "project",
    status: str = "active",
) -> MemoryEntry:
    return MemoryEntry.model_validate(
        {
            "id": entry_id,
            "memory_key": f"{kind}.{entry_id}",
            "scope": "project" if kind == "project" else "global",
            "kind": kind,
            "title": f"Title {entry_id}",
            "summary": f"Summary for {user_id} {entry_id}",
            "content": f"Content for {entry_id}",
            "keywords": ["project", entry_id],
            "source_type": "explicit_user",
            "source_thread_id": f"thread-{entry_id}",
            "created_at": "2026-08-28T00:00:00Z",
            "updated_at": "2026-08-28T00:00:00Z",
            "status": status,
        }
    )


class MemorySearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_returns_scored_active_entries_with_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    await service.put_entry(make_entry("project-1"))
                    await service.put_entry(make_entry("project-2"))

                    results = await service.find_similar_entries(
                        "project",
                        "project preference",
                        limit=1,
                    )

                    self.assertEqual(len(results), 1)
                    entry, score = results[0]
                    self.assertIsInstance(entry, MemoryEntry)
                    self.assertIn(entry.id, {"project-1", "project-2"})
                    self.assertIsNotNone(score)

    async def test_search_excludes_superseded_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    await service.put_entry(
                        make_entry("active-project", status="active")
                    )
                    await service.put_entry(
                        make_entry("old-project", status="superseded")
                    )

                    results = await service.find_similar_entries(
                        "project",
                        "project",
                        limit=10,
                    )

                    self.assertEqual(
                        {entry.id for entry, _ in results},
                        {"active-project"},
                    )

    async def test_search_does_not_cross_user_or_kind_namespaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    local_service = MemoryService(store, user_id="local-user")
                    other_service = MemoryService(store, user_id="other-user")

                    await local_service.put_entry(make_entry("local-project"))
                    await local_service.put_entry(
                        make_entry("local-user-memory", kind="user")
                    )
                    await other_service.put_entry(
                        make_entry("other-project", user_id="other-user")
                    )

                    results = await local_service.find_similar_entries(
                        "project",
                        "project",
                        limit=10,
                    )

                    self.assertEqual(
                        {entry.id for entry, _ in results},
                        {"local-project"},
                    )

    async def test_search_rejects_empty_query_and_nonpositive_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")

                    with self.assertRaises(ValueError):
                        await service.find_similar_entries(
                            "project",
                            "   ",
                            limit=1,
                        )

                    with self.assertRaises(ValueError):
                        await service.find_similar_entries(
                            "project",
                            "project",
                            limit=0,
                        )


if __name__ == "__main__":
    unittest.main()
