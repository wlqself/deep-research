import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from deep_research.config import settings
from deep_research.memory.decision import decide_candidate
from deep_research.memory.service import MemoryService
from deep_research.memory.type import MemoryCandidate, MemoryEntry
from deep_research.persistence.memory import sqlite_memory_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0, 0.0]


def make_candidate(
    *,
    content: str = "Prefer concise answers.",
) -> MemoryCandidate:
    return MemoryCandidate.model_validate(
        {
            "kind": "user",
            "memory_key": "user.answer_style.conciseness",
            "scope": "global",
            "title": "Answer style",
            "summary": "The user prefers concise answers.",
            "content": content,
            "keywords": ["concise", "answer"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
        }
    )


def make_entry(entry_id: str = "memory-1") -> MemoryEntry:
    return MemoryEntry.model_validate(
        {
            "id": entry_id,
            "memory_key": "user.answer_style.conciseness",
            "scope": "global",
            "kind": "user",
            "title": "Answer style",
            "summary": "The user prefers concise answers.",
            "content": "Prefer concise answers.",
            "keywords": ["concise", "answer"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
            "created_at": "2026-08-28T00:00:00Z",
            "updated_at": "2026-08-28T00:00:00Z",
            "status": "active",
        }
    )


class MemoryApplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_writes_active_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    candidate = make_candidate()
                    decision = decide_candidate(
                        candidate,
                        [],
                        intent="automatic",
                    )

                    created = await service.apply_decision(
                        candidate,
                        decision,
                    )

                    self.assertIsNotNone(created)
                    self.assertEqual(created.status, "active")
                    self.assertEqual(
                        created.memory_key,
                        candidate.memory_key,
                    )
                    self.assertIsNotNone(
                        await service.get_entry("user", created.id)
                    )

    async def test_no_op_does_not_create_a_new_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    existing = make_entry()
                    await service.put_entry(existing)
                    candidate = make_candidate()
                    decision = decide_candidate(
                        candidate,
                        [existing],
                        intent="explicit_remember",
                    )

                    result = await service.apply_decision(
                        candidate,
                        decision,
                        existing=existing,
                    )

                    self.assertEqual(result.id, existing.id)
                    entries = await service.list_entries(
                        "user",
                        limit=10,
                    )
                    self.assertEqual(
                        [entry.id for entry in entries],
                        [existing.id],
                    )

    async def test_ignore_does_not_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    candidate = MemoryCandidate.model_validate(
                        {
                            **make_candidate().model_dump(mode="python"),
                            "kind": "ignore",
                        }
                    )
                    decision = decide_candidate(
                        candidate,
                        [],
                        intent="automatic",
                    )

                    result = await service.apply_decision(
                        candidate,
                        decision,
                    )

                    self.assertIsNone(result)
                    self.assertEqual(
                        await store.alist_namespaces(
                            prefix=("memories", "local-user"),
                        ),
                        [],
                    )

    async def test_update_supersedes_old_entry_and_creates_new_active_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    existing = make_entry()
                    await service.put_entry(existing)
                    candidate = make_candidate(
                        content="Prefer detailed answers."
                    )
                    decision = decide_candidate(
                        candidate,
                        [existing],
                        intent="explicit_update",
                    )

                    updated = await service.apply_decision(
                        candidate,
                        decision,
                        existing=existing,
                    )

                    self.assertIsNotNone(updated)
                    self.assertNotEqual(updated.id, existing.id)
                    self.assertEqual(updated.status, "active")
                    self.assertEqual(
                        updated.memory_key,
                        existing.memory_key,
                    )

                    old_after_update = await service.get_entry(
                        "user",
                        existing.id,
                    )
                    new_after_update = await service.get_entry(
                        "user",
                        updated.id,
                    )

                    self.assertEqual(
                        old_after_update.status,
                        "superseded",
                    )
                    self.assertEqual(
                        new_after_update.content,
                        "Prefer detailed answers.",
                    )
                    self.assertEqual(
                        new_after_update.status,
                        "active",
                    )


if __name__ == "__main__":
    unittest.main()
