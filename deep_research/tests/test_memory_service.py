import tempfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from deep_research.config import settings
from deep_research.memory.decision import decide_candidate
from deep_research.memory.service import MemoryService
from deep_research.memory.type import (
    MemoryCandidate,
    MemoryEntry,
    MemorySuppression,
    SummaryArchive,
)
from deep_research.persistence.memory import sqlite_memory_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0, 0.0]


def make_entry(
    entry_id: str,
    *,
    kind: str = "project",
) -> MemoryEntry:
    return MemoryEntry.model_validate(
        {
            "id": entry_id,
            "memory_key": f"{kind}.{entry_id}",
            "scope": "project" if kind == "project" else "global",
            "kind": kind,
            "title": f"Title {entry_id}",
            "summary": f"Summary {entry_id}",
            "content": f"Content {entry_id}",
            "keywords": [entry_id],
            "source_type": "explicit_user",
            "source_thread_id": f"thread-{entry_id}",
            "created_at": "2026-08-28T00:00:00Z",
            "updated_at": "2026-08-28T00:00:00Z",
            "status": "active",
        }
    )


def make_candidate(
    *,
    kind: str,
    memory_key: str,
    scope: str,
    content: str,
) -> MemoryCandidate:
    return MemoryCandidate.model_validate(
        {
            "kind": kind,
            "memory_key": memory_key,
            "scope": scope,
            "title": "Remembered fact",
            "summary": content,
            "content": content,
            "keywords": ["memory"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-remember",
        }
    )


def make_archive(
    *,
    memory_key: str,
    summary_hash: str = "archive-hash",
) -> SummaryArchive:
    return SummaryArchive.model_validate(
        {
            "id": f"archive-{summary_hash}",
            "thread_id": "thread-archive",
            "summary_hash": summary_hash,
            "original_summary": "Internal audit summary.",
            "retrieval_summary": "A recallable memory fact.",
            "topics": ["memory"],
            "memory_keys": [memory_key],
            "recallable": True,
            "version": 1,
            "archived_at": datetime.now(timezone.utc),
        }
    )


class MemoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_crud_round_trip_and_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    entry = make_entry("project-1")

                    await service.put_entry(entry)
                    loaded = await service.get_entry("project", "project-1")

                    self.assertIsNotNone(loaded)
                    self.assertIsInstance(loaded, MemoryEntry)
                    self.assertEqual(loaded.id, "project-1")
                    self.assertEqual(loaded.content, "Content project-1")

                    await service.delete_entry("project", "project-1")

                    self.assertIsNone(
                        await service.get_entry("project", "project-1")
                    )

    async def test_list_entries_supports_pagination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")

                    for entry_id in ("project-1", "project-2", "project-3"):
                        await service.put_entry(make_entry(entry_id))

                    first_page = await service.list_entries(
                        "project",
                        limit=2,
                        offset=0,
                    )
                    second_page = await service.list_entries(
                        "project",
                        limit=2,
                        offset=2,
                    )

                    first_ids = {entry.id for entry in first_page}
                    second_ids = {entry.id for entry in second_page}

                    self.assertEqual(len(first_ids), 2)
                    self.assertEqual(len(second_ids), 1)
                    self.assertTrue(first_ids.isdisjoint(second_ids))
                    self.assertEqual(
                        first_ids | second_ids,
                        {"project-1", "project-2", "project-3"},
                    )

    async def test_kind_and_user_namespaces_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    local_service = MemoryService(store, user_id="local-user")
                    other_service = MemoryService(store, user_id="other-user")

                    await local_service.put_entry(make_entry("same-id"))
                    await local_service.put_entry(
                        make_entry("same-id", kind="user")
                    )
                    await other_service.put_entry(make_entry("same-id"))

                    local_project = await local_service.get_entry(
                        "project",
                        "same-id",
                    )
                    local_user = await local_service.get_entry(
                        "user",
                        "same-id",
                    )
                    other_project = await other_service.get_entry(
                        "project",
                        "same-id",
                    )

                    self.assertEqual(local_project.source_thread_id, "thread-same-id")
                    self.assertEqual(local_user.kind, "user")
                    self.assertEqual(other_project.kind, "project")

                    self.assertEqual(
                        await local_service.list_entries(
                            "reference",
                            limit=10,
                        ),
                        [],
                    )

    async def test_forget_deletes_active_entry_and_creates_suppression(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"
            projection_path = Path(temp_dir) / "projection"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "memory_projection_path", str(projection_path)
            ), patch.object(settings, "embedding_dimensions", 4):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    entry = make_entry("user-1", kind="user")
                    await service.put_entry(entry)

                    forgotten = await service.forget_entry(
                        "user",
                        entry.id,
                        source_thread_id="thread-forget",
                    )

                    self.assertEqual(forgotten.id, entry.id)
                    self.assertIsNone(
                        await service.get_entry("user", entry.id)
                    )

                    suppression = await service.find_suppression(
                        entry.kind,
                        entry.memory_key,
                        entry.scope,
                    )
                    self.assertIsNotNone(suppression)
                    self.assertEqual(
                        suppression.forgotten_memory_id,
                        entry.id,
                    )
                    self.assertEqual(
                        suppression.source_thread_id,
                        "thread-forget",
                    )

    async def test_explicit_remember_clears_suppression_in_same_write_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"
            projection_path = Path(temp_dir) / "projection"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "memory_projection_path", str(projection_path)
            ), patch.object(settings, "embedding_dimensions", 4):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    entry = make_entry("user-1", kind="user")
                    await service.put_entry(entry)
                    await service.forget_entry("user", entry.id)

                    candidate = make_candidate(
                        kind="user",
                        memory_key=entry.memory_key,
                        scope=entry.scope,
                        content="The user explicitly remembered this again.",
                    )
                    decision = decide_candidate(
                        candidate,
                        [],
                        intent="explicit_remember",
                    )

                    restored = await service.apply_decision(
                        candidate,
                        decision,
                        clear_suppression=True,
                    )

                    self.assertIsNotNone(restored)
                    self.assertIsNone(
                        await service.find_suppression(
                            entry.kind,
                            entry.memory_key,
                            entry.scope,
                        )
                    )
                    self.assertIsNotNone(
                        await service.get_entry("user", restored.id)
                    )

    async def test_archive_with_suppressed_identity_is_not_searchable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"
            projection_path = Path(temp_dir) / "projection"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "memory_projection_path", str(projection_path)
            ), patch.object(settings, "embedding_dimensions", 4):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    memory_key = "user:global:user.phone"
                    await service.put_summary_archive(
                        make_archive(memory_key=memory_key)
                    )
                    await store.aput(
                        service._suppression_namespace(),
                        memory_key,
                        MemorySuppression(
                            suppression_id=memory_key,
                            kind="user",
                            memory_key="user.phone",
                            scope="global",
                            forgotten_memory_id="memory-phone",
                            source_thread_id="thread-forget",
                            created_at=datetime.now(timezone.utc),
                            reason="explicit_user_forget",
                        ).model_dump(mode="json"),
                        index=[],
                    )

                    self.assertEqual(
                        await service.search_summary_archives(
                            "memory",
                            limit=5,
                        ),
                        [],
                    )

    async def test_clear_all_removes_business_history_archives_and_suppressions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"
            projection_path = Path(temp_dir) / "projection"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "memory_projection_path", str(projection_path)
            ), patch.object(settings, "embedding_dimensions", 4):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    user_entry = make_entry("user-1", kind="user")
                    project_entry = make_entry("project-1", kind="project")
                    await service.put_entry(user_entry)
                    await service.put_entry(project_entry)

                    update_candidate = make_candidate(
                        kind="user",
                        memory_key=user_entry.memory_key,
                        scope=user_entry.scope,
                        content="Updated user fact.",
                    )
                    update_decision = decide_candidate(
                        update_candidate,
                        [user_entry],
                        intent="explicit_update",
                    )
                    await service.apply_decision(
                        update_candidate,
                        update_decision,
                        existing=user_entry,
                    )

                    await service.put_summary_archive(
                        make_archive(
                            memory_key=(
                                f"{project_entry.kind}:"
                                f"{project_entry.scope}:"
                                f"{project_entry.memory_key}"
                            )
                        )
                    )
                    suppression = MemorySuppression(
                        suppression_id=(
                            f"{project_entry.kind}:"
                            f"{project_entry.scope}:"
                            f"{project_entry.memory_key}"
                        ),
                        kind=project_entry.kind,
                        memory_key=project_entry.memory_key,
                        scope=project_entry.scope,
                        forgotten_memory_id=project_entry.id,
                        source_thread_id="thread-forget",
                        created_at=datetime.now(timezone.utc),
                        reason="explicit_user_forget",
                    )
                    await service.put_suppression(suppression)

                    deleted_count = await service.clear_all_memories()

                    self.assertEqual(deleted_count, 3)
                    for kind in ("user", "reference", "project", "feedback"):
                        self.assertEqual(
                            await service.list_entries(kind, limit=10),
                            [],
                        )
                    self.assertEqual(
                        await service.list_summary_archives(limit=10),
                        [],
                    )
                    self.assertIsNone(
                        await service.find_suppression(
                            project_entry.kind,
                            project_entry.memory_key,
                            project_entry.scope,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
