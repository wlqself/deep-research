import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from deep_research.config import settings
from deep_research.memory.service import MemoryService
from deep_research.persistence.memory import sqlite_memory_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0, 0.0]


def make_entry_data(
    entry_id: str,
    *,
    kind: str,
    memory_key: str,
) -> dict[str, object]:
    return {
        "id": entry_id,
        "memory_key": memory_key,
        "scope": "project" if kind == "project" else "global",
        "kind": kind,
        "title": f"Title {entry_id}",
        "summary": f"Summary {entry_id}",
        "content": f"Content {entry_id}",
        "keywords": [entry_id],
        "source_type": "explicit_user",
        "source_thread_id": "thread-1",
        "created_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T00:00:00Z",
        "status": "active",
    }


class MemoryRebuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_rebuilds_user_index_and_pages_from_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"
            projection_path = Path(temp_dir) / "projection"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "memory_projection_path", str(projection_path)
            ), patch.object(settings, "embedding_dimensions", 4), patch.object(
                settings, "memory_page_size", 2
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")

                    await store.aput(
                        ("memories", "local-user", "user"),
                        "user-1",
                        make_entry_data(
                            "user-1",
                            kind="user",
                            memory_key="user.preference.one",
                        ),
                        index=["title", "summary", "keywords"],
                    )

                    for index in range(1, 4):
                        await store.aput(
                            ("memories", "local-user", "project"),
                            f"project-{index}",
                            make_entry_data(
                                f"project-{index}",
                                kind="project",
                                memory_key=f"project.item.{index}",
                            ),
                            index=["title", "summary", "keywords"],
                        )

                    await service.rebuild_projections()

                    user_markdown = (
                        projection_path / "memories" / "USER.md"
                    ).read_text(encoding="utf-8")
                    project_index = (
                        projection_path / "memories" / "project" / "index.md"
                    ).read_text(encoding="utf-8")

                    self.assertIn("user.preference.one", user_markdown)
                    self.assertNotIn("project.item.1", user_markdown)
                    self.assertIn("project-1", project_index)
                    self.assertIn("page-001.md", project_index)
                    self.assertIn("page-002.md", project_index)
                    self.assertTrue(
                        (
                            projection_path
                            / "memories"
                            / "project"
                            / "page-001.md"
                        ).exists()
                    )
                    self.assertTrue(
                        (
                            projection_path
                            / "memories"
                            / "project"
                            / "page-002.md"
                        ).exists()
                    )

    async def test_rebuild_removes_stale_pages_after_entries_are_deleted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"
            projection_path = Path(temp_dir) / "projection"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "memory_projection_path", str(projection_path)
            ), patch.object(settings, "embedding_dimensions", 4), patch.object(
                settings, "memory_page_size", 2
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")

                    for index in range(1, 4):
                        await store.aput(
                            ("memories", "local-user", "project"),
                            f"project-{index}",
                            make_entry_data(
                                f"project-{index}",
                                kind="project",
                                memory_key=f"project.item.{index}",
                            ),
                            index=["title", "summary", "keywords"],
                        )

                    await service.rebuild_projections()
                    stale_page = (
                        projection_path
                        / "memories"
                        / "project"
                        / "page-002.md"
                    )
                    self.assertTrue(stale_page.exists())

                    await service.delete_entry("project", "project-3")
                    await service.rebuild_projections()

                    self.assertFalse(stale_page.exists())

    async def test_projection_failure_does_not_remove_store_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"
            projection_path = Path(temp_dir) / "projection"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "memory_projection_path", str(projection_path)
            ), patch.object(settings, "embedding_dimensions", 4):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    service = MemoryService(store, user_id="local-user")
                    await store.aput(
                        ("memories", "local-user", "project"),
                        "project-1",
                        make_entry_data(
                            "project-1",
                            kind="project",
                            memory_key="project.item.one",
                        ),
                        index=["title", "summary", "keywords"],
                    )

                    with patch(
                        "deep_research.memory.service.write_text_atomic",
                        side_effect=OSError("projection write failed"),
                    ):
                        with self.assertRaisesRegex(
                            OSError,
                            "projection write failed",
                        ):
                            await service.rebuild_projections()

                    self.assertIsNotNone(
                        await service.get_entry("project", "project-1")
                    )


if __name__ == "__main__":
    unittest.main()
