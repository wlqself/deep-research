import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from deep_research.config import settings
from deep_research.persistence.memory import sqlite_memory_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0, 0.0]


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_index_and_cross_connection_persistence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    await store.setup()

                    self.assertEqual(
                        store.index_config["fields"],
                        ["title", "summary", "keywords"],
                    )

                    await store.aput(
                        ("memories", "local-user", "user"),
                        "memory-1",
                        {
                            "title": "Preference",
                            "summary": "User prefers concise answers.",
                            "keywords": ["style"],
                            "source_thread_id": "thread-a",
                        },
                    )

                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as reopened:
                    item = await reopened.aget(
                        ("memories", "local-user", "user"),
                        "memory-1",
                    )

                    self.assertIsNotNone(item)
                    self.assertEqual(item.value["source_thread_id"], "thread-a")

    async def test_thread_ids_share_user_namespace_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    namespace = ("memories", "local-user", "project")

                    await store.aput(
                        namespace,
                        "project-a",
                        {
                            "title": "Project A",
                            "summary": "From thread A.",
                            "keywords": ["a"],
                            "source_thread_id": "thread-a",
                        },
                    )
                    await store.aput(
                        namespace,
                        "project-b",
                        {
                            "title": "Project B",
                            "summary": "From thread B.",
                            "keywords": ["b"],
                            "source_thread_id": "thread-b",
                        },
                    )

                    project_a = await store.aget(namespace, "project-a")
                    project_b = await store.aget(namespace, "project-b")

                    self.assertEqual(project_a.value["source_thread_id"], "thread-a")
                    self.assertEqual(project_b.value["source_thread_id"], "thread-b")

    async def test_namespaces_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite"

            with patch.object(settings, "memory_db_path", str(database_path)), patch.object(
                settings, "embedding_dimensions", 4
            ):
                async with sqlite_memory_store(embeddings=FakeEmbeddings()) as store:
                    await store.aput(
                        ("memories", "local-user", "user"),
                        "same-key",
                        {
                            "title": "User memory",
                            "summary": "User namespace.",
                            "keywords": ["user"],
                        },
                    )
                    await store.aput(
                        ("memories", "local-user", "project"),
                        "same-key",
                        {
                            "title": "Project memory",
                            "summary": "Project namespace.",
                            "keywords": ["project"],
                        },
                    )

                    user_item = await store.aget(
                        ("memories", "local-user", "user"),
                        "same-key",
                    )
                    project_item = await store.aget(
                        ("memories", "local-user", "project"),
                        "same-key",
                    )

                    self.assertEqual(user_item.value["title"], "User memory")
                    self.assertEqual(project_item.value["title"], "Project memory")


if __name__ == "__main__":
    unittest.main()
