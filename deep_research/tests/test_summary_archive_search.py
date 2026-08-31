import unittest
from types import SimpleNamespace

from deep_research.memory.service import MemoryService
from deep_research.memory.type import SummaryArchive


def make_archive() -> SummaryArchive:
    return SummaryArchive.model_validate(
        {
            "id": "archive-1",
            "thread_id": "thread-1",
            "summary_hash": "hash-1",
            "original_summary": "Original summary.",
            "retrieval_summary": "Retrieval summary.",
            "topics": ["memory"],
            "memory_keys": ["user:global:user.phone"],
            "recallable": True,
            "version": 1,
            "archived_at": "2026-08-30T00:00:00Z",
        }
    )


class ArchiveSearchStore:
    def __init__(self, suppressed_keys: set[str] | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
        self.suppressed_keys = suppressed_keys or set()

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        return object() if key in self.suppressed_keys else None

    async def asearch(
        self,
        namespace: tuple[str, ...],
        *,
        query: str | None,
        limit: int,
        offset: int = 0,
    ) -> list[object]:
        self.calls.append(
            (
                namespace,
                {
                    "query": query,
                    "limit": limit,
                    "offset": offset,
                },
            )
        )
        return [
            SimpleNamespace(
                value=make_archive().model_dump(
                    mode="json"
                )
            )
        ]


class SummaryArchiveSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_can_scope_to_one_thread_or_all_threads(self):
        store = ArchiveSearchStore()
        service = MemoryService(store, user_id="local-user")

        one_thread = await service.list_summary_archives(
            thread_id="thread-1",
            limit=2,
            offset=3,
        )
        all_threads = await service.list_summary_archives(
            limit=5,
        )

        self.assertEqual(len(one_thread), 1)
        self.assertEqual(len(all_threads), 1)
        self.assertEqual(
            store.calls[0],
            (
                (
                    "memory-internal",
                    "local-user",
                    "summary-archive",
                    "thread-1",
                ),
                {"query": None, "limit": 2, "offset": 3},
            ),
        )
        self.assertEqual(
            store.calls[1],
            (
                (
                    "memory-internal",
                    "local-user",
                    "summary-archive",
                ),
                {"query": None, "limit": 5, "offset": 0},
            ),
        )

    async def test_search_uses_user_archive_prefix_and_normalizes_query(self):
        store = ArchiveSearchStore()
        service = MemoryService(store, user_id="local-user")

        result = await service.search_summary_archives(
            "  memory boundary  ",
            limit=3,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            store.calls[0],
            (
                (
                    "memory-internal",
                    "local-user",
                    "summary-archive",
                ),
                {"query": "memory boundary", "limit": 3, "offset": 0},
            ),
        )

    async def test_empty_search_does_not_call_store(self):
        store = ArchiveSearchStore()
        service = MemoryService(store, user_id="local-user")

        self.assertEqual(
            await service.search_summary_archives("   ", limit=3),
            [],
        )
        self.assertEqual(store.calls, [])

    async def test_search_excludes_archive_with_suppressed_identity(self):
        store = ArchiveSearchStore(
            suppressed_keys={"user:global:user.phone"}
        )
        service = MemoryService(store, user_id="local-user")

        result = await service.search_summary_archives(
            "memory",
            limit=3,
        )

        self.assertEqual(result, [])

    async def test_archive_list_arguments_are_validated(self):
        store = ArchiveSearchStore()
        service = MemoryService(store, user_id="local-user")

        with self.assertRaises(ValueError):
            await service.list_summary_archives(limit=0)

        with self.assertRaises(ValueError):
            await service.list_summary_archives(limit=1, offset=-1)

        with self.assertRaises(ValueError):
            await service.list_summary_archives(
                thread_id="thread.with.dot",
                limit=1,
            )

        with self.assertRaises(ValueError):
            await service.search_summary_archives(
                "memory",
                limit=0,
            )


if __name__ == "__main__":
    unittest.main()
