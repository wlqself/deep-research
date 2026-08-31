import unittest
from types import SimpleNamespace

from deep_research.memory.service import MemoryService
from deep_research.memory.type import MemoryEntry


def make_entry(
    entry_id: str,
    *,
    kind: str,
    updated_at: str,
) -> MemoryEntry:
    data: dict[str, object] = {
            "id": entry_id,
            "memory_key": f"{kind}.{entry_id}",
            "scope": "project" if kind == "project" else "global",
            "kind": kind,
            "title": f"Title {entry_id}",
            "summary": f"Summary {entry_id}",
            "content": f"Content {entry_id}",
            "keywords": [entry_id],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
            "created_at": "2026-08-30T00:00:00Z",
            "updated_at": updated_at,
            "status": "active",
    }

    if kind == "reference":
        data.update(
            {
                "url": "https://example.com/reference",
                "verified_at": "2026-08-30T00:00:00Z",
            }
        )

    return MemoryEntry.model_validate(data)


class ListingStore:
    def __init__(self, entries_by_namespace: dict[tuple[str, ...], list[MemoryEntry]]):
        self.entries_by_namespace = entries_by_namespace
        self.calls: list[dict[str, object]] = []

    async def asearch(
        self,
        namespace: tuple[str, ...],
        *,
        query: str | None,
        filter: dict[str, object],
        limit: int,
    ) -> list[object]:
        self.calls.append(
            {
                "namespace": namespace,
                "query": query,
                "filter": filter,
                "limit": limit,
            }
        )
        return [
            SimpleNamespace(
                value=entry.model_dump(
                    mode="json",
                    exclude_none=False,
                )
            )
            for entry in self.entries_by_namespace.get(namespace, [])[:limit]
        ]


class MemoryListTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_all_kinds_and_applies_global_pagination(self):
        store = ListingStore(
            {
                ("memories", "local-user", "user"): [
                    make_entry(
                        "user-1",
                        kind="user",
                        updated_at="2026-08-30T01:00:00Z",
                    )
                ],
                ("memories", "local-user", "project"): [
                    make_entry(
                        "project-1",
                        kind="project",
                        updated_at="2026-08-30T03:00:00Z",
                    )
                ],
                ("memories", "local-user", "reference"): [
                    make_entry(
                        "reference-1",
                        kind="reference",
                        updated_at="2026-08-30T02:00:00Z",
                    )
                ],
                ("memories", "local-user", "feedback"): [],
            }
        )
        service = MemoryService(store, user_id="local-user")

        page = await service.list_active_memories(
            page=1,
            page_size=2,
        )

        self.assertEqual(
            [entry.id for entry in page],
            ["project-1", "reference-1"],
        )
        self.assertEqual(len(store.calls), 4)
        self.assertTrue(
            all(call["filter"] == {"status": "active"} for call in store.calls)
        )
        self.assertTrue(all(call["query"] is None for call in store.calls))

    async def test_kind_and_keyword_are_forwarded_without_exposing_namespace(self):
        store = ListingStore({})
        service = MemoryService(store, user_id="local-user")

        await service.list_active_memories(
            kind="project",
            keyword="memory boundary",
            page=2,
            page_size=3,
        )

        self.assertEqual(len(store.calls), 1)
        self.assertEqual(
            store.calls[0],
            {
                "namespace": ("memories", "local-user", "project"),
                "query": "memory boundary",
                "filter": {"status": "active"},
                "limit": 6,
            },
        )

    async def test_invalid_page_arguments_are_rejected(self):
        service = MemoryService(ListingStore({}), user_id="local-user")

        with self.assertRaises(ValueError):
            await service.list_active_memories(page=0)

        with self.assertRaises(ValueError):
            await service.list_active_memories(page_size=0)

        with self.assertRaises(ValueError):
            await service.list_active_memories(kind="invalid")


if __name__ == "__main__":
    unittest.main()
