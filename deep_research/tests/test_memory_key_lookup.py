import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from deep_research.memory.service import MemoryService
from deep_research.memory.type import MemoryEntry


def make_entry() -> MemoryEntry:
    return MemoryEntry.model_validate(
        {
            "id": "memory-1",
            "memory_key": "user.answer_style.conciseness",
            "scope": "global",
            "kind": "user",
            "title": "Answer style",
            "summary": "The user prefers concise answers.",
            "content": "Prefer concise answers.",
            "keywords": ["concise"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
            "created_at": "2026-08-29T00:00:00Z",
            "updated_at": "2026-08-29T00:00:00Z",
            "status": "active",
        }
    )


class MemoryKeyLookupTests(unittest.IsolatedAsyncioTestCase):
    async def test_lookup_uses_fixed_namespace_and_exact_active_filter(self):
        store = MagicMock()
        store.asearch = AsyncMock(return_value=[])
        service = MemoryService(store, user_id="local-user")

        result = await service.find_active_by_key(
            "user",
            " user.answer_style.conciseness ",
            "global",
        )

        self.assertEqual(result, [])
        store.asearch.assert_awaited_once_with(
            ("memories", "local-user", "user"),
            query=None,
            filter={
                "memory_key": "user.answer_style.conciseness",
                "scope": "global",
                "status": "active",
            },
            limit=2,
        )

    async def test_lookup_rehydrates_store_items_as_memory_entries(self):
        entry = make_entry()
        store = MagicMock()
        store.asearch = AsyncMock(
            return_value=[
                SimpleNamespace(
                    value=entry.model_dump(
                        mode="json",
                        exclude_none=False,
                    )
                )
            ]
        )
        service = MemoryService(store, user_id="local-user")

        result = await service.find_active_by_key(
            "user",
            entry.memory_key,
            entry.scope,
        )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], MemoryEntry)
        self.assertEqual(result[0].id, entry.id)

    async def test_lookup_rejects_empty_key_before_store_call(self):
        store = MagicMock()
        store.asearch = AsyncMock()
        service = MemoryService(store, user_id="local-user")

        with self.assertRaisesRegex(ValueError, "memory_key must not be empty"):
            await service.find_active_by_key(
                "user",
                "   ",
                "global",
            )

        store.asearch.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
