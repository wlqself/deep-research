import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from deep_research.memory.decision import decide_candidate
from deep_research.memory.service import MemoryService
from deep_research.memory.type import MemoryCandidate, MemoryEntry


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
            "created_at": "2026-08-30T00:00:00Z",
            "updated_at": "2026-08-30T00:00:00Z",
            "status": "active",
        }
    )


def make_candidate() -> MemoryCandidate:
    return MemoryCandidate.model_validate(
        {
            "kind": "user",
            "memory_key": "user.answer_style.conciseness",
            "scope": "global",
            "title": "Answer style",
            "summary": "The user prefers concise answers.",
            "content": "Prefer concise answers.",
            "keywords": ["concise"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-2",
        }
    )


def make_service(store: MagicMock) -> MemoryService:
    return MemoryService(store, user_id="local-user")


class MemoryRevisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_store_mutations_increment_revision(self):
        store = MagicMock()
        store.aput = AsyncMock()
        service = make_service(store)

        with patch.object(
            service,
            "rebuild_projections",
            new_callable=AsyncMock,
        ):
            await service.put_entry(make_entry())

        self.assertEqual(service.recall_revision, 1)

    async def test_no_op_does_not_increment_revision(self):
        store = MagicMock()
        service = make_service(store)
        existing = make_entry()
        decision = decide_candidate(
            make_candidate(),
            [existing],
            intent="explicit_remember",
        )

        result = await service.apply_decision(
            make_candidate(),
            decision,
            existing=existing,
        )

        self.assertEqual(result.id, existing.id)
        self.assertEqual(service.recall_revision, 0)

    async def test_failed_store_write_does_not_increment_revision(self):
        store = MagicMock()
        store.aput = AsyncMock(
            side_effect=RuntimeError("store failed")
        )
        service = make_service(store)

        with self.assertRaisesRegex(RuntimeError, "store failed"):
            await service.put_entry(make_entry())

        self.assertEqual(service.recall_revision, 0)


if __name__ == "__main__":
    unittest.main()
