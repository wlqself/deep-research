import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from deep_research.memory.decision import decide_candidate
from deep_research.memory.service import MemoryService
from deep_research.memory.type import MemoryCandidate, MemoryEntry


def make_candidate(
    *,
    kind: str = "user",
    content: str = "Prefer concise answers.",
) -> MemoryCandidate:
    return MemoryCandidate.model_validate(
        {
            "kind": kind,
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
            "keywords": ["concise", "answer"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
            "created_at": "2026-08-28T00:00:00Z",
            "updated_at": "2026-08-28T00:00:00Z",
            "status": "active",
        }
    )


def make_service() -> tuple[MemoryService, MagicMock]:
    store = MagicMock()
    store.aput = AsyncMock()
    store.abatch = AsyncMock()
    store.adelete = AsyncMock()
    return MemoryService(store, user_id="local-user"), store


class MemoryProjectionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_put_entry_rebuilds_after_store_write(self):
        service, store = make_service()

        with patch.object(
            service,
            "rebuild_projections",
            new_callable=AsyncMock,
        ) as rebuild:
            await service.put_entry(make_entry())

        store.aput.assert_awaited_once()
        rebuild.assert_awaited_once()

    async def test_delete_entry_rebuilds_after_store_delete(self):
        service, store = make_service()

        with patch.object(
            service,
            "rebuild_projections",
            new_callable=AsyncMock,
        ) as rebuild:
            await service.delete_entry("user", "memory-1")

        store.adelete.assert_awaited_once()
        rebuild.assert_awaited_once()

    async def test_no_op_and_ignore_do_not_rebuild(self):
        service, store = make_service()
        existing = make_entry()

        no_op_decision = decide_candidate(
            make_candidate(),
            [existing],
            intent="explicit_remember",
        )
        ignore_candidate = make_candidate(kind="ignore")
        ignore_decision = decide_candidate(
            ignore_candidate,
            [],
            intent="automatic",
        )

        with patch.object(
            service,
            "rebuild_projections",
            new_callable=AsyncMock,
        ) as rebuild:
            await service.apply_decision(
                make_candidate(),
                no_op_decision,
                existing=existing,
            )
            await service.apply_decision(
                ignore_candidate,
                ignore_decision,
            )

        store.aput.assert_not_awaited()
        store.abatch.assert_not_awaited()
        store.adelete.assert_not_awaited()
        rebuild.assert_not_awaited()

    async def test_create_rebuilds_after_store_write(self):
        service, store = make_service()
        candidate = make_candidate()
        decision = decide_candidate(
            candidate,
            [],
            intent="automatic",
        )

        with patch.object(
            service,
            "rebuild_projections",
            new_callable=AsyncMock,
        ) as rebuild:
            created = await service.apply_decision(
                candidate,
                decision,
            )

        self.assertIsNotNone(created)
        store.abatch.assert_awaited_once()
        rebuild.assert_awaited_once()

    async def test_update_rebuilds_after_store_write(self):
        service, store = make_service()
        existing = make_entry()
        candidate = make_candidate(
            content="Prefer detailed answers.",
        )
        decision = decide_candidate(
            candidate,
            [existing],
            intent="explicit_update",
        )

        with patch.object(
            service,
            "rebuild_projections",
            new_callable=AsyncMock,
        ) as rebuild:
            updated = await service.apply_decision(
                candidate,
                decision,
                existing=existing,
            )

        self.assertIsNotNone(updated)
        store.abatch.assert_awaited_once()
        rebuild.assert_awaited_once()

    async def test_projection_failure_does_not_fail_store_write(self):
        service, store = make_service()

        with patch.object(
            service,
            "rebuild_projections",
            new_callable=AsyncMock,
            side_effect=OSError("projection failed"),
        ):
            await service.put_entry(make_entry())

        store.aput.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
