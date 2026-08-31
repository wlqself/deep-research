import unittest

from deep_research.memory.decision import decide_candidate
from deep_research.memory.service import MemoryService
from deep_research.memory.type import MemoryCandidate, MemoryEntry


def make_candidate() -> MemoryCandidate:
    return MemoryCandidate.model_validate(
        {
            "kind": "user",
            "memory_key": "user.answer_style.conciseness",
            "scope": "global",
            "title": "Answer style",
            "summary": "The user prefers concise answers.",
            "content": "Prefer detailed answers.",
            "keywords": ["detailed", "answer"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-2",
        }
    )


def make_existing() -> MemoryEntry:
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


class FailingBatchStore:
    def __init__(self, *, fail_compensation: bool = False) -> None:
        self.fail_compensation = fail_compensation
        self.batch_calls: list[list[object]] = []
        self.delete_calls: list[tuple[tuple[str, ...], str]] = []
        self.put_calls: list[dict[str, object]] = []

    async def abatch(self, ops: list[object]) -> list[object]:
        self.batch_calls.append(ops)
        raise RuntimeError("second write failed")

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        self.delete_calls.append((namespace, key))
        if self.fail_compensation:
            raise RuntimeError("compensation delete failed")

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, object],
        *,
        index: list[str],
    ) -> None:
        self.put_calls.append(
            {
                "namespace": namespace,
                "key": key,
                "value": value,
                "index": index,
            }
        )


class MemoryRollbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_update_compensates_new_entry_and_restores_old_entry(self):
        store = FailingBatchStore()
        service = MemoryService(store, user_id="local-user")
        existing = make_existing()
        candidate = make_candidate()
        decision = decide_candidate(
            candidate,
            [existing],
            intent="explicit_update",
        )

        with self.assertRaisesRegex(RuntimeError, "second write failed"):
            await service.apply_decision(
                candidate,
                decision,
                existing=existing,
            )

        self.assertEqual(len(store.batch_calls), 1)
        batch = store.batch_calls[0]
        new_entry_id = batch[1].key

        self.assertEqual(
            store.delete_calls,
            [(
                ("memories", "local-user", "user"),
                new_entry_id,
            )],
        )
        self.assertEqual(len(store.put_calls), 1)
        self.assertEqual(store.put_calls[0]["key"], existing.id)
        self.assertEqual(
            store.put_calls[0]["value"],
            existing.model_dump(mode="json", exclude_none=False),
        )

    async def test_compensation_failure_is_not_reported_as_success(self):
        store = FailingBatchStore(fail_compensation=True)
        service = MemoryService(store, user_id="local-user")
        existing = make_existing()
        candidate = make_candidate()
        decision = decide_candidate(
            candidate,
            [existing],
            intent="explicit_update",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "memory update failed and compensation failed",
        ):
            await service.apply_decision(
                candidate,
                decision,
                existing=existing,
            )

        self.assertEqual(store.put_calls, [])


if __name__ == "__main__":
    unittest.main()
