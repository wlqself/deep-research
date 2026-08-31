import unittest

from deep_research.memory.type import MemoryEntry
from deep_research.tools.memory import build_forget_memory_tool


def make_entry(
    *,
    entry_id: str = "memory-1",
    status: str = "active",
) -> MemoryEntry:
    return MemoryEntry.model_validate(
        {
            "id": entry_id,
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
            "status": status,
        }
    )


class FakeForgetService:
    def __init__(self, entries: list[MemoryEntry] | None = None) -> None:
        self.entries = entries or []
        self.get_calls: list[tuple[str, str]] = []
        self.key_calls: list[tuple[str, str, str]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.forget_calls: list[tuple[str, str, dict[str, object]]] = []

    async def get_entry(self, kind: str, entry_id: str) -> MemoryEntry | None:
        self.get_calls.append((kind, entry_id))
        return next(
            (
                entry
                for entry in self.entries
                if entry.kind == kind and entry.id == entry_id
            ),
            None,
        )

    async def find_active_by_key(
        self,
        kind: str,
        memory_key: str,
        scope: str,
    ) -> list[MemoryEntry]:
        self.key_calls.append((kind, memory_key, scope))
        return self.entries

    async def delete_entry(self, kind: str, entry_id: str) -> None:
        self.delete_calls.append((kind, entry_id))

    async def forget_entry(
        self,
        kind: str,
        entry_id: str,
        **kwargs: object,
    ) -> None:
        self.forget_calls.append((kind, entry_id, kwargs))


class FailingForgetService(FakeForgetService):
    async def get_entry(self, kind: str, entry_id: str) -> MemoryEntry | None:
        raise RuntimeError("private delete details")


class MemoryForgetToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_does_not_expose_user_id_or_namespace(self):
        tool = build_forget_memory_tool(FakeForgetService())

        self.assertEqual(
            set(tool.tool_call_schema.model_fields),
            {"kind", "memory_id", "memory_key", "scope"},
        )

    async def test_forget_by_id_deletes_active_entry(self):
        service = FakeForgetService([make_entry()])
        tool = build_forget_memory_tool(service)

        result = await tool.coroutine(
            kind="user",
            memory_id="memory-1",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["deleted_memory_id"], "memory-1")
        self.assertEqual(
            service.forget_calls,
            [
                (
                    "user",
                    "memory-1",
                    {
                        "source_thread_id": None,
                        "reason": "explicit_user_forget",
                    },
                )
            ],
        )

    async def test_forget_by_key_requires_unique_scope_match(self):
        service = FakeForgetService([make_entry()])
        tool = build_forget_memory_tool(service)

        result = await tool.coroutine(
            kind="user",
            memory_key="user.answer_style.conciseness",
            scope="global",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(service.key_calls, [
            ("user", "user.answer_style.conciseness", "global")
        ])
        self.assertEqual(len(service.forget_calls), 1)
        self.assertEqual(
            service.forget_calls[0][:2],
            ("user", "memory-1"),
        )

    async def test_forget_requires_exactly_one_target(self):
        service = FakeForgetService([make_entry()])
        tool = build_forget_memory_tool(service)

        for kwargs in (
            {"kind": "user"},
            {
                "kind": "user",
                "memory_id": "memory-1",
                "memory_key": "user.answer_style.conciseness",
            },
        ):
            result = await tool.coroutine(**kwargs)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "invalid_forget_request")

        self.assertEqual(service.delete_calls, [])

    async def test_forget_does_not_delete_superseded_entry(self):
        service = FakeForgetService(
            [make_entry(status="superseded")]
        )
        tool = build_forget_memory_tool(service)

        result = await tool.coroutine(
            kind="user",
            memory_id="memory-1",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "memory_not_found")
        self.assertEqual(service.delete_calls, [])

    async def test_duplicate_key_and_service_error_are_safe(self):
        duplicate_service = FakeForgetService(
            [make_entry(entry_id="memory-1"), make_entry(entry_id="memory-2")]
        )
        duplicate_tool = build_forget_memory_tool(duplicate_service)

        duplicate_result = await duplicate_tool.coroutine(
            kind="user",
            memory_key="user.answer_style.conciseness",
            scope="global",
        )

        self.assertFalse(duplicate_result["ok"])
        self.assertEqual(
            duplicate_result["error"],
            "duplicate_active_memory_key",
        )
        self.assertEqual(duplicate_service.delete_calls, [])

        failing_tool = build_forget_memory_tool(FailingForgetService())
        failed_result = await failing_tool.coroutine(
            kind="user",
            memory_id="memory-1",
        )

        self.assertFalse(failed_result["ok"])
        self.assertEqual(failed_result["error"], "memory_forget_failed")
        self.assertNotIn("private delete details", failed_result["message"])


if __name__ == "__main__":
    unittest.main()
