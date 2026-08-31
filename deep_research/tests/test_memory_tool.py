import unittest
from types import SimpleNamespace

from deep_research.tools.memory import build_remember_memory_tool
from deep_research.memory.type import MemoryEntry


def make_runtime(thread_id: str | None = "thread-1") -> SimpleNamespace:
    configurable = {}
    if thread_id is not None:
        configurable["thread_id"] = thread_id

    return SimpleNamespace(
        config={"configurable": configurable},
        state={},
        tool_call_id="tool-call-1",
    )


def tool_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "kind": "user",
        "memory_key": "user.answer_style.conciseness",
        "scope": "global",
        "title": "Answer style",
        "summary": "The user prefers concise answers.",
        "content": "Prefer concise answers.",
        "keywords": ["concise", "answer"],
        "runtime": make_runtime(),
    }
    values.update(overrides)
    return values


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
            "created_at": "2026-08-29T00:00:00Z",
            "updated_at": "2026-08-29T00:00:00Z",
            "status": "active",
        }
    )


class FakeMemoryService:
    def __init__(self, entries: list[MemoryEntry] | None = None) -> None:
        self.entries = entries or []
        self.lookup_calls: list[tuple[str, str, str]] = []
        self.apply_calls: list[tuple[object, object, object]] = []
        self.clear_suppression_calls: list[bool] = []

    async def find_active_by_key(
        self,
        kind: str,
        memory_key: str,
        scope: str,
    ) -> list[MemoryEntry]:
        self.lookup_calls.append((kind, memory_key, scope))
        return self.entries

    async def apply_decision(
        self,
        candidate: object,
        decision: object,
        *,
        existing: object = None,
        clear_suppression: bool = False,
    ) -> MemoryEntry | None:
        self.apply_calls.append((candidate, decision, existing))
        self.clear_suppression_calls.append(clear_suppression)
        return existing if existing is not None else make_existing()


class FailingMemoryService(FakeMemoryService):
    async def find_active_by_key(
        self,
        kind: str,
        memory_key: str,
        scope: str,
    ) -> list[MemoryEntry]:
        raise RuntimeError("secret database details")


class MemoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_schema_does_not_expose_namespace_or_user_id(self):
        tool = build_remember_memory_tool(FakeMemoryService())
        schema_fields = set(tool.tool_call_schema.model_fields)

        self.assertNotIn("user_id", schema_fields)
        self.assertNotIn("namespace", schema_fields)
        self.assertNotIn("runtime", schema_fields)

    async def test_remember_creates_explicit_user_candidate(self):
        service = FakeMemoryService()
        tool = build_remember_memory_tool(service)

        result = await tool.coroutine(**tool_kwargs())

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "create")
        self.assertEqual(
            service.lookup_calls,
            [
                (
                    "user",
                    "user.answer_style.conciseness",
                    "global",
                )
            ],
        )
        candidate, _decision, _existing = service.apply_calls[0]
        self.assertEqual(candidate.source_type, "explicit_user")
        self.assertEqual(candidate.source_thread_id, "thread-1")
        self.assertEqual(service.clear_suppression_calls, [True])

    async def test_update_requires_an_existing_active_memory(self):
        service = FakeMemoryService()
        tool = build_remember_memory_tool(service)

        result = await tool.coroutine(
            **tool_kwargs(mode="update")
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "memory_not_found")
        self.assertEqual(service.apply_calls, [])

    async def test_invalid_reference_is_rejected_before_store_lookup(self):
        service = FakeMemoryService()
        tool = build_remember_memory_tool(service)

        result = await tool.coroutine(
            **tool_kwargs(
                kind="reference",
                memory_key="reference.example",
                url="ftp://example.com",
                verified_at="2026-08-29T00:00:00Z",
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "invalid_memory")
        self.assertEqual(service.lookup_calls, [])

    async def test_missing_thread_id_is_rejected(self):
        service = FakeMemoryService()
        tool = build_remember_memory_tool(service)

        result = await tool.coroutine(
            **tool_kwargs(runtime=make_runtime(None))
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "missing_thread_id")
        self.assertEqual(service.lookup_calls, [])

    async def test_service_error_is_sanitized(self):
        tool = build_remember_memory_tool(FailingMemoryService())

        result = await tool.coroutine(**tool_kwargs())

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "memory_write_failed")
        self.assertNotIn("secret database details", result["message"])


if __name__ == "__main__":
    unittest.main()
