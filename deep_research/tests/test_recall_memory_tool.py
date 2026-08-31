import unittest
from unittest.mock import patch

from deep_research.config import settings
from deep_research.memory.type import MemoryEntry
from deep_research.tools.memory import build_recall_memories_tool


def make_entry() -> MemoryEntry:
    return MemoryEntry.model_validate(
        {
            "id": "memory-1",
            "memory_key": "project.memory.boundary",
            "scope": "project",
            "kind": "project",
            "title": "Memory boundary",
            "summary": "Memory belongs to Main.",
            "content": "Researcher must not access long-term memory.",
            "keywords": ["Main", "Researcher"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
            "created_at": "2026-08-30T00:00:00Z",
            "updated_at": "2026-08-30T00:00:00Z",
            "status": "active",
        }
    )


class FakeRecallService:
    def __init__(self, entries: list[MemoryEntry] | None = None) -> None:
        self.entries = entries or []
        self.calls: list[tuple[str, int]] = []

    async def find_review_memories(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[MemoryEntry]:
        self.calls.append((query, limit))
        return self.entries


class FailingRecallService:
    async def find_review_memories(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[MemoryEntry]:
        raise RuntimeError("private store details")


class RecallMemoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_exposes_only_query(self):
        tool = build_recall_memories_tool(FakeRecallService())

        self.assertEqual(
            set(tool.tool_call_schema.model_fields),
            {"query"},
        )

    async def test_recall_returns_safe_memory_fields_and_configured_limit(self):
        service = FakeRecallService([make_entry()])
        tool = build_recall_memories_tool(service)

        with patch.object(settings, "memory_recall_limit", 2):
            result = await tool.coroutine(
                query="Main memory boundary"
            )

        self.assertTrue(result["ok"])
        self.assertEqual(service.calls, [("Main memory boundary", 2)])
        self.assertEqual(len(result["memories"]), 1)

        memory = result["memories"][0]
        self.assertEqual(memory["memory_key"], "project.memory.boundary")
        self.assertEqual(memory["content"], "Researcher must not access long-term memory.")
        self.assertNotIn("source_thread_id", memory)
        self.assertNotIn("embedding", memory)
        self.assertNotIn("namespace", memory)

    async def test_empty_query_does_not_call_service(self):
        service = FakeRecallService([make_entry()])
        tool = build_recall_memories_tool(service)

        result = await tool.coroutine(query="   ")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "invalid_query")
        self.assertEqual(service.calls, [])

    async def test_recall_failure_is_sanitized(self):
        tool = build_recall_memories_tool(FailingRecallService())

        result = await tool.coroutine(query="memory boundary")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "memory_recall_failed")
        self.assertNotIn("private store details", result["message"])


if __name__ == "__main__":
    unittest.main()
