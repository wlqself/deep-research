import unittest
from unittest.mock import patch

from deep_research.config import settings
from deep_research.memory.type import MemoryEntry
from deep_research.tools.memory import build_list_memories_tool


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


class FakeListService:
    def __init__(self, entries: list[MemoryEntry] | None = None) -> None:
        self.entries = entries or []
        self.calls: list[dict[str, object]] = []

    async def list_active_memories(
        self,
        *,
        kind: str | None,
        keyword: str | None,
        page: int,
        page_size: int,
    ) -> list[MemoryEntry]:
        self.calls.append(
            {
                "kind": kind,
                "keyword": keyword,
                "page": page,
                "page_size": page_size,
            }
        )
        return self.entries


class FailingListService:
    async def list_active_memories(self, **kwargs: object) -> list[MemoryEntry]:
        raise RuntimeError("private list details")


class ListMemoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_exposes_only_list_filters(self):
        tool = build_list_memories_tool(FakeListService())

        self.assertEqual(
            set(tool.tool_call_schema.model_fields),
            {"kind", "keyword", "page"},
        )

    async def test_list_forwards_filters_and_uses_configured_page_size(self):
        service = FakeListService([make_entry()])
        tool = build_list_memories_tool(service)

        with patch.object(settings, "memory_page_size", 7):
            result = await tool.coroutine(
                kind="project",
                keyword="boundary",
                page=2,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            service.calls,
            [
                {
                    "kind": "project",
                    "keyword": "boundary",
                    "page": 2,
                    "page_size": 7,
                }
            ],
        )
        self.assertEqual(result["page"], 2)
        self.assertEqual(result["page_size"], 7)
        self.assertEqual(len(result["memories"]), 1)
        self.assertNotIn("source_thread_id", result["memories"][0])

    async def test_invalid_request_and_service_error_are_sanitized(self):
        class InvalidListService:
            async def list_active_memories(self, **kwargs: object) -> list[MemoryEntry]:
                raise ValueError("invalid page")

        invalid_tool = build_list_memories_tool(InvalidListService())
        invalid_result = await invalid_tool.coroutine(page=0)
        self.assertFalse(invalid_result["ok"])
        self.assertEqual(
            invalid_result["error"],
            "invalid_memory_list_request",
        )

        failing_tool = build_list_memories_tool(FailingListService())
        failing_result = await failing_tool.coroutine(page=1)
        self.assertFalse(failing_result["ok"])
        self.assertEqual(failing_result["error"], "memory_list_failed")
        self.assertNotIn("private list details", failing_result["message"])


if __name__ == "__main__":
    unittest.main()
