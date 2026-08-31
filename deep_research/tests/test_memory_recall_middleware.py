import unittest

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.memory.type import MemoryEntry, SummaryArchive
from deep_research.middleware.recall_middleware import (
    MainMemoryRecallMiddleware,
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
            "keywords": ["concise"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
            "created_at": "2026-08-30T00:00:00Z",
            "updated_at": "2026-08-30T00:00:00Z",
            "status": "active",
        }
    )


def make_archive() -> SummaryArchive:
    return SummaryArchive.model_validate(
        {
            "id": "archive-1",
            "thread_id": "thread-1",
            "summary_hash": "hash-1",
            "original_summary": "Original summary.",
            "retrieval_summary": "Archived project decision.",
            "topics": ["project"],
            "memory_keys": ["project:global:project.decision"],
            "recallable": True,
            "version": 1,
            "archived_at": "2026-08-30T00:00:00Z",
        }
    )


class FakeMemoryService:
    def __init__(self, archives: list[SummaryArchive] | None = None) -> None:
        self.user_calls = 0
        self.related_calls = 0
        self.recall_revision = 0
        self.archives = archives or []

    async def list_active_memories(self, **kwargs: object) -> list[MemoryEntry]:
        self.user_calls += 1
        return [make_entry()]

    async def find_review_memories(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[MemoryEntry]:
        self.related_calls += 1
        return []

    async def search_summary_archives(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[SummaryArchive]:
        return self.archives


class FailingMemoryService:
    async def list_active_memories(self, **kwargs: object) -> list[MemoryEntry]:
        raise RuntimeError("memory unavailable")

    async def find_review_memories(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[MemoryEntry]:
        raise RuntimeError("memory unavailable")

    async def search_summary_archives(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[SummaryArchive]:
        raise RuntimeError("memory unavailable")


def make_request(message_id: str, content: str) -> ModelRequest:
    return ModelRequest(
        model=object(),
        messages=[
            HumanMessage(
                id=message_id,
                content=content,
            )
        ],
        system_message=SystemMessage(content="Base system prompt"),
    )


class MemoryRecallMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_user_turn_uses_one_cached_recall(self):
        service = FakeMemoryService()
        middleware = MainMemoryRecallMiddleware(service)
        seen_requests: list[ModelRequest] = []

        async def handler(request: ModelRequest):
            seen_requests.append(request)
            return "ok"

        request = make_request("user-1", "Please answer concisely.")

        await middleware.awrap_model_call(request, handler)
        await middleware.awrap_model_call(request, handler)

        self.assertEqual(service.user_calls, 1)
        self.assertEqual(service.related_calls, 1)
        self.assertEqual(len(seen_requests), 2)
        self.assertIn("Base system prompt", seen_requests[0].system_message.text)
        self.assertIn("Answer style", seen_requests[0].system_message.text)
        self.assertEqual(len(seen_requests[0].messages), 1)

    async def test_new_user_message_invalidates_turn_cache(self):
        service = FakeMemoryService()
        middleware = MainMemoryRecallMiddleware(service)

        async def handler(request: ModelRequest):
            return "ok"

        await middleware.awrap_model_call(
            make_request("user-1", "First question"),
            handler,
        )
        await middleware.awrap_model_call(
            make_request("user-2", "Second question"),
            handler,
        )

        self.assertEqual(service.user_calls, 2)
        self.assertEqual(service.related_calls, 2)

    async def test_store_revision_invalidates_same_turn_cache(self):
        service = FakeMemoryService()
        middleware = MainMemoryRecallMiddleware(service)

        async def handler(request: ModelRequest):
            return "ok"

        request = make_request("user-1", "Same question")

        await middleware.awrap_model_call(request, handler)
        service.recall_revision = 1
        await middleware.awrap_model_call(request, handler)

        self.assertEqual(service.user_calls, 2)
        self.assertEqual(service.related_calls, 2)

    async def test_summary_archive_is_injected_as_related_memory(self):
        service = FakeMemoryService([make_archive()])
        middleware = MainMemoryRecallMiddleware(service)
        seen_requests: list[ModelRequest] = []

        async def handler(request: ModelRequest):
            seen_requests.append(request)
            return "ok"

        await middleware.awrap_model_call(
            make_request("user-1", "What was the project decision?"),
            handler,
        )

        system_text = seen_requests[0].system_message.text
        self.assertNotIn("Summary Archive", system_text)
        self.assertNotIn("Original summary.", system_text)
        self.assertIn("Archived project decision.", system_text)

    async def test_legacy_archive_is_excluded_from_main_prompt(self):
        legacy_archive = make_archive().model_copy(
            update={
                "memory_keys": [],
                "recallable": False,
            }
        )
        service = FakeMemoryService([legacy_archive])
        middleware = MainMemoryRecallMiddleware(service)
        seen_requests: list[ModelRequest] = []

        async def handler(request: ModelRequest):
            seen_requests.append(request)
            return "ok"

        await middleware.awrap_model_call(
            make_request("user-1", "What was the project decision?"),
            handler,
        )

        system_text = seen_requests[0].system_message.text
        self.assertNotIn("Archived project decision.", system_text)
        self.assertNotIn("Original summary.", system_text)

    async def test_recall_failure_degrades_to_original_request(self):
        middleware = MainMemoryRecallMiddleware(FailingMemoryService())
        seen_requests: list[ModelRequest] = []

        async def handler(request: ModelRequest):
            seen_requests.append(request)
            return "ok"

        result = await middleware.awrap_model_call(
            make_request("user-1", "Question"),
            handler,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(
            seen_requests[0].system_message.text,
            "Base system prompt",
        )


if __name__ == "__main__":
    unittest.main()
