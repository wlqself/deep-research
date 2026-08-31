import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from deep_research.handlers import research as research_module


def fake_http_request() -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                memory_service=object(),
                memory_extractor=object(),
            )
        )
    )


async def collect_body(response) -> list[dict[str, object]]:
    lines = [
        line
        async for line in response.body_iterator
    ]
    return [json.loads(line) for line in lines]


class MemoryHttpStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_done_with_nonempty_answer_reviews_once(self):
        async def events(question, thread_id):
            yield {"type": "text", "text": "第一段"}
            yield {"type": "text", "text": "第二段"}
            yield {"type": "done"}

        with patch.object(
            research_module,
            "stream_research_events",
            new=events,
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            response = await research_module.research_stream(
                research_module.ResearchRequest(question="问题"),
                fake_http_request(),
            )
            body = await collect_body(response)

        self.assertEqual(
            body,
            [
                {"type": "text", "text": "第一段"},
                {"type": "text", "text": "第二段"},
                {"type": "done"},
            ],
        )
        review.assert_awaited_once()
        self.assertEqual(review.await_args.kwargs["answer"], "第一段第二段")

    async def test_empty_answer_does_not_review(self):
        async def events(question, thread_id):
            yield {"type": "done"}

        with patch.object(
            research_module,
            "stream_research_events",
            new=events,
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            response = await research_module.research_stream(
                research_module.ResearchRequest(question="问题"),
                fake_http_request(),
            )
            await collect_body(response)

        review.assert_not_awaited()

    async def test_saved_artifact_marks_review_as_report_saved(self):
        async def events(question, thread_id):
            yield {"type": "text", "text": "报告内容"}
            yield {
                "type": "artifact_saved",
                "artifact_id": "artifact-1",
            }
            yield {"type": "done"}

        with patch.object(
            research_module,
            "stream_research_events",
            new=events,
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            response = await research_module.research_stream(
                research_module.ResearchRequest(question="问题"),
                fake_http_request(),
            )
            await collect_body(response)

        review.assert_awaited_once()
        self.assertTrue(review.await_args.kwargs["report_saved"])

    async def test_cancelled_stream_does_not_review(self):
        async def events(question, thread_id):
            yield {"type": "text", "text": "未完成"}
            raise asyncio.CancelledError

        with patch.object(
            research_module,
            "stream_research_events",
            new=events,
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            response = await research_module.research_stream(
                research_module.ResearchRequest(question="问题"),
                fake_http_request(),
            )

            with self.assertRaises(asyncio.CancelledError):
                await collect_body(response)

        review.assert_not_awaited()

    async def test_failed_stream_does_not_review(self):
        async def events(question, thread_id):
            yield {"type": "text", "text": "失败前内容"}
            raise RuntimeError("stream failed")

        with patch.object(
            research_module,
            "stream_research_events",
            new=events,
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            response = await research_module.research_stream(
                research_module.ResearchRequest(question="问题"),
                fake_http_request(),
            )

            with self.assertRaisesRegex(RuntimeError, "stream failed"):
                await collect_body(response)

        review.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
