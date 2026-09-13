import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import HTTPException

from deep_research.handlers import research as research_module
from deep_research.handlers.research_stream import stream_answer


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
    async def test_new_turn_is_blocked_while_native_hitl_is_pending(self):
        class InterruptedAgent:
            async def aget_state(self, config):
                return SimpleNamespace(
                    interrupts=(
                        SimpleNamespace(
                            value={
                                "kind": "publication_approval",
                                "interaction_id": "interaction-1",
                            },
                        ),
                    ),
                    tasks=(),
                )

        http_request = fake_http_request()
        http_request.app.state.agent = InterruptedAgent()

        with self.assertRaises(HTTPException) as raised:
            await research_module.research_stream(
                research_module.ResearchRequest(
                    question="新的消息",
                    thread_id=uuid4(),
                ),
                http_request,
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(
            raised.exception.detail["interaction_id"],
            "interaction-1",
        )

    async def test_waiting_for_user_preserves_native_hitl_checkpoint(self):
        async def events(question, thread_id):
            yield {"type": "done", "goal_status": "waiting_for_user"}

        persisted_agent = SimpleNamespace(
            aupdate_state=AsyncMock(),
        )

        iterator = stream_answer(
            "需要审批",
            "waiting-thread",
            memory_service=None,
            memory_extractor=None,
            agent=persisted_agent,
            stream_events=events,
            review_callback=AsyncMock(),
            review_runner=AsyncMock(),
            has_explicit_correction=lambda question: False,
            has_confirmed_project_decision=lambda question: False,
            has_memory_rule_signal=lambda question: False,
            logger=research_module.logger,
        )

        await collect_body(SimpleNamespace(body_iterator=iterator))

        persisted_agent.aupdate_state.assert_not_awaited()

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

        raw_body = [
            event
            for event in body
            if event.get("type") != "activity"
        ]
        activity_body = [
            event["activity"]
            for event in body
            if event.get("type") == "activity"
        ]

        self.assertEqual(
            raw_body,
            [
                {"type": "text", "text": "第一段"},
                {"type": "text", "text": "第二段"},
                {"type": "done"},
            ],
        )
        self.assertGreaterEqual(len(activity_body), 3)
        self.assertEqual(activity_body[0]["kind"], "question")
        self.assertEqual(activity_body[-1]["status"], "completed")
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

    async def test_closing_stream_records_standard_cancellation(self):
        async def events(question, thread_id):
            yield {"type": "text", "text": "partial"}
            await asyncio.Event().wait()

        persisted_agent = SimpleNamespace(
            aupdate_state=AsyncMock(),
        )

        with patch.object(
            research_module,
            "stream_research_events",
            new=events,
        ), patch.object(
            research_module.agent_module,
            "agent",
            persisted_agent,
        ), patch(
            "deep_research.handlers.research_stream.log_event",
        ) as log_event:
            iterator = stream_answer(
                "问题",
                "cancelled-thread",
                memory_service=None,
                memory_extractor=None,
                agent=persisted_agent,
                stream_events=events,
                review_callback=AsyncMock(),
                review_runner=AsyncMock(),
                has_explicit_correction=lambda question: False,
                has_confirmed_project_decision=lambda question: False,
                has_memory_rule_signal=lambda question: False,
                logger=research_module.logger,
            )
            for _ in range(3):
                await anext(iterator)
            await iterator.aclose()

        self.assertTrue(
            any(
                call.args[2] == "research.cancelled"
                for call in log_event.call_args_list
            )
        )
        persisted_agent.aupdate_state.assert_awaited()


if __name__ == "__main__":
    unittest.main()
