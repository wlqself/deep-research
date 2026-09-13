import unittest

import deep_research.agent as agent_module
from deep_research.context import ResearchContext
from deep_research.agent.streaming import (
    stream_research_events_with_agent,
)
from langgraph.types import Interrupt
from langgraph.errors import GraphInterrupt


class FakeToken:
    def __init__(self, text: str):
        self.text = text

class FakeSnapshot:
    values = {
        "sources": {
            "S1": {
                "source_id": "S1",
                "title": "Test source",
                "url": "https://example.com/test",
                "snippet": "Test snippet",
            }
        },
        "next_source_number": 2,
    }

class FakeAgent:
    async def aget_state(self, config):
        return FakeSnapshot()
    
    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        yield {
            "event": "on_tool_start",
            "name": "web_search",
            "data": {
                "input": {
                    "query": "test query",
                    "secret": "must not be displayed",
                }
            },
        }

        yield {
            "event": "on_tool_end",
            "name": "web_search",
            "data": {
                "output": {
                    "ok": True,
                    "results": [],
                }
            },
        }

        yield {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {
                "lc_source": "summarization",
                "langgraph_node": "summarize",
            },
            "data": {
                "chunk": FakeToken(
                    "## ORIGINAL QUESTION internal summary",
                ),
            },
        }

        yield {
            "event": "on_chat_model_end",
            "name": "ChatOpenAI",
            "metadata": {
                "lc_source": "summarization",
                "langgraph_node": "summarize",
            },
            "data": {
                "output": "internal summary must not be exposed",
            },
        }

        yield {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {
                "langgraph_node": "model",
            },
            "data": {
                "chunk": FakeToken("答案 [S1]"),
            },
        }


class InterruptFakeAgent:
    def __init__(self):
        self.drained_after_interrupt = False

    async def aget_state(self, config):
        return FakeSnapshot()

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        yield {
            "event": "on_tool_start",
            "name": "request_publication_approval",
            "data": {"input": {}},
        }

        yield {
            "event": "on_chain_stream",
            "name": "LangGraph",
            "data": {
                "chunk": {
                    "__interrupt__": (
                        Interrupt(
                            value={
                                "kind": "publication_approval",
                                "ok": True,
                                "action": "approve",
                                "resolution_status": "resolved",
                                "target": {
                                    "approval_id": "approval-1",
                                    "article_id": "article-1",
                                    "article_title": "Test article",
                                    "article_slug": "test-article",
                                    "article_version": 1,
                                    "channel": "local_static_site",
                                    "approval_status": "pending",
                                },
                                "candidates": [],
                                "requires_user_confirmation": True,
                                "interaction_id": "interaction-1",
                                "interaction_status": "pending",
                            },
                            id="interrupt-1",
                        ),
                    ),
                },
            },
        }

        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {}},
        }
        self.drained_after_interrupt = True


class LateTextInterruptFakeAgent(InterruptFakeAgent):
    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        async for event in super().astream_events(
            input_data,
            config,
            context,
            version,
        ):
            yield event
        yield {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {"langgraph_node": "model"},
            "data": {"chunk": FakeToken("模型仍在输出的尾部")},
        }


class ToolErrorInterruptFakeAgent:
    async def aget_state(self, config):
        return FakeSnapshot()

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        value = {
            "kind": "publication_approval",
            "ok": True,
            "action": "approve",
            "resolution_status": "resolved",
            "target": {
                "approval_id": "approval-error-1",
                "article_id": "article-error-1",
                "article_title": "Error event article",
                "article_slug": "error-event-article",
                "article_version": 1,
                "channel": "local_static_site",
                "approval_status": "pending",
            },
            "candidates": [],
            "requires_user_confirmation": True,
            "interaction_id": "interaction-error-1",
            "interaction_status": "pending",
        }
        yield {
            "event": "on_tool_start",
            "name": "request_publication_approval",
            "data": {"input": {}},
        }
        yield {
            "event": "on_tool_error",
            "name": "request_publication_approval",
            "data": {
                "error": GraphInterrupt(
                    (Interrupt(value=value, id="interrupt-error-1"),)
                ),
            },
        }


class BusinessFailureFakeAgent:
    async def aget_state(self, config):
        return FakeSnapshot()

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        yield {
            "event": "on_tool_start",
            "name": "prepare_article_for_publication",
            "data": {"input": {"slug": "langchain-1.0-updates"}},
        }
        yield {
            "event": "on_tool_end",
            "name": "prepare_article_for_publication",
            "data": {
                "output": {
                    "ok": False,
                    "error_code": "invalid_slug",
                    "retryable": False,
                }
            },
        }


class RepeatingOutputFakeAgent:
    def __init__(self):
        self.closed = False

    async def aget_state(self, config):
        return FakeSnapshot()

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        block = "".join(
            f"step-{index:03d}:prepare-artifact;"
            for index in range(12)
        )
        try:
            for _ in range(3):
                yield {
                    "event": "on_chat_model_stream",
                    "name": "ChatOpenAI",
                    "metadata": {"langgraph_node": "model"},
                    "data": {"chunk": FakeToken(block)},
                }
            yield {
                "event": "on_chat_model_stream",
                "name": "ChatOpenAI",
                "metadata": {"langgraph_node": "model"},
                "data": {"chunk": FakeToken("must not be emitted")},
            }
        finally:
            self.closed = True

class FakeContextFactory:
    context = ResearchContext(
        max_search_calls=4,
        max_page_reads=6,
        max_page_chars=12000,
    )

    context.register_source(
        "Test source",
        "https://example.com/test",
        "Test snippet",
    )

    @classmethod
    def from_settings(
        cls,
        *,
        next_source_number: int = 1,
    ):
        return cls.context


class StreamEventsTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_events_and_text_are_emitted(self):
        original_agent = agent_module.agent
        original_context = agent_module.ResearchContext

        agent_module.agent = FakeAgent()
        agent_module.ResearchContext = FakeContextFactory

        try:
            events = [
                event
                async for event in agent_module.stream_research_events(
                    "test question", "test-thread"
                )
            ]
        finally:
            agent_module.agent = original_agent
            agent_module.ResearchContext = original_context

        event_types = [
            event["type"]
            for event in events
        ]

        self.assertEqual(
            event_types,
            [
                "tool_start",
                "tool_end",
                "memory_compacted",
                "text",
                "text",
                "done",
            ],
        )

        tool_start = events[0]

        self.assertEqual(
            tool_start["input"],
            {"query": "test query"},
        )

        self.assertNotIn(
            "secret",
            str(tool_start),
        )

        memory_event = events[2]

        self.assertEqual(
            memory_event,
            {
                "type": "memory_compacted",
                "message": "当前会话已进行一次上下文整理。",
            },
        )

        self.assertNotIn(
            "internal summary",
            str(memory_event),
        )

        final_text = "".join(
            event["text"]
            for event in events
            if event["type"] == "text"
        )

        self.assertIn("答案 [S1]", final_text)
        self.assertNotIn("ORIGINAL QUESTION", final_text)
        self.assertNotIn("internal summary", final_text)
        self.assertIn(
            "[S1] Test source - https://example.com/test",
            final_text,
        )
        self.assertEqual(events[-1]["goal_status"], "completed")

    async def test_native_interrupt_drains_stream_without_repeated_tool_call(self):
        agent = InterruptFakeAgent()
        events = [
            event
            async for event in stream_research_events_with_agent(
                agent,
                "request publication approval",
                "interrupt-thread",
            )
        ]

        self.assertEqual(
            [event["type"] for event in events],
            ["tool_start", "publishing_intent", "done"],
        )
        self.assertEqual(
            events[1]["interaction_id"],
            "interaction-1",
        )
        self.assertTrue(events[1]["native_interrupt"])
        self.assertEqual(events[-1]["goal_status"], "waiting_for_user")
        self.assertEqual(
            sum(
                event.get("name") == "request_publication_approval"
                for event in events
            ),
            1,
        )
        self.assertTrue(agent.drained_after_interrupt)

    async def test_tool_error_interrupt_is_waiting_not_failed(self):
        events = [
            event
            async for event in stream_research_events_with_agent(
                ToolErrorInterruptFakeAgent(),
                "request publication approval",
                "tool-error-interrupt-thread",
            )
        ]

        self.assertEqual(
            [event["type"] for event in events],
            ["tool_start", "publishing_intent", "done"],
        )
        self.assertEqual(events[1]["interaction_id"], "interaction-error-1")
        self.assertEqual(events[-1]["goal_status"], "waiting_for_user")

    async def test_hitl_card_waits_until_assistant_stream_is_complete(self):
        events = [
            event
            async for event in stream_research_events_with_agent(
                LateTextInterruptFakeAgent(),
                "request publication approval",
                "late-text-interrupt-thread",
            )
        ]

        event_types = [event["type"] for event in events]
        self.assertLess(event_types.index("text"), event_types.index("publishing_intent"))
        self.assertLess(event_types.index("publishing_intent"), event_types.index("done"))
        self.assertEqual(events[-1]["goal_status"], "waiting_for_user")

    async def test_business_failure_is_not_reported_as_tool_success(self):
        events = [
            event
            async for event in stream_research_events_with_agent(
                BusinessFailureFakeAgent(),
                "prepare article",
                "business-failure-thread",
            )
        ]

        self.assertEqual(events[0]["type"], "tool_start")
        self.assertEqual(
            events[1],
            {
                "type": "tool_end",
                "name": "prepare_article_for_publication",
                "status": "failed",
                "elapsed_ms": events[1]["elapsed_ms"],
                "error_code": "invalid_slug",
                "retryable": False,
            },
        )
        self.assertEqual(
            events[-1]["goal_status"],
            "completed_with_failure",
        )
        self.assertEqual(events[-1]["error_codes"], ["invalid_slug"])

    async def test_repeated_model_output_is_stopped_and_stream_is_closed(self):
        agent = RepeatingOutputFakeAgent()

        events = [
            event
            async for event in stream_research_events_with_agent(
                agent,
                "prepare article",
                "repeating-output-thread",
            )
        ]

        self.assertTrue(agent.closed)
        self.assertFalse(
            any(
                event.get("text") == "must not be emitted"
                for event in events
            )
        )
        error = next(event for event in events if event["type"] == "error")
        self.assertEqual(error["code"], "model_repetition_detected")
        self.assertEqual(
            events[-1]["goal_status"],
            "completed_with_failure",
        )
        self.assertIn(
            "model_repetition_detected",
            events[-1]["error_codes"],
        )

    async def test_first_non_retryable_tool_failure_stops_agent_loop(self):
        class NoProgressAgent:
            def __init__(self):
                self.closed = False

            async def aget_state(self, config):
                return FakeSnapshot()

            async def astream_events(
                inner_self,
                input_data,
                config,
                context,
                version,
            ):
                try:
                    for index in range(3):
                        yield {
                            "event": "on_tool_start",
                            "name": "request_publication_approval",
                            "run_id": f"call-{index}",
                            "data": {"input": {"target_hint": str(index)}},
                        }
                        yield {
                            "event": "on_tool_end",
                            "name": "request_publication_approval",
                            "run_id": f"call-{index}",
                            "data": {
                                "output": {
                                    "ok": True,
                                    "resolution_status": "not_found",
                                    "error_code": "publication_target_not_found",
                                    "retryable": False,
                                }
                            },
                        }
                finally:
                    inner_self.closed = True

        agent = NoProgressAgent()
        events = [
            event
            async for event in stream_research_events_with_agent(
                agent,
                "publish missing article",
                "tool-loop-thread",
            )
        ]

        self.assertTrue(agent.closed)
        self.assertEqual(
            sum(event.get("type") == "tool_start" for event in events),
            1,
        )
        error = next(event for event in events if event["type"] == "error")
        self.assertEqual(
            error["code"],
            "publication_target_not_found",
        )
        self.assertEqual(
            events[-1]["goal_status"],
            "completed_with_failure",
        )

    async def test_repairable_publication_failure_allows_one_repair_then_stops(self):
        class RepairStillFailsAgent:
            def __init__(self):
                self.closed = False

            async def aget_state(self, config):
                return FakeSnapshot()

            async def astream_events(
                inner_self,
                input_data,
                config,
                context,
                version,
            ):
                try:
                    for index in range(3):
                        yield {
                            "event": "on_tool_start",
                            "name": "request_publication_approval",
                            "run_id": f"repair-call-{index}",
                            "data": {"input": {"target_hint": "article"}},
                        }
                        yield {
                            "event": "on_tool_end",
                            "name": "request_publication_approval",
                            "run_id": f"repair-call-{index}",
                            "data": {
                                "output": {
                                    "ok": False,
                                    "error_code": "xiaohongshu_title_too_long",
                                    "retryable": False,
                                    "repairable": True,
                                    "repair_action": "revise_article_for_publication",
                                    "field": "title",
                                    "max_length": 20,
                                }
                            },
                        }
                finally:
                    inner_self.closed = True

        agent = RepairStillFailsAgent()
        events = [
            event
            async for event in stream_research_events_with_agent(
                agent,
                "publish article",
                "repairable-tool-thread",
            )
        ]

        self.assertTrue(agent.closed)
        self.assertEqual(
            sum(event.get("type") == "tool_start" for event in events),
            2,
        )
        error = next(event for event in events if event["type"] == "error")
        self.assertEqual(error["code"], "publication_repair_exhausted")
        self.assertEqual(events[-1]["goal_status"], "completed_with_failure")


if __name__ == "__main__":
    unittest.main()
