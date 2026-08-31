import unittest

import deep_research.agent as agent_module
from deep_research.context import ResearchContext


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


if __name__ == "__main__":
    unittest.main()
