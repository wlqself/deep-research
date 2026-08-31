import unittest

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.state import ResearchState


class ThreadStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_sources_are_persisted_and_isolated(self):
        graph = create_agent(
            model=GenericFakeChatModel(messages=iter([])),
            tools=[],
            state_schema=ResearchState,
            checkpointer=InMemorySaver(),
        )

        thread_a = {
            "configurable": {
                "thread_id": "source-thread-a",
            }
        }
        thread_b = {
            "configurable": {
                "thread_id": "source-thread-b",
            }
        }

        await graph.aupdate_state(
            thread_a,
            {
                "sources": {
                    "S1": {
                        "source_id": "S1",
                        "title": "Thread A source",
                        "url": "https://example.com/a",
                        "snippet": "source A",
                    }
                },
                "next_source_number": 2,
            },
        )

        state_a = await graph.aget_state(thread_a)
        state_b = await graph.aget_state(thread_b)

        self.assertEqual(
            state_a.values["sources"]["S1"]["title"],
            "Thread A source",
        )
        self.assertEqual(
            state_a.values["next_source_number"],
            2,
        )
        self.assertNotIn("sources", state_b.values)
        self.assertNotIn("next_source_number", state_b.values)


if __name__ == "__main__":
    unittest.main()