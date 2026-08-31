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
    
    async def astream(
        self,
        input_data,
        config,
        context,
        stream_mode,
    ):
        yield (
            FakeToken("结论 [S1]，伪造引用 [S99]"),
            {"langgraph_node": "model"},
        )
        yield (
            FakeToken("。"),
            {"langgraph_node": "model"},
        )


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


class StreamCitationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_appends_verified_sources(self):
        original_agent = agent_module.agent
        original_context = agent_module.ResearchContext

        agent_module.agent = FakeAgent()
        agent_module.ResearchContext = FakeContextFactory

        try:
            chunks = [
                chunk
                async for chunk in agent_module.stream_research(
                    "test question", "test-thread"
                )
            ]
        finally:
            agent_module.agent = original_agent
            agent_module.ResearchContext = original_context

        answer = "".join(chunks)

        self.assertIn("结论 [S1]，伪造引用 [S99]。", answer)
        self.assertIn(
            "来源\n[S1] Test source - https://example.com/test",
            answer,
        )
        self.assertNotIn("\n[S99] ", answer)
        self.assertEqual(answer.count("结论"), 1)


if __name__ == "__main__":
    unittest.main()