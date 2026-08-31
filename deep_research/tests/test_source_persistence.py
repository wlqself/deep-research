import unittest
import importlib

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain.tools import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_research.context import ResearchContext
from deep_research.state import ResearchState

web_search_module = importlib.import_module(
    "deep_research.tools.web_search"
)

class FakeSearch:
    results = []

    def __init__(self, **kwargs):
        pass

    def invoke(self, input_data):
        return {
            "results": list(self.results),
        }


class FakeWrapper:
    def __init__(self, **kwargs):
        pass


def make_runtime(
    context: ResearchContext,
    tool_call_id: str,
) -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=None,
    )


class SourcePersistenceTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_sources_append_in_same_thread_and_isolate_threads(
        self,
    ):
        original_search = web_search_module.TavilySearch
        original_wrapper = (
            web_search_module.TavilySearchAPIWrapper
        )
        original_key = (
            web_search_module.settings.tavily_api_key
        )

        web_search_module.TavilySearch = FakeSearch
        web_search_module.TavilySearchAPIWrapper = FakeWrapper
        web_search_module.settings.tavily_api_key = "test-key"

        try:
            graph = create_agent(
                model=GenericFakeChatModel(
                    messages=iter([])
                ),
                tools=[],
                state_schema=ResearchState,
                checkpointer=InMemorySaver(),
            )

            config_a = {
                "configurable": {
                    "thread_id": "thread-a",
                }
            }
            config_b = {
                "configurable": {
                    "thread_id": "thread-b",
                }
            }

            FakeSearch.results = [
                {
                    "title": "Source one",
                    "url": "https://example.com/one",
                    "snippet": "one",
                }
            ]

            context_a = ResearchContext(
                max_search_calls=4,
                max_page_reads=6,
                max_page_chars=12000,
                next_source_number=1,
            )

            first = web_search_module.web_search.func(
                "first query",
                make_runtime(context_a, "call-1"),
            )

            self.assertIsInstance(first, Command)

            await graph.aupdate_state(
                config_a,
                first.update,
            )

            FakeSearch.results = [
                {
                    "title": "Source two",
                    "url": "https://example.com/two",
                    "snippet": "two",
                }
            ]

            context_a_second = ResearchContext(
                max_search_calls=4,
                max_page_reads=6,
                max_page_chars=12000,
                next_source_number=2,
            )

            second = web_search_module.web_search.func(
                "second query",
                make_runtime(
                    context_a_second,
                    "call-2",
                ),
            )

            self.assertIsInstance(second, Command)

            await graph.aupdate_state(
                config_a,
                second.update,
                as_node="model",
            )

            state_a = await graph.aget_state(config_a)
            state_b = await graph.aget_state(config_b)

            self.assertEqual(
                set(state_a.values["sources"]),
                {"S1", "S2"},
            )
            self.assertEqual(
                state_a.values["next_source_number"],
                3,
            )
            self.assertNotIn(
                "sources",
                state_b.values,
            )
        finally:
            web_search_module.TavilySearch = original_search
            web_search_module.TavilySearchAPIWrapper = (
                original_wrapper
            )
            web_search_module.settings.tavily_api_key = (
                original_key
            )


if __name__ == "__main__":
    unittest.main()