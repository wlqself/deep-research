import asyncio
import importlib
import unittest

from langchain.tools import ToolRuntime

from deep_research.context import ResearchContext


web_search_module = importlib.import_module(
    "deep_research.tools.web_search"
)
read_page_module = importlib.import_module(
    "deep_research.tools.read_page"
)


def make_runtime(
    context: ResearchContext,
    state: dict[str, object] | None = None,
) -> ToolRuntime:
    return ToolRuntime(
        state=state or {},
        context=context,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="budget-test-call",
        store=None,
    )


class RejectingSearchContext(ResearchContext):
    def __init__(self):
        super().__init__(
            max_search_calls=4,
            max_page_reads=6,
            max_page_chars=1000,
        )
        self.search_slot_requests = 0

    def try_acquire_search_slot(self) -> bool:
        self.search_slot_requests += 1
        return False


class RejectingPageReadContext(ResearchContext):
    def __init__(self):
        super().__init__(
            max_search_calls=4,
            max_page_reads=6,
            max_page_chars=1000,
        )
        self.page_read_slot_requests = 0

    def try_acquire_page_read_slot(self) -> bool:
        self.page_read_slot_requests += 1
        return False


class ResearchBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_search_slots_never_exceed_limit(self):
        context = ResearchContext(
            max_search_calls=4,
            max_page_reads=6,
            max_page_chars=1000,
        )

        results = await asyncio.gather(
            *[
                asyncio.to_thread(context.try_acquire_search_slot)
                for _ in range(10)
            ]
        )

        self.assertEqual(sum(results), 4)
        self.assertEqual(context.search_count, 4)
        self.assertLessEqual(
            context.search_count,
            context.max_search_calls,
        )

    async def test_concurrent_page_read_slots_never_exceed_limit(self):
        context = ResearchContext(
            max_search_calls=4,
            max_page_reads=6,
            max_page_chars=1000,
        )

        results = await asyncio.gather(
            *[
                asyncio.to_thread(context.try_acquire_page_read_slot)
                for _ in range(10)
            ]
        )

        self.assertEqual(sum(results), 6)
        self.assertEqual(context.page_read_count, 6)
        self.assertLessEqual(
            context.page_read_count,
            context.max_page_reads,
        )

    async def test_concurrent_source_registration_uses_unique_ids(self):
        context = ResearchContext(
            max_search_calls=4,
            max_page_reads=6,
            max_page_chars=1000,
        )

        sources = await asyncio.gather(
            *[
                asyncio.to_thread(
                    context.register_source,
                    f"Source {index}",
                    f"https://example.com/{index}",
                    "test snippet",
                )
                for index in range(10)
            ]
        )

        source_ids = {source["source_id"] for source in sources}

        self.assertEqual(source_ids, {f"S{index}" for index in range(1, 11)})
        self.assertEqual(set(context.new_sources), source_ids)
        self.assertEqual(context.next_source_number, 11)

    def test_web_search_uses_atomic_budget_method(self):
        context = RejectingSearchContext()
        original_key = web_search_module.settings.tavily_api_key
        web_search_module.settings.tavily_api_key = "test-key"

        try:
            result = web_search_module.web_search.func(
                "test query",
                make_runtime(context),
            )
        finally:
            web_search_module.settings.tavily_api_key = original_key

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "search_budget_exhausted")
        self.assertEqual(context.search_slot_requests, 1)
        self.assertEqual(context.search_count, 0)

    async def test_read_page_uses_atomic_budget_method(self):
        context = RejectingPageReadContext()
        state = {
            "sources": {
                "S1": {
                    "source_id": "S1",
                    "title": "Source one",
                    "url": "https://example.com/source-one",
                    "snippet": "test snippet",
                }
            }
        }

        result = await read_page_module.read_page.coroutine(
            "S1",
            make_runtime(context, state),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "page_read_budget_exhausted")
        self.assertEqual(context.page_read_slot_requests, 1)
        self.assertEqual(context.page_read_count, 0)


if __name__ == "__main__":
    unittest.main()
