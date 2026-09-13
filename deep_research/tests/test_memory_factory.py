import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

import deep_research.agent.factory as factory_module
from deep_research.context import ResearchContext


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class FakeMemoryService:
    def __init__(self) -> None:
        self.user_calls = 0
        self.related_calls = 0
        self.archive_list_calls = 0

    async def list_active_memories(self, **kwargs: object) -> list[object]:
        self.user_calls += 1
        return []

    async def find_review_memories(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[object]:
        self.related_calls += 1
        return []

    async def search_summary_archives(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[object]:
        return []

    async def list_recallable_summary_archives(
        self,
        *,
        limit: int,
    ) -> list[object]:
        self.archive_list_calls += 1
        return []


class MemoryFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_service_enables_main_recall_middleware(self):
        memory_service = FakeMemoryService()
        fake_model = BindableFakeChatModel(
            messages=iter(["Main answer"])
        )

        with patch.object(factory_module, "model", fake_model):
            agent = factory_module.build_agent(
                InMemorySaver(),
                memory_service=memory_service,
            )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "A question",
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "memory-factory-thread",
                }
            },
            context=ResearchContext.from_settings(),
        )

        self.assertEqual(
            result["messages"][-1].content,
            "Main answer",
        )
        self.assertEqual(memory_service.user_calls, 2)
        self.assertEqual(memory_service.related_calls, 0)
        self.assertEqual(memory_service.archive_list_calls, 1)


if __name__ == "__main__":
    unittest.main()
