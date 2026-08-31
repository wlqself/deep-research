import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import deep_research.agent.factory as factory_module
from deep_research.context import ResearchContext


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class SupervisorRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_simple_question_does_not_call_task(self):
        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="这是一个简单问题的直接回答。",
                    )
                ]
            )
        )

        with patch.object(factory_module, "model", fake_model):
            agent = factory_module.build_agent(InMemorySaver())

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请把这句话翻译成英文。",
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "simple-routing-thread",
                }
            },
            context=ResearchContext.from_settings(),
        )

        task_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
            and message.tool_call_id == "task-call-1"
        ]

        self.assertEqual(task_messages, [])
        self.assertEqual(
            result["messages"][-1].content,
            "这是一个简单问题的直接回答。",
        )


if __name__ == "__main__":
    unittest.main()
