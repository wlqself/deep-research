# - 创建出来的 Agent，确实包含：
# - 自定义 Todo Middleware
# - 自动生成的 write_todos
# - Todo state
# - 现有研究工具
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import deep_research.agent.factory as factory_module
from deep_research.context import ResearchContext


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class TodoFactoryTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_factory_agent_supports_write_todos(self):
        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "核对研究范围",
                                            "status": "in_progress",
                                        },
                                        {
                                            "content": "整理研究结论",
                                            "status": "pending",
                                        },
                                    ]
                                },
                                "id": "factory-todo-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="计划已创建。",
                    ),
                ]
            )
        )

        with patch.object(
            factory_module,
            "model",
            fake_model,
        ):
            graph = factory_module.build_agent(
                InMemorySaver()
            )

        config = {
            "configurable": {
                "thread_id": "factory-todo-thread",
            }
        }

        result = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请制定一个研究计划",
                    }
                ]
            },
            config=config,
            context=ResearchContext.from_settings(),
        )

        self.assertEqual(
            result["todos"],
            [
                {
                    "content": "核对研究范围",
                    "status": "in_progress",
                },
                {
                    "content": "整理研究结论",
                    "status": "pending",
                },
            ],
        )

        self.assertEqual(
            result["messages"][-1].content,
            "计划已创建。",
        )


if __name__ == "__main__":
    unittest.main()