# 验证真实 create_agent + TodoListMiddleware + astream_events 链路能够产生：
# plan_update
# tool_start
# tool_end
# text
# done

import unittest

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.streaming import (
    stream_research_events_with_agent,
)
from deep_research.context import ResearchContext
from deep_research.prompts.research import (
    TODO_SYSTEM_PROMPT,
    TODO_TOOL_DESCRIPTION,
)
from deep_research.state import ResearchState


class BindableFakeChatModel(
    FakeMessagesListChatModel
):
    def bind_tools(self, tools, **kwargs):
        return self


class PlanUpdateIntegrationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_real_agent_stream_emits_plan_update(self):
        fake_model = BindableFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_todos",
                            "args": {
                                "todos": [
                                    {
                                        "content": "核对核心概念",
                                        "status": "in_progress",
                                    },
                                    {
                                        "content": "整理研究结论",
                                        "status": "pending",
                                    },
                                ]
                            },
                            "id": "stream-todo-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="计划已创建。",
                ),
            ]
        )

        graph = create_agent(
            model=fake_model,
            tools=[],
            context_schema=ResearchContext,
            state_schema=ResearchState,
            middleware=[
                TodoListMiddleware(
                    system_prompt=TODO_SYSTEM_PROMPT,
                    tool_description=TODO_TOOL_DESCRIPTION,
                )
            ],
            checkpointer=InMemorySaver(),
        )

        events = [
            event
            async for event in stream_research_events_with_agent(
                graph,
                "请制定研究计划",
                "stream-plan-thread",
            )
        ]

        event_types = [
            event["type"]
            for event in events
        ]

        self.assertIn(
            "plan_update",
            event_types,
        )
        self.assertIn(
            "tool_start",
            event_types,
        )
        self.assertIn(
            "tool_end",
            event_types,
        )
        self.assertIn(
            "done",
            event_types,
        )

        plan_events = [
            event
            for event in events
            if event["type"] == "plan_update"
        ]

        self.assertEqual(
            plan_events[0]["todos"],
            [
                {
                    "content": "核对核心概念",
                    "status": "in_progress",
                },
                {
                    "content": "整理研究结论",
                    "status": "pending",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()