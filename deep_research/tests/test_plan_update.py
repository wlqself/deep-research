import unittest

import deep_research.agent as agent_module
from deep_research.context import ResearchContext


class FakeToken:
    def __init__(self, text: str):
        self.text = text


class FakeSnapshot:
    values = {}


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
            "name": "write_todos",
            "data": {
                "input": {
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
                }
            },
        }

        yield {
            "event": "on_tool_end",
            "name": "write_todos",
            "data": {
                "output": {
                    "ok": True,
                }
            },
        }

        yield {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {
                "langgraph_node": "model",
            },
            "data": {
                "chunk": FakeToken("计划已创建。"),
            },
        }


class PlanUpdateTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_plan_update_event_is_emitted(self):
        original_agent = agent_module.agent

        agent_module.agent = FakeAgent()

        try:
            events = [
                event
                async for event in agent_module.stream_research_events(
                    "测试计划",
                    "plan-thread",
                )
            ]
        finally:
            agent_module.agent = original_agent

        event_types = [
            event["type"]
            for event in events
        ]

        self.assertEqual(
            event_types,
            [
                "plan_update",
                "tool_start",
                "tool_end",
                "text",
                "done",
            ],
        )

        self.assertEqual(
            events[0],
            {
                "type": "plan_update",
                "todos": [
                    {
                        "content": "核对核心概念",
                        "status": "in_progress",
                    },
                    {
                        "content": "整理研究结论",
                        "status": "pending",
                    },
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
