import unittest
import json

from langgraph.types import Command
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage


class BindableFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class SubAgentEventMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_subagent_event_metadata(self):
        model = BindableFakeChatModel(
            responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "执行一个子任务。",
                                    "subagent_type": "researcher",
                                },
                                "id": "task-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="researcher internal answer"),
                    AIMessage(content="supervisor final answer"),
                ]
        )

        researcher = {
            "name": "researcher",
            "description": "Runs one delegated task.",
            "system_prompt": "Return a concise result.",
            "model": model,
            "tools": [],
        }

        graph = create_agent(
            model=model,
            tools=[],
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend(),
                    subagents=[researcher],
                )
            ],
            checkpointer=InMemorySaver(),
        )

        captured = []

        config = {"configurable": {"thread_id": "subagent-event-metadata"}}

        async for event in graph.astream_events(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "开始研究。",
                    }
                ]
            },
            config = config,
            version="v2",
        ):
            if event.get("event") in {
                "on_chat_model_start",
                "on_chat_model_stream",
                "on_chat_model_end",
                "on_tool_start",
                "on_tool_end",
            }:
                captured.append(
                    {
                        "event": event.get("event"),
                        "name": event.get("name"),
                        "metadata": event.get("metadata"),
                        "tags": event.get("tags"),
                        "parent_ids": event.get("parent_ids"),
                        "run_id": str(event.get("run_id", "")),
                        "parent_ids": list(event.get("parent_ids", [])),
                        "metadata": event.get("metadata", {}),
                        "data": event.get("data", {}),
                    }
                )

        self.assertTrue(captured)

        task_events = [
            item
            for item in captured
            if item["name"] == "task"
        ]
        self.assertEqual(
            [item["event"] for item in task_events],
            ["on_tool_start", "on_tool_end"],
        )

        child_model_events = [
            item
            for item in captured
            if item["metadata"].get("lc_agent_name") == "researcher"
        ]
        self.assertTrue(child_model_events)

        parent_model_events = [
            item
            for item in captured
            if item["event"].startswith("on_chat_model_")
            and item["metadata"].get("lc_agent_name") != "researcher"
        ]
        self.assertTrue(parent_model_events)

        for item in child_model_events:
            self.assertGreaterEqual(len(item["parent_ids"]), 3)

        snapshot = await graph.aget_state(config)
        task_message = next(
            message
            for message in snapshot.values["messages"]
            if isinstance(message, ToolMessage)
        )

        task_end = next(
            item
            for item in task_events
            if item["event"] == "on_tool_end"
        )

        output = task_end["data"]["output"]
        self.assertIsInstance(output, Command)

        returned_message = next(
            message
            for message in output.update["messages"]
            if isinstance(message, ToolMessage)
        )

        self.assertEqual(returned_message.tool_call_id, "task-call-1")

if __name__ == "__main__":
    unittest.main()
