# 本步目标
# 验证真实 HTTP 流式接口能发送：
# {
#   "type": "plan_update",
#   "todos": [...]
# }
# 并保留：
# tool_start
# tool_end
# text
# done
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

import deep_research.agent as agent_module
import deep_research.main as main_module
from deep_research.config import settings
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


class FakeRagService:
    def close(self) -> None:
        pass


class PlanUpdateHttpTests(unittest.TestCase):
    def test_stream_endpoint_emits_plan_update(self):
        original_path = settings.checkpoint_db_path
        original_agent = agent_module.agent

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = (
                Path(temp_dir) / "plan-http.sqlite"
            )

            settings.checkpoint_db_path = str(
                database_path
            )

            def fake_build_agent(
                checkpointer,
                rag_service,
                memory_service=None,
            ):
                return create_agent(
                    model=BindableFakeChatModel(
                        responses=[
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "write_todos",
                                        "args": {
                                            "todos": [
                                                {
                                                    "content": (
                                                        "核对核心概念"
                                                    ),
                                                    "status": (
                                                        "in_progress"
                                                    ),
                                                },
                                                {
                                                    "content": (
                                                        "整理研究结论"
                                                    ),
                                                    "status": (
                                                        "pending"
                                                    ),
                                                },
                                            ]
                                        },
                                        "id": "http-plan-call-1",
                                        "type": "tool_call",
                                    }
                                ],
                            ),
                            AIMessage(
                                content="计划已创建。",
                            ),
                        ]
                    ),
                    tools=[],
                    context_schema=ResearchContext,
                    state_schema=ResearchState,
                    middleware=[
                        TodoListMiddleware(
                            system_prompt=(
                                TODO_SYSTEM_PROMPT
                            ),
                            tool_description=(
                                TODO_TOOL_DESCRIPTION
                            ),
                        )
                    ],
                    checkpointer=checkpointer,
                )

            thread_id = (
                "33333333-3333-3333-3333-333333333333"
            )

            try:
                with patch.object(
                    agent_module,
                    "build_agent",
                    side_effect=fake_build_agent,
                ), patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=FakeRagService(),
                ):
                    with TestClient(
                        main_module.app
                    ) as client:
                        response = client.post(
                            "/research/stream",
                            json={
                                "question": "测试流式研究计划",
                                "thread_id": thread_id,
                            },
                        )

                self.assertEqual(
                    response.status_code,
                    200,
                )

                events = [
                    json.loads(line)
                    for line in response.text.splitlines()
                    if line.strip()
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
                    "done",
                    event_types,
                )

                plan_event = next(
                    event
                    for event in events
                    if event["type"] == "plan_update"
                )

                self.assertEqual(
                    plan_event["todos"][0]["status"],
                    "in_progress",
                )
                self.assertEqual(
                    plan_event["todos"][1]["status"],
                    "pending",
                )

            finally:
                settings.checkpoint_db_path = (
                    original_path
                )
                agent_module.agent = original_agent


if __name__ == "__main__":
    unittest.main()
