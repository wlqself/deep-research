import unittest

from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage

from deep_research.agent.results import ResearcherResult


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ResearcherResultTests(unittest.TestCase):
    def test_researcher_result_is_returned_through_task_tool(self):
        supervisor_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": (
                                        "执行 task-001，"
                                        "研究当前主题并返回结构化结果。"
                                    ),
                                    "subagent_type": "researcher",
                                },
                                "id": "task-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="supervisor final answer",
                    ),
                ]
            )
        )

        researcher_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ResearcherResult",
                                "args": {
                                    "summary": "本次任务完成了来源核查。",
                                    "finding_ids": ["F1"],
                                    "source_ids": ["S1", "S2"],
                                    "knowledge_gaps": [
                                        "缺少长期趋势数据",
                                    ],
                                    "conflicts": [],
                                    "recommended_action": "answer",
                                },
                                "id": "structured-result-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            )
        )

        researcher = {
            "name": "researcher",
            "description": "完成委派的研究任务并返回结构化结果。",
            "system_prompt": (
                "执行委派任务。最终必须返回 ResearcherResult。"
            ),
            "model": researcher_model,
            "tools": [],
            "response_format": ToolStrategy(
                ResearcherResult
            ),
        }

        graph = create_agent(
            model=supervisor_model,
            tools=[],
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend(),
                    subagents=[researcher],
                )
            ],
        )

        result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请完成研究任务。",
                    }
                ]
            }
        )

        task_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]

        self.assertEqual(len(task_messages), 1)

        parsed = ResearcherResult.model_validate_json(
            task_messages[0].content
        )

        self.assertEqual(
            parsed.summary,
            "本次任务完成了来源核查。",
        )
        self.assertEqual(parsed.finding_ids, ["F1"])
        self.assertEqual(parsed.source_ids, ["S1", "S2"])
        self.assertEqual(
            parsed.recommended_action,
            "answer",
        )

        self.assertNotIn(
            "structured_response",
            result,
        )

        self.assertEqual(
            result["messages"][-1].content,
            "supervisor final answer",
        )


if __name__ == "__main__":
    unittest.main()
