import json
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


class FindingsIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_persists_finding_in_checkpoint(self):
        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": (
                                        "执行 task-001，记录重要 finding。"
                                    ),
                                    "subagent_type": "researcher",
                                },
                                "id": "task-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "record_research_finding",
                                "args": {
                                    "claim": "A verified claim",
                                    "evidence_summary": "Evidence from S1.",
                                    "source_ids": ["S1"],
                                    "status": "supported",
                                    "uncertainty": "",
                                    "conflicts": "",
                                },
                                "id": "finding-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ResearcherResult",
                                "args": {
                                    "summary": "完成了 finding 记录。",
                                    "finding_ids": [],
                                    "source_ids": ["S1"],
                                    "knowledge_gaps": [],
                                    "conflicts": [],
                                    "recommended_action": "answer",
                                },
                                "id": "result-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Final answer [S1]"),
                ]
            )
        )

        with patch.object(factory_module, "model", fake_model):
            agent = factory_module.build_agent(InMemorySaver())

        config = {
            "configurable": {
                "thread_id": "finding-integration-thread",
            }
        }

        await agent.aupdate_state(
            config,
            {
                "sources": {
                    "S1": {
                        "source_id": "S1",
                        "title": "Known source",
                        "url": "https://example.com/source",
                        "snippet": "Known evidence",
                    }
                }
            },
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Research this claim.",
                    }
                ]
            },
            config=config,
            context=ResearchContext.from_settings(),
        )

        self.assertEqual(len(result["findings"]), 1)

        finding_id = next(iter(result["findings"]))
        finding = result["findings"][finding_id]

        self.assertEqual(finding["claim"], "A verified claim")
        self.assertEqual(finding["source_ids"], ["S1"])

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
            and message.tool_call_id == "task-call-1"
        ]

        self.assertEqual(len(tool_messages), 1)

        payload = json.loads(tool_messages[0].content)
        self.assertEqual(payload["source_ids"], ["S1"])

        snapshot = await agent.aget_state(config)
        self.assertIn(finding_id, snapshot.values["findings"])


if __name__ == "__main__":
    unittest.main()
