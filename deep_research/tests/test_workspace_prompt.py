# 只验证“提示词注入成功”，不测试真实模型是否一定遵守规则
import unittest
from unittest.mock import patch

from pydantic import PrivateAttr
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

import deep_research.agent.factory as factory_module
from deep_research.context import ResearchContext


class CapturingFakeChatModel(FakeMessagesListChatModel):
    _seen_messages: list = PrivateAttr(
        default_factory=list
    )

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ):
        self._seen_messages = list(messages)

        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


def content_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            if isinstance(block, dict)
            else str(block)
            for block in content
        )

    return str(content)


class WorkspacePromptTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_workspace_rules_reach_model_system_message(
        self,
    ):
        fake_model = CapturingFakeChatModel(
            responses=[
                AIMessage(content="ok"),
            ]
        )

        with patch.object(
            factory_module,
            "model",
            fake_model,
        ):
            agent = factory_module.build_agent(
                InMemorySaver()
            )

        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "回答问题",
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "workspace-prompt-thread",
                }
            },
            context=ResearchContext.from_settings(),
        )

        system_messages = [
            message
            for message in fake_model._seen_messages
            if isinstance(message, SystemMessage)
        ]

        self.assertEqual(
            len(system_messages),
            1,
        )

        system_text = content_text(
            system_messages[0].content
        )

        for marker in [
            "/notes/",
            "/drafts/",
            "/final/",
            "宿主机",
            "execute",
        ]:
            self.assertIn(marker, system_text)

        self.assertIn(
            "Main Agent",
            system_text,
        )


if __name__ == "__main__":
    unittest.main()
