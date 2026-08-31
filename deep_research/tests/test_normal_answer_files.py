# 测试普通回答不会自动创建文件
# 这里要区分两件事：
# - 普通回答没有调用文件工具时，不应该产生文件；
# - 如果模型主动调用 write_file("/final/...")，当前工具层仍然可能执行。
# 第二点会在 V0.3-D 的 save_report 中处理。提示词不能代替工具权限控制。

import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import deep_research.agent.factory as factory_module
from deep_research.context import ResearchContext


class BindableFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class NormalAnswerTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_normal_answer_does_not_create_files(self):
        fake_model = BindableFakeChatModel(
            responses=[
                AIMessage(
                    content="这是普通研究回答。"
                )
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

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "请解释 LangGraph 和 "
                            "LangChain 的区别"
                        ),
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "normal-answer-thread",
                }
            },
            context=ResearchContext.from_settings(),
        )

        self.assertEqual(
            result.get("files", {}),
            {},
        )

        self.assertEqual(
            result.get("findings", {}),
            {},
        )

        self.assertEqual(
            result["messages"][-1].content,
            "这是普通研究回答。",
        )


if __name__ == "__main__":
    unittest.main()
