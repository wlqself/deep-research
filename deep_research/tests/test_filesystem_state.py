# 验证：
# - Agent 可以写入 /notes/test.md；
# - 文件进入 checkpoint 的 files state；
# - 同一线程可以看到文件；
# - 不同线程看不到该文件。
# StateBackend 的文件会保存在当前线程的 LangGraph state 中，并随 checkpoint 跨轮次保留。
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


class FileStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_files_are_thread_scoped(self):
        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {
                                    "file_path": "/notes/test.md",
                                    "content": "hello",
                                },
                                "id": "write-file-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="saved A"),
                    AIMessage(content="empty B"),
                ]
            )
        )

        saver = InMemorySaver()

        with patch.object(
            factory_module,
            "model",
            fake_model,
        ):
            agent = factory_module.build_agent(saver)

        result_a = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "保存一条研究笔记",
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "filesystem-thread-a",
                }
            },
            context=ResearchContext.from_settings(),
        )

        result_b = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "查看当前文件",
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "filesystem-thread-b",
                }
            },
            context=ResearchContext.from_settings(),
        )

        self.assertEqual(
            result_a["files"]["/notes/test.md"]["content"],
            "hello",
        )

        self.assertEqual(
            result_b.get("files", {}),
            {},
        )


if __name__ == "__main__":
    unittest.main()