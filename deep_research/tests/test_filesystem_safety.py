# 本步测试：
# - ../ 路径穿越被拒绝；
# - Windows 宿主机绝对路径被拒绝；
# - /.env 只能被当作虚拟 workspace 路径，不会读取项目根目录的 .env；
# - 失败操作不会写入 files state。
# StateBackend 不访问宿主机目录，文件只保存在当前线程 state 中。
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


class FilesystemSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def invoke_tool(
        self,
        tool_name: str,
        args: dict[str, object],
        thread_id: str,
    ):
        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": tool_name,
                                "args": args,
                                "id": "safety-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="检查完成"),
                ]
            )
        )

        with patch.object(
            factory_module,
            "model",
            fake_model,
        ):
            agent = factory_module.build_agent(
                InMemorySaver()
            )

        return await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "执行安全检查",
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": thread_id,
                }
            },
            context=ResearchContext.from_settings(),
        )

    async def test_virtual_workspace_rejects_escape_and_env_access(
        self,
    ):
        cases = [
            (
                "write_file",
                {
                    "file_path": "../outside.md",
                    "content": "x",
                },
            ),
            (
                "write_file",
                {
                    "file_path": "/../outside.md",
                    "content": "x",
                },
            ),
            (
                "write_file",
                {
                    "file_path": "C:/temp/outside.md",
                    "content": "x",
                },
            ),
            (
                "read_file",
                {
                    "file_path": "/.env",
                },
            ),
            (
                "read_file",
                {
                    "file_path": "E:/my_agent/old coding/.env",
                },
            ),
        ]

        for index, (tool_name, args) in enumerate(cases):
            result = await self.invoke_tool(
                tool_name,
                args,
                f"safety-thread-{index}",
            )

            self.assertEqual(
                result.get("files", {}),
                {},
            )

            self.assertIn(
                "Error:",
                result["messages"][-2].content,
            )


if __name__ == "__main__":
    unittest.main()