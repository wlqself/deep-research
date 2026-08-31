# 测试 SQLite 下的线程文件隔离
# 本步让两个线程写入同一个虚拟路径：
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import deep_research.agent.factory as factory_module
from deep_research.context import ResearchContext


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class SQLiteFilesystemIsolationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_same_path_isolated_between_threads(self):
        calls = [
            {
                "name": "write_file",
                "args": {
                    "file_path": "/drafts/current-report.md",
                    "content": "thread A",
                },
            },
            {
                "name": "write_file",
                "args": {
                    "file_path": "/drafts/current-report.md",
                    "content": "thread B",
                },
            },
            {
                "name": "read_file",
                "args": {
                    "file_path": "/drafts/current-report.md",
                },
            },
            {
                "name": "read_file",
                "args": {
                    "file_path": "/drafts/current-report.md",
                },
            },
        ]

        model_messages = []

        for index, call in enumerate(calls):
            model_messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": call["name"],
                            "args": call["args"],
                            "id": f"isolation-call-{index}",
                            "type": "tool_call",
                        }
                    ],
                )
            )
            model_messages.append(
                AIMessage(content="操作完成")
            )

        fake_model = BindableFakeChatModel(
            messages=iter(model_messages)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(
                Path(temp_dir) / "checkpoints.sqlite"
            )

            async with AsyncSqliteSaver.from_conn_string(
                database_path
            ) as saver:
                with patch.object(
                    factory_module,
                    "model",
                    fake_model,
                ):
                    agent = factory_module.build_agent(
                        saver
                    )

                config_a = {
                    "configurable": {
                        "thread_id": "filesystem-thread-a",
                    }
                }

                config_b = {
                    "configurable": {
                        "thread_id": "filesystem-thread-b",
                    }
                }

                await agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "写入线程 A 草稿",
                            }
                        ]
                    },
                    config=config_a,
                    context=ResearchContext.from_settings(),
                )

                await agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "写入线程 B 草稿",
                            }
                        ]
                    },
                    config=config_b,
                    context=ResearchContext.from_settings(),
                )

                result_a = await agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "读取线程 A 草稿",
                            }
                        ]
                    },
                    config=config_a,
                    context=ResearchContext.from_settings(),
                )

                result_b = await agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "读取线程 B 草稿",
                            }
                        ]
                    },
                    config=config_b,
                    context=ResearchContext.from_settings(),
                )

        file_path = "/drafts/current-report.md"

        self.assertEqual(
            result_a["files"][file_path]["content"],
            "thread A",
        )

        self.assertEqual(
            result_b["files"][file_path]["content"],
            "thread B",
        )

        self.assertIn(
            "thread A",
            result_a["messages"][-2].content,
        )

        self.assertIn(
            "thread B",
            result_b["messages"][-2].content,
        )


if __name__ == "__main__":
    unittest.main()