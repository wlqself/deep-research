# 本步验证：
# 1. 第一个 Agent 写入 /drafts/current-report.md；
# 2. 关闭第一个 AsyncSqliteSaver；
# 3. 使用同一个 SQLite 文件创建第二个 Agent；
# 4. 使用同一个 thread_id 读取之前的草稿。
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


class SQLiteFilesystemRestartTests(
    unittest.IsolatedAsyncioTestCase
):
    @staticmethod
    def model_for(
        tool_name: str,
        tool_args: dict[str, object],
        final_text: str,
    ):
        return BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": tool_name,
                                "args": tool_args,
                                "id": "restart-file-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content=final_text),
                ]
            )
        )

    async def test_files_survive_sqlite_saver_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(
                Path(temp_dir) / "checkpoints.sqlite"
            )

            config = {
                "configurable": {
                    "thread_id": "restart-files-thread",
                }
            }

            first_model = self.model_for(
                "write_file",
                {
                    "file_path": "/drafts/current-report.md",
                    "content": "draft v1",
                },
                "草稿已写入",
            )

            with patch.object(
                factory_module,
                "model",
                first_model,
            ):
                async with AsyncSqliteSaver.from_conn_string(
                    database_path
                ) as first_saver:
                    first_agent = factory_module.build_agent(
                        first_saver
                    )

                    await first_agent.ainvoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "写入当前草稿",
                                }
                            ]
                        },
                        config=config,
                        context=ResearchContext.from_settings(),
                    )

            second_model = self.model_for(
                "read_file",
                {
                    "file_path": "/drafts/current-report.md",
                },
                "草稿已读取",
            )

            with patch.object(
                factory_module,
                "model",
                second_model,
            ):
                async with AsyncSqliteSaver.from_conn_string(
                    database_path
                ) as second_saver:
                    second_agent = factory_module.build_agent(
                        second_saver
                    )

                    result = await second_agent.ainvoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "读取当前草稿",
                                }
                            ]
                        },
                        config=config,
                        context=ResearchContext.from_settings(),
                    )

            self.assertEqual(
                result["files"][
                    "/drafts/current-report.md"
                ]["content"],
                "draft v1",
            )

            self.assertIn(
                "draft v1",
                result["messages"][-2].content,
            )


if __name__ == "__main__":
    unittest.main()