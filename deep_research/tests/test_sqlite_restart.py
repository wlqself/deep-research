import tempfile
import unittest
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

#验证：SQLite 文件是“跨连接存活”的，而不是“随 saver 生命周期销毁”的。

class SQLiteRestartTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_thread_state_survives_saver_restart(self):
        # 模拟真实数据库文件，tempfile 指的是临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(
                Path(temp_dir) / "checkpoints.sqlite"
            )

            config = {
                "configurable": {
                    "thread_id": "restart-thread",
                }
            }

            async with AsyncSqliteSaver.from_conn_string(
                db_path
            ) as first_saver:
                first_graph = create_agent(
                    model=GenericFakeChatModel(
                        messages=iter(["第一轮回答"])
                    ),
                    tools=[],
                    checkpointer=first_saver,
                )

                await first_graph.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "第一轮问题",
                            }
                        ]
                    },
                    config=config,
                )
            # 这是全新的 saver 实例，不是复用
            async with AsyncSqliteSaver.from_conn_string(
                db_path
            ) as second_saver:
                second_graph = create_agent(
                    model=GenericFakeChatModel(
                        messages=iter(["第二轮回答"])
                    ),
                    tools=[],
                    checkpointer=second_saver,
                )

                result = await second_graph.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "第二轮问题",
                            }
                        ]
                    },
                    config=config,
                )

            contents = [
                message.content
                for message in result["messages"]
            ]
            # 断言 —— 证明“历史没丢”
            self.assertEqual(
                contents,
                [
                    "第一轮问题",
                    "第一轮回答",
                    "第二轮问题",
                    "第二轮回答",
                ],
            )


if __name__ == "__main__":
    unittest.main()