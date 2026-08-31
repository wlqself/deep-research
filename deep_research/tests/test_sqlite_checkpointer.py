import unittest

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class SQLiteCheckpointerTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_async_sqlite_persists_thread_state(self):
        async with AsyncSqliteSaver.from_conn_string(  #LangGraph 提供的异步 SQLite 检查点存储器。
            ":memory:"
        ) as checkpointer:
            graph = create_agent(
                model=GenericFakeChatModel(
                    messages=iter(["SQLite answer"])  #把 SQLite 检查点绑定到 Agent 上。
                ),
                tools=[],
                checkpointer=checkpointer,
            )

            config = {
                "configurable": {
                    "thread_id": "sqlite-test-thread",
                }
            }

            await graph.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "test question",
                        }
                    ]
                },
                config=config,
            )

            snapshot = await graph.aget_state(config)

            messages = snapshot.values["messages"]

            self.assertEqual(
                messages[-1].content,
                "SQLite answer",
            )


if __name__ == "__main__":
    unittest.main()