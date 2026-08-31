import unittest

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langgraph.checkpoint.memory import InMemorySaver

#作用：这是一个 LangChain 提供的假模型。
#行为：它不联网，只是按顺序吐出列表里的内容。
class ThreadMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_messages_resume_and_threads_are_isolated(self):
        model = GenericFakeChatModel(
            messages=iter([
                "第一轮回答",
                "第二轮回答",
                "隔离线程回答",
            ])
        )
        test_agent = create_agent(
            model=model,
            tools=[],
            checkpointer=InMemorySaver(),
        )

        thread_a = {
            "configurable": {"thread_id": "thread-a"}
        }
        thread_b = {
            "configurable": {"thread_id": "thread-b"}
        }

        await test_agent.ainvoke(
            {"messages": [{"role": "user", "content": "第一轮"}]},
            config=thread_a,
        )
        resumed = await test_agent.ainvoke(
            {"messages": [{"role": "user", "content": "第二轮"}]},
            config=thread_a,
        )
        isolated = await test_agent.ainvoke(
            {"messages": [{"role": "user", "content": "另一线程"}]},
            config=thread_b,
        )

        self.assertEqual(
            [message.content for message in resumed["messages"]],
            ["第一轮", "第一轮回答", "第二轮", "第二轮回答"],
        )
        self.assertEqual(
            [message.content for message in isolated["messages"]],
            ["另一线程", "隔离线程回答"],
        )


if __name__ == "__main__":
    unittest.main()