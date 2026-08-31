# 本步验证同一线程内的文件操作能够连续工作：
# 1. write_file
# 2. read_file
# 3. edit_file
# 4. glob
# 5. ls
# 所有操作仍使用 Fake Chat Model，不消耗真实 API。
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


class FilesystemToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_filesystem_tools_persist_in_one_thread(self):
        calls = [
            {
                "name": "write_file",
                "args": {
                    "file_path": "/notes/research.md",
                    "content": "first line",
                },
            },
            {
                "name": "read_file",
                "args": {
                    "file_path": "/notes/research.md",
                },
            },
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/notes/research.md",
                    "old_string": "first line",
                    "new_string": "first line\nsecond line",
                    "replace_all": False,
                },
            },
            {
                "name": "glob",
                "args": {
                    "pattern": "*.md",
                    "path": "/notes",
                },
            },
            {
                "name": "ls",
                "args": {
                    "path": "/notes",
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
                            "id": f"filesystem-call-{index}",
                            "type": "tool_call",
                        }
                    ],
                )
            )
            model_messages.append(
                AIMessage(
                    content=f"完成 {call['name']}",
                )
            )

        fake_model = BindableFakeChatModel(
            messages=iter(model_messages)
        )

        saver = InMemorySaver()

        with patch.object(
            factory_module,
            "model",
            fake_model,
        ):
            agent = factory_module.build_agent(saver)

        results = []

        for call in calls:
            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": call["name"],
                        }
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": "filesystem-tools-thread",
                    }
                },
                context=ResearchContext.from_settings(),
            )
            results.append(result)

        self.assertEqual(
            results[0]["files"]["/notes/research.md"]["content"],
            "first line",
        )

        self.assertIn(
            "first line",
            results[1]["messages"][-2].content,
        )

        self.assertEqual(
            results[2]["files"]["/notes/research.md"]["content"],
            "first line\nsecond line",
        )

        self.assertIn(
            "/notes/research.md",
            results[3]["messages"][-2].content,
        )

        self.assertIn(
            "/notes/research.md",
            results[4]["messages"][-2].content,
        )


if __name__ == "__main__":
    unittest.main()