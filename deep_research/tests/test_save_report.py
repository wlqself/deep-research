import json
import unittest
import hashlib

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


class SaveReportTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_save_report_writes_final_markdown(self):
        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "save_report",
                                "args": {
                                    "title": "Test report",
                                    "content": "# Hello\n\nDraft",
                                },
                                "id": "save-report-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="报告已保存。"
                    ),
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

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "保存这份报告",
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "save-report-thread",
                }
            },
            context=ResearchContext.from_settings(),
        )

        files = result["files"]

        artifacts = result["artifacts"]

        self.assertEqual(
            len(artifacts),
            1,
        )

        artifact_id = next(iter(artifacts))

        self.assertEqual(
            len(artifact_id),
            32,
        )

        artifact = artifacts[artifact_id]

        expected_bytes = (
            "# Hello\n\nDraft"
        ).encode("utf-8")

        self.assertEqual(
            artifact["artifact_id"],
            artifact_id,
        )

        self.assertEqual(
            artifact["size_bytes"],
            len(expected_bytes),
        )

        self.assertEqual(
            artifact["sha256"],
            hashlib.sha256(
                expected_bytes,
            ).hexdigest(),
        )

        self.assertTrue(
            artifact["created_at"].endswith(
                "+00:00"
            )
        )

        self.assertEqual(
            len(files),
            1,
        )

        workspace_path = next(iter(files))

        self.assertTrue(
            workspace_path.startswith("/final/")
        )

        self.assertTrue(
            workspace_path.endswith(".md")
        )

        self.assertEqual(
            files[workspace_path]["content"],
            "# Hello\n\nDraft",
        )

        payload = json.loads(
            result["messages"][-2].content
        )

        self.assertTrue(
            payload["ok"]
        )

        self.assertEqual(
            payload["workspace_path"],
            workspace_path,
        )

        self.assertTrue(
            payload["filename"].endswith(".md")
        )

        self.assertIn(
            "/artifacts/save-report-thread/",
            payload["download_url"],
        )
        
# 两次保存生成两个完整 32 位 artifact_id；
# 两个 Artifact 元数据同时存在；
# 两个 /final/*.md 文件同时存在；
# 第二次保存不会覆盖第一次。
    async def test_two_saves_create_two_independent_versions(self):
        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "save_report",
                                "args": {
                                    "title": "同一报告",
                                    "content": "# Version 1",
                                },
                                "id": "save-report-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "save_report",
                                "args": {
                                    "title": "同一报告",
                                    "content": "# Version 2",
                                },
                                "id": "save-report-2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="两份报告都已保存。",
                    ),
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

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "保存两个版本",
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "two-save-thread",
                }
            },
            context=ResearchContext.from_settings(),
        )

        artifacts = result["artifacts"]
        files = result["files"]

        self.assertEqual(
            len(artifacts),
            2,
        )

        self.assertEqual(
            len(files),
            2,
        )

        artifact_ids = list(artifacts)

        self.assertEqual(
            len(set(artifact_ids)),
            2,
        )

        for artifact_id in artifact_ids:
            self.assertEqual(
                len(artifact_id),
                32,
            )

        contents = {
            file_data["content"]
            for file_data in files.values()
        }

        self.assertEqual(
            contents,
            {
                "# Version 1",
                "# Version 2",
            },
        )

if __name__ == "__main__":
    unittest.main()