import json
import unittest

from langchain_core.messages import ToolMessage

from deep_research.agent.streaming import (
    stream_research_events_with_agent,
)


class FakeSnapshot:
    values = {
        "sources": {},
    }


class FakeToken:
    def __init__(self, text: str):
        self.text = text


class FakeAgent:
    async def aget_state(self, config):
        return FakeSnapshot()

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        yield {
            "event": "on_tool_start",
            "name": "save_report",
            "data": {
                "input": {
                    "title": "Test report",
                    "content": "# Test",
                }
            },
        }

        yield {
            "event": "on_tool_end",
            "name": "save_report",
            "data": {
                "output": ToolMessage(
                    content=json.dumps(
                        {
                            "ok": True,
                            "artifact_id": (
                                "abcdef1234567890"
                                "abcdef1234567890"
                            ),
                            "filename": (
                                "Test-report-abcdef123456.md"
                            ),
                            "workspace_path": (
                                "/final/"
                                "Test-report-abcdef123456.md"
                            ),
                            "download_url": (
                                "/artifacts/thread/"
                                "abcdef1234567890"
                                "abcdef1234567890"
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id="save-report-1",
                )
            },
        }

        yield {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {
                "langgraph_node": "model",
            },
            "data": {
                "chunk": FakeToken(
                    "报告已保存。"
                ),
            },
        }


class StreamArtifactTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_artifact_saved_event_is_emitted(self):
        events = [
            event
            async for event in (
                stream_research_events_with_agent(
                    FakeAgent(),
                    "保存报告",
                    "artifact-thread",
                )
            )
        ]

        event_types = [
            event["type"]
            for event in events
        ]

        self.assertEqual(
            event_types,
            [
                "tool_start",
                "tool_end",
                "artifact_saved",
                "text",
                "done",
            ],
        )

        artifact_event = next(
            event
            for event in events
            if event["type"] == "artifact_saved"
        )

        self.assertEqual(
            artifact_event["filename"],
            "Test-report-abcdef123456.md",
        )

        self.assertEqual(
            artifact_event["artifact_id"],
            (
                "abcdef1234567890"
                "abcdef1234567890"
            ),
        )

        self.assertNotIn(
            "content",
            artifact_event,
        )


if __name__ == "__main__":
    unittest.main()
