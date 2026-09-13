import asyncio
import unittest
from types import SimpleNamespace

from langchain.tools import ToolRuntime

from deep_research.tools.publishing import build_analyze_uploaded_image_tool


class _AttachmentService:
    def __init__(self):
        self.repository = SimpleNamespace()

    def get(self, attachment_id):
        return SimpleNamespace(
            attachment_id=attachment_id,
            filename="test.png",
        )


class _AnalysisService:
    def __init__(self):
        self.calls = []

    async def analyze_now(self, attachment_id, *, force=False, focus=None):
        self.calls.append(attachment_id)
        return SimpleNamespace(
            status="completed",
            image_type="question",
            confidence=0.99,
            summary="一道题目",
            ocr_text="1 + 1 = ?",
            structured_result='{"key_points": ["题目"]}',
            error_code=None,
        )


def _runtime(attachment_ids, conversation_attachment_ids=(), state=None):
    return ToolRuntime(
        state=state or {},
        context=SimpleNamespace(
            selected_attachment_ids=tuple(attachment_ids),
            conversation_attachment_ids=tuple(conversation_attachment_ids),
        ),
        config={},
        stream_writer=lambda _value: None,
        tool_call_id=None,
        store=None,
    )


class ImageAnalysisToolTests(unittest.TestCase):
    def test_tool_does_not_activate_without_request_attachments(self):
        analysis_service = _AnalysisService()
        tool = build_analyze_uploaded_image_tool(
            _AttachmentService(),
            analysis_service,
        )

        result = asyncio.run(
            tool.coroutine(
                attachment_ids=["image-1"],
                runtime=_runtime([]),
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "image_analysis_not_requested")
        self.assertEqual(analysis_service.calls, [])

    def test_tool_activates_for_current_or_historical_ids(self):
        analysis_service = _AnalysisService()
        tool = build_analyze_uploaded_image_tool(
            _AttachmentService(),
            analysis_service,
        )

        result = asyncio.run(
            tool.coroutine(
                attachment_ids=["image-1", "stale-image"],
                runtime=_runtime(["image-1"], ["history-image"]),
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(analysis_service.calls, ["image-1"])
        self.assertEqual(result["results"][0]["attachment_id"], "stale-image")
        self.assertEqual(result["results"][1]["attachment_id"], "image-1")

        historical_result = asyncio.run(
            tool.coroutine(
                attachment_ids=["history-image"],
                runtime=_runtime([], ["history-image"]),
            )
        )
        self.assertTrue(historical_result["ok"])
        self.assertEqual(analysis_service.calls[-1], "history-image")

    def test_fresh_mode_is_forwarded_to_analysis_service(self):
        analysis_service = _AnalysisService()
        tool = build_analyze_uploaded_image_tool(
            _AttachmentService(),
            analysis_service,
        )

        result = asyncio.run(
            tool.coroutine(
                attachment_ids=["image-1"],
                mode="fresh",
                runtime=_runtime(["image-1"]),
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["results"][0]["analysis_source"], "fresh")

    def test_tool_can_fall_back_to_persisted_thread_image_ids(self):
        analysis_service = _AnalysisService()
        tool = build_analyze_uploaded_image_tool(
            _AttachmentService(),
            analysis_service,
        )

        result = asyncio.run(
            tool.coroutine(
                attachment_ids=["image-1"],
                runtime=_runtime(
                    [],
                    state={"image_attachment_ids": ["image-1"]},
                ),
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(analysis_service.calls, ["image-1"])

    def test_empty_or_string_ids_use_the_authorized_image(self):
        analysis_service = _AnalysisService()
        tool = build_analyze_uploaded_image_tool(
            _AttachmentService(),
            analysis_service,
        )

        result = asyncio.run(
            tool.coroutine(
                attachment_ids=None,
                runtime=_runtime(["image-1"]),
            )
        )
        string_result = asyncio.run(
            tool.coroutine(
                attachment_ids="image-1",
                runtime=_runtime(["image-1"]),
            )
        )

        self.assertTrue(result["ok"])
        self.assertTrue(string_result["ok"])
        self.assertEqual(analysis_service.calls, ["image-1", "image-1"])


if __name__ == "__main__":
    unittest.main()
