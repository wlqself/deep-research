import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain.tools import ToolRuntime

from deep_research.publishing.image_generation import (
    ImageGenerationError,
    SiliconFlowImageGenerationService,
)
from deep_research.tools import build_generate_image_tool


class _AttachmentService:
    def __init__(self):
        self.calls = []

    def save(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            attachment_id=f"attachment-{len(self.calls)}",
            filename=kwargs["filename"],
            content_type=kwargs["content_type"],
            size_bytes=len(kwargs["content"]),
        )


class _Response:
    def __init__(self, payload=None, *, content=b"", headers=None):
        self._payload = payload
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self, *args, **kwargs):
        self.posts = []
        self.gets = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return _Response({"images": [{"url": "https://cdn.example/generated.png"}], "seed": 42})

    async def get(self, url):
        self.gets.append(url)
        return _Response(
            content=b"png-bytes",
            headers={"content-type": "image/png"},
        )


def _runtime(thread_id="thread-1"):
    return ToolRuntime(
        state={},
        context=SimpleNamespace(),
        config={"configurable": {"thread_id": thread_id}},
        stream_writer=lambda _value: None,
        tool_call_id=None,
        store=None,
    )


class ImageGenerationServiceTests(unittest.TestCase):
    def test_prompt_is_bounded_before_provider_call(self):
        service = SiliconFlowImageGenerationService(
            _AttachmentService(),
            model_name="Qwen/Qwen-Image",
            api_key="key",
            base_url="https://api.siliconflow.cn/v1",
            prompt_max_chars=8,
        )

        with self.assertRaises(ImageGenerationError) as raised:
            asyncio.run(service.generate(thread_id="thread-1", prompt="123456789"))

        self.assertEqual(raised.exception.error_code, "image_generation_prompt_too_long")

    def test_provider_result_is_downloaded_and_saved_as_attachment(self):
        attachments = _AttachmentService()
        service = SiliconFlowImageGenerationService(
            attachments,
            model_name="Qwen/Qwen-Image",
            api_key="key",
            base_url="https://api.siliconflow.cn/v1",
        )

        with patch(
            "deep_research.publishing.image_generation.httpx.AsyncClient",
            _Client,
        ):
            result = asyncio.run(
                service.generate(
                    thread_id="thread-1",
                    prompt="A warm editorial food photo",
                    count=2,
                    seed=10,
                )
            )

        self.assertEqual([item["attachment_id"] for item in result], ["attachment-1", "attachment-2"])
        self.assertEqual(len(attachments.calls), 2)
        self.assertEqual(attachments.calls[0]["thread_id"], "thread-1")
        self.assertEqual(attachments.calls[0]["content_type"], "image/png")

    def test_base64_provider_result_is_supported(self):
        attachments = _AttachmentService()
        service = SiliconFlowImageGenerationService(
            attachments,
            model_name="Qwen/Qwen-Image",
            api_key="key",
            base_url="https://api.siliconflow.cn/v1",
        )

        class Base64Client(_Client):
            async def post(self, url, **kwargs):
                return _Response({
                    "images": [{"b64_json": base64.b64encode(b"image").decode()}],
                })

        with patch(
            "deep_research.publishing.image_generation.httpx.AsyncClient",
            Base64Client,
        ):
            result = asyncio.run(
                service.generate(thread_id="thread-1", prompt="A blue icon")
            )

        self.assertEqual(result[0]["attachment_id"], "attachment-1")
        self.assertEqual(attachments.calls[0]["content"], b"image")


class GenerateImageToolTests(unittest.TestCase):
    def test_tool_returns_ids_for_direct_publication_reuse(self):
        class FakeService:
            model_name = "Qwen/Qwen-Image"

            async def generate(self, **kwargs):
                return [{
                    "attachment_id": "generated-1",
                    "filename": "generated.png",
                    "content_type": "image/png",
                    "size_bytes": 10,
                    "seed": 1,
                }]

        tool = build_generate_image_tool(FakeService())
        result = asyncio.run(
            tool.coroutine(
                prompt="A clean product illustration",
                runtime=_runtime(),
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["attachment_ids"], ["generated-1"])

    def test_tool_requires_thread_id(self):
        class FakeService:
            model_name = "Qwen/Qwen-Image"

            async def generate(self, **kwargs):
                raise AssertionError("must not call provider")

        tool = build_generate_image_tool(FakeService())
        result = asyncio.run(
            tool.coroutine(
                prompt="A clean product illustration",
                runtime=_runtime(thread_id=""),
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "missing_thread_id")


if __name__ == "__main__":
    unittest.main()
