import importlib
import unittest

import httpx

from langchain.tools import ToolRuntime
from deep_research.context import ResearchContext

module = importlib.import_module(
    "deep_research.tools.read_page"
)

class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        content_length: str | None = None,
    ):
        self.url = "https://example.com/article"
        self.encoding = "utf-8"
        self.status_code = status_code
        self.headers = {
            "content-type": "text/html; charset=utf-8",
        }

        if content_length is not None:
            self.headers["content-length"] = content_length

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request(
                "GET",
                self.url,
            )
            response = httpx.Response(
                self.status_code,
                request=request,
            )
            raise httpx.HTTPStatusError(
                "fake http failure",
                request=request,
                response=response,
            )

    async def aiter_bytes(self):
        yield b"<html><body>fake</body></html>"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class FakeClient:
    response = None

    def __init__(self, **kwargs):
        pass

    def stream(self, method, url):
        return self.response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


def make_runtime(
    context: ResearchContext,
) -> ToolRuntime:
    return ToolRuntime(
        state={
            "sources": {
                "S1": {
                    "source_id": "S1",
                    "title": "Test source",
                    "url": "https://example.com/article",
                    "snippet": "Test snippet",
                }
            }
        },
        context=context,
        config={},
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )


def make_context() -> ResearchContext:
    return ResearchContext(
        max_search_calls=4,
        max_page_reads=6,
        max_page_chars=12000,
    )


class ReadPageTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_failure_returns_structured_error(self):
        original_client = module.httpx.AsyncClient
        FakeClient.response = FakeResponse(status_code=503)
        module.httpx.AsyncClient = FakeClient

        context = make_context()

        try:
            result = await module.read_page.coroutine(
                "S1",
                make_runtime(context),
            )
        finally:
            module.httpx.AsyncClient = original_client

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "http_error")
        self.assertEqual(context.page_read_count, 1)

    async def test_large_response_returns_structured_error(self):
        original_client = module.httpx.AsyncClient
        FakeClient.response = FakeResponse(
            content_length=str(module.MAX_RESPONSE_BYTES + 1),
        )
        module.httpx.AsyncClient = FakeClient

        context = make_context()

        try:
            result = await module.read_page.coroutine(
                "S1",
                make_runtime(context),
            )
        finally:
            module.httpx.AsyncClient = original_client

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "response_too_large")
        self.assertEqual(context.page_read_count, 1)


if __name__ == "__main__":
    unittest.main()