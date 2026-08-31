import json
import unittest
from unittest.mock import patch

from langchain.tools import ToolRuntime
from langchain_core.documents import Document
from langgraph.types import Command

from deep_research.context import ResearchContext
from deep_research.tools.search_knowledge_base import (
    build_search_knowledge_base_tool,
)


def make_runtime(
    context: ResearchContext,
    tool_call_id: str = "rag-call-1",
) -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=None,
    )


class FakeRagService:
    def __init__(
        self,
        results=None,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    def search_chunks(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "document_id": document_id,
            }
        )
        if self.error is not None:
            raise self.error
        return self.results


def make_result(
    content: str,
    *,
    document_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    page_number: int = 2,
) -> tuple[Document, float]:
    return (
        Document(
            page_content=content,
            metadata={
                "document_id": document_id,
                "chunk_id": chunk_id,
                "filename": "guide.pdf",
                "page_number": page_number,
                "section_title": "安装",
            },
        ),
        0.91,
    )


class SearchKnowledgeBaseTests(
    unittest.IsolatedAsyncioTestCase
):
    async def call_tool(
        self,
        rag_service: FakeRagService,
        *,
        query: str = "如何安装？",
        top_k: int | None = None,
        document_id: str | None = None,
        context: ResearchContext | None = None,
    ):
        tool = build_search_knowledge_base_tool(
            rag_service  # type: ignore[arg-type]
        )
        context = context or ResearchContext(
            max_search_calls=4,
            max_page_reads=6,
            max_page_chars=12000,
            next_source_number=1,
        )
        return await tool.coroutine(
            query=query,
            top_k=top_k,
            document_id=document_id,
            runtime=make_runtime(context),
        ), context

    async def test_success_registers_rag_sources_and_returns_command(self):
        rag_service = FakeRagService(
            results=[
                make_result("第一段证据"),
                make_result(
                    "第二段证据",
                    chunk_id="chunk-2",
                    page_number=3,
                ),
            ]
        )

        result, context = await self.call_tool(
            rag_service,
            top_k=5,
            document_id="doc-1",
        )

        self.assertIsInstance(result, Command)
        self.assertEqual(
            rag_service.calls,
            [
                {
                    "query": "如何安装？",
                    "top_k": 5,
                    "document_id": "doc-1",
                }
            ],
        )
        self.assertEqual(set(result.update["sources"]), {"S1", "S2"})
        self.assertEqual(result.update["next_source_number"], 3)
        self.assertEqual(set(context.new_sources), {"S1", "S2"})
        self.assertEqual(
            context.new_sources["S1"]["source_type"],
            "rag",
        )
        self.assertEqual(
            context.new_sources["S1"]["chunk_id"],
            "chunk-1",
        )

        message = result.update["messages"][0]
        payload = json.loads(message.content)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["results"][0]["source_id"], "S1")

    async def test_top_k_is_limited_by_server_setting(self):
        rag_service = FakeRagService()

        with patch.object(
            __import__(
                "deep_research.tools.search_knowledge_base",
                fromlist=["settings"],
            ).settings,
            "rag_top_k",
            3,
        ):
            result, _ = await self.call_tool(
                rag_service,
                top_k=99,
            )

        self.assertEqual(rag_service.calls[0]["top_k"], 3)
        self.assertEqual(result["ok"], True)

    async def test_context_char_limit_truncates_results(self):
        rag_service = FakeRagService(
            results=[
                make_result("abcdefgh"),
                make_result("ijklmnop", chunk_id="chunk-2"),
            ]
        )

        tools_module = __import__(
            "deep_research.tools.search_knowledge_base",
            fromlist=["settings"],
        )
        with patch.object(
            tools_module.settings,
            "rag_max_context_chars",
            10,
        ):
            result, _ = await self.call_tool(
                rag_service,
                top_k=2,
            )

        self.assertIsInstance(result, Command)
        payload = json.loads(result.update["messages"][0].content)
        contents = [item["content"] for item in payload["results"]]
        self.assertEqual(contents, ["abcdefgh", "ij"])
        self.assertLessEqual(sum(map(len, contents)), 10)

    async def test_empty_results_return_structured_empty_response(self):
        result, context = await self.call_tool(
            FakeRagService(results=[])
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["results"], [])
        self.assertEqual(context.new_sources, {})

    async def test_search_failure_returns_safe_error(self):
        result, context = await self.call_tool(
            FakeRagService(error=RuntimeError("secret backend detail"))
        )

        self.assertEqual(result["ok"], False)
        self.assertEqual(result["error"], "search_failed")
        self.assertEqual(result["results"], [])
        self.assertEqual(context.new_sources, {})


if __name__ == "__main__":
    unittest.main()
