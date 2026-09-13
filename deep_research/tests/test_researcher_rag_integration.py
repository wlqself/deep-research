import unittest

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from deep_research.agent.sub_agent import build_researcher
from deep_research.context import ResearchContext


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class FakeRagService:
    def search_contexts(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        return [
            (
                Document(
                    page_content="本地知识库证据",
                    metadata={
                        "document_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "filename": "guide.md",
                        "page_number": 1,
                        "section_title": "安装",
                    },
                ),
                0.95,
            )
        ]


class ResearcherRagIntegrationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_researcher_calls_rag_and_returns_source_state(self):
        model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_knowledge_base",
                                "args": {
                                    "query": "如何安装？",
                                    "top_k": 3,
                                    "document_id": None,
                                },
                                "id": "rag-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ResearcherResult",
                                "args": {
                                    "summary": "已找到本地证据。",
                                    "finding_ids": [],
                                    "source_ids": ["S1"],
                                    "knowledge_gaps": [],
                                    "conflicts": [],
                                    "recommended_action": "answer",
                                },
                                "id": "result-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            )
        )

        researcher = build_researcher(
            model,
            FakeRagService(),
        )

        result = await researcher["runnable"].ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "研究本地安装说明。",
                    }
                ]
            },
            context=ResearchContext(
                max_search_calls=4,
                max_page_reads=6,
                max_page_chars=12000,
            ),
        )

        self.assertIn("sources", result)
        self.assertIn("S1", result["sources"])
        self.assertEqual(
            result["sources"]["S1"]["source_type"],
            "rag",
        )
        self.assertEqual(
            result["sources"]["S1"]["chunk_id"],
            "chunk-1",
        )


if __name__ == "__main__":
    unittest.main()
