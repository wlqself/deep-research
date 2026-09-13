import asyncio
import json

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from ..config import settings
from ..context import ResearchContext
from ..rag.service import RagService

def _error_result(
    error: str,
    message: str,
) -> dict[str, object]:
    return {
        "ok": False,
        "error": error,
        "message": message,
        "results": [],
    }

# 接收 LLM 的查询意图 → 在 Qdrant 里向量检索 → 按字符预算截断上下文 → 为每个结果注册来源编号（S1, S2...）→ 把结果和更新后的来源列表返回给 Agent 状态。
def build_search_knowledge_base_tool(
        rag_service: RagService,
):
    @tool
    async def search_knowledge_base(
        query: str,
        top_k: int | None,
        document_id: str | None,
        runtime: ToolRuntime[ResearchContext],
    ) -> dict[str, object] | Command:
        """Search indexed chunks from the local knowledge base."""
        query = query.strip()

        if not query:
            return _error_result(
                "invalid_query",
                "Knowledge base query must not be empty.",
            )
        # top_k 的双重限制（防御性设计）
        requested_top_k = (
            settings.rag_top_k
            if top_k is None
            else top_k
        )

        effective_top_k = min(
            requested_top_k,
            settings.rag_top_k,
        )

        try:
            # 丢进线程池搜索
            results = await asyncio.to_thread(
                rag_service.search_contexts,
                query,
                top_k=effective_top_k,
                document_id=document_id,
            )
        except ValueError:
            return _error_result(
                "invalid_query",
                "Knowledge base query is invalid.",
            )
        except Exception:
            return _error_result(
                "search_failed",
                "The knowledge base search failed.",
            )
        context = runtime.context
        normalized_results: list[dict[str, object]] = []
        used_context_chars = 0

        for document, score in results:
            metadata = document.metadata

            raw_document_id = metadata.get(
                "document_id"
            )
            raw_chunk_id = metadata.get(
                "chunk_id"
            )
            raw_filename = metadata.get(
                "filename"
            )
            # 碰到脏数据跳过
            if not all(
                isinstance(value, str) and value
                for value in (
                    raw_document_id,
                    raw_chunk_id,
                    raw_filename,
                )
            ):
                continue

            content = document.page_content.strip()

            if not content:
                continue
            # 控制字符预算
            remaining = (
                settings.rag_max_context_chars
                - used_context_chars
            )

            if remaining <= 0:
                break # 预算用完了，停止处理更多结果

            content = content[:remaining] # 截断最后一条，确保不超预算
            used_context_chars += len(content)

            page_number = metadata.get(
                "page_number"
            )
            section_title = metadata.get(
                "section_title",
                "",
            )

            title = (
                f"{raw_filename} - page {page_number}"
                if page_number
                else raw_filename
            )
            # 来源注册（S1, S2, S3...）
            source = context.register_source(
                title=title,
                url=(
                    "/knowledge/documents/"
                    f"{raw_document_id}/download"
                ),
                snippet=content[:240],
                source_type="rag",
                document_id=raw_document_id,
                chunk_id=raw_chunk_id,
                page_number=(
                    page_number
                    if isinstance(page_number, int)
                    else None
                ),
                section_title=(
                    section_title
                    if isinstance(section_title, str)
                    else ""
                ),
            )

            normalized_results.append(
                {
                    "document_id": raw_document_id,
                    "chunk_id": raw_chunk_id,
                    "filename": raw_filename,
                    "page_number": page_number,
                    "section_title": section_title,
                    "content": content,
                    "score": score,
                    "parent_id": metadata.get("parent_id"),
                    "source_id": source["source_id"],
                }
            )
        payload = {
            "ok": True,
            "query": query,
            "results": normalized_results,
        }
        if not context.new_sources:
            return payload
        return Command(
            update={
                "sources": dict(
                    context.new_sources
                ),
                "next_source_number": (
                    context.next_source_number
                ),
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),
                        tool_call_id=(
                            runtime.tool_call_id
                        ),
                    )
                ],
            }
        )
    return search_knowledge_base
