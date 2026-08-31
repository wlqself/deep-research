import json
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..config import settings
from ..context import ResearchContext
from ..sources import PersistentSource

"""
接受错误函数，接受错误和信息，并返回为什么错误和错误代码
"""
def _error_result(error: str, message: str) -> dict[str, object]:
    return {
        "ok": False,
        "error": error,
        "message": message,
        "results": [],
    }
"""
网络搜索
"""

@tool
def web_search(
    query: str,
    runtime: ToolRuntime[ResearchContext],
) -> dict[str, object]:
    """Search the web with Tavily and return source title, URL, and snippet."""

    query = query.strip()
    if not query:
        return _error_result(
            "invalid_query",
            "Search query must not be empty.",
        )

    if not settings.tavily_api_key:
        return _error_result(
            "missing_configuration",
            "TAVILY_API_KEY is not configured.",
        )

    context = runtime.context

    if not context.try_acquire_search_slot():
        return _error_result(
            "search_budget_exhausted",
            "Maximum search calls reached.",
        )
    
    try:
        api_wrapper = TavilySearchAPIWrapper(
            # The SDK field is already typed as SecretStr. Passing a plain
            # string lets Pydantic wrap it once; wrapping it here first turns
            # the actual key into the literal mask "**********".
            tavily_api_key=settings.tavily_api_key,
        )

        search = TavilySearch(
            max_results=5,
            topic="general",
            api_wrapper=api_wrapper,
        )
        raw_result: Any = search.invoke({"query": query})

        if isinstance(raw_result, str):
            raw_result = json.loads(raw_result)

        if not isinstance(raw_result, dict):
            return _error_result(
                "invalid_search_response",
                "The search service returned an invalid response.",
            )

        if "error" in raw_result:
            return _error_result(
                "search_failed",
                "The web search service is temporarily unavailable.",
            )

        raw_results = raw_result.get("results")

        if not isinstance(raw_results, list):
            return _error_result(
                "invalid_search_response",
                "The search service returned an invalid response.",
            )

        normalized_results: list[dict[str, str]] = []

        for item in raw_results:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            snippet = str(
                item.get("snippet") or item.get("content") or ""
            ).strip()   #snippet = 摘要 / 片段 / 预览文本，最后的搜索我们只需要 标题、链接、摘要和片段


            if title and url and snippet:
                source = context.register_source(
                    title=title,
                    url=url,
                    snippet=snippet,
                )

                normalized_results.append(source)

        payload = {
            "ok": True,
            "query": query,
            "results": normalized_results,
        }

        if not context.new_sources:
            return payload

        return Command(
            update={
                "sources": dict(context.new_sources),
                "next_source_number": context.next_source_number,
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    except Exception:
        return _error_result(
            "search_failed",
            "The web search service is temporarily unavailable.",
        )
