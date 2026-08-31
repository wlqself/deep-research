import asyncio
from urllib.parse import urlparse

import httpx
import trafilatura
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ..context import ResearchContext


READ_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 2_000_000
ALLOWED_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
}


def _error_result(
    error: str,
    message: str,
    source_id: str,
) -> dict[str, object]:
    return {
        "ok": False,
        "error": error,
        "message": message,
        "source_id": source_id,
        "content": "",
    }


@tool
async def read_page(
    source_id: str,
    runtime: ToolRuntime[ResearchContext],
) -> dict[str, object]:
    """Read cleaned text from a source registered by web_search."""
    raw_sources = runtime.state.get(
        "sources",
        {},
    )

    source = (
        raw_sources.get(source_id)
        if isinstance(raw_sources, dict)
        else None
    )

    if not isinstance(source, dict):
        return _error_result(
            "unknown_source_id",
            "The source_id was not registered by this thread.",
            source_id,
        )

    if source is None:
        return _error_result(
            "unknown_source_id",
            "The source_id was not registered by this request.",
            source_id,
        )

    parsed_url = urlparse(source["url"])

    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return _error_result(
            "unsupported_url",
            "Only valid HTTP and HTTPS source URLs are supported.",
            source_id,
        )

    context = runtime.context

    if not context.try_acquire_page_read_slot():
        return _error_result(
            "page_read_budget_exhausted",
            "Maximum page reads reached.",
            source_id,
        )

    try:
        async with httpx.AsyncClient(
            timeout=READ_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={
                "User-Agent": "DeepResearchAgent/0.2",
            },
        ) as client:
            async with client.stream("GET", source["url"]) as response:
                response.raise_for_status()

                content_type = response.headers.get(
                    "content-type",
                    "",
                ).split(";", 1)[0].strip().lower()

                if content_type not in ALLOWED_CONTENT_TYPES:
                    return _error_result(
                        "unsupported_content_type",
                        "The page did not return supported text content.",
                        source_id,
                    )

                content_length = response.headers.get("content-length")

                if content_length:
                    try:
                        if int(content_length) > MAX_RESPONSE_BYTES:
                            return _error_result(
                                "response_too_large",
                                "The page response is too large.",
                                source_id,
                            )
                    except ValueError:
                        pass

                chunks: list[bytes] = []
                total_bytes = 0

                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)

                    if total_bytes > MAX_RESPONSE_BYTES:
                        return _error_result(
                            "response_too_large",
                            "The page response is too large.",
                            source_id,
                        )

                    chunks.append(chunk)

                final_scheme = urlparse(
                    str(response.url)
                ).scheme

                if final_scheme not in {"http", "https"}:
                    return _error_result(
                        "unsupported_redirect",
                        "The page redirected to an unsupported URL.",
                        source_id,
                    )

                page_text = b"".join(chunks).decode(
                    response.encoding or "utf-8",
                    errors="replace",
                )

    except httpx.TimeoutException:
        return _error_result(
            "page_timeout",
            "The page did not respond within the timeout.",
            source_id,
        )
    except httpx.HTTPStatusError:
        return _error_result(
            "http_error",
            "The page returned an HTTP error status.",
            source_id,
        )
    except httpx.HTTPError:
        return _error_result(
            "page_fetch_failed",
            "The page could not be fetched.",
            source_id,
        )
    except Exception:
        return _error_result(
            "page_read_failed",
            "The page could not be read.",
            source_id,
        )

    try:
        if content_type == "text/plain":
            cleaned_text = page_text.strip()
        else:
            cleaned_text = await asyncio.to_thread(
                trafilatura.extract,
                page_text,
                url=source["url"],
                output_format="txt",
                favor_precision=True,
                include_comments=False,
                include_images=False,
                include_links=False,
            ) or ""

            cleaned_text = cleaned_text.strip()

    except Exception:
        return _error_result(
            "content_extraction_failed",
            "The page body could not be extracted.",
            source_id,
        )

    if not cleaned_text:
        return _error_result(
            "empty_page_content",
            "No readable page content was found.",
            source_id,
        )

    truncated = len(cleaned_text) > context.max_page_chars

    return {
        "ok": True,
        "source_id": source["source_id"],
        "title": source["title"],
        "url": source["url"],
        "content": cleaned_text[:context.max_page_chars],
        "truncated": truncated,
    }
