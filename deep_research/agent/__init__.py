from collections.abc import AsyncIterator

from langgraph.checkpoint.memory import InMemorySaver

from ..context import ResearchContext
from .factory import build_agent
from .service import run_research_with_agent
from .streaming import (
    stream_research_events_with_agent,
    stream_research_with_agent,
)
from ..log.log_context import get_request_id

agent = build_agent(InMemorySaver())


async def run_research(
    question: str,
    thread_id: str,
    *,
    selected_attachment_ids: tuple[str, ...] = (),
    attachment_selection_confirmed: bool = False,
) -> str:
    return await run_research_with_agent(
        agent,
        question,
        thread_id,
        selected_attachment_ids=selected_attachment_ids,
        attachment_selection_confirmed=attachment_selection_confirmed,
    )


async def stream_research(
    question: str,
    thread_id: str,
    *,
    selected_attachment_ids: tuple[str, ...] = (),
    attachment_selection_confirmed: bool = False,
):
    async for chunk in stream_research_with_agent(
        agent,
        question,
        thread_id,
        selected_attachment_ids=selected_attachment_ids,
        attachment_selection_confirmed=attachment_selection_confirmed,
    ):
        yield chunk


async def stream_research_events(
    question: str,
    thread_id: str,
    selected_attachment_ids: tuple[str, ...] = (),
    attachment_selection_confirmed: bool = False,
) -> AsyncIterator[dict[str, object]]:
    async for event in stream_research_events_with_agent(
        agent,
        question,
        thread_id,
        selected_attachment_ids=selected_attachment_ids,
        attachment_selection_confirmed=attachment_selection_confirmed,
        observability=get_request_id() is not None,
    ):
        yield event


__all__ = [
    "agent",
    "build_agent",
    "ResearchContext",
    "run_research",
    "stream_research",
    "stream_research_events",
]
