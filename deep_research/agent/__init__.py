from collections.abc import AsyncIterator

from langgraph.checkpoint.memory import InMemorySaver

from ..context import ResearchContext
from .factory import build_agent
from .service import run_research_with_agent
from .streaming import (
    stream_research_events_with_agent,
    stream_research_with_agent,
)

agent = build_agent(InMemorySaver())


async def run_research(question: str, thread_id: str) -> str:
    return await run_research_with_agent(
        agent,
        question,
        thread_id,
    )


async def stream_research(question: str, thread_id: str):
    async for chunk in stream_research_with_agent(
        agent,
        question,
        thread_id,
    ):
        yield chunk


async def stream_research_events(
    question: str,
    thread_id: str,
) -> AsyncIterator[dict[str, object]]:
    async for event in stream_research_events_with_agent(
        agent,
        question,
        thread_id,
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
