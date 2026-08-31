from typing import Any

from ..context import ResearchContext
from .runtime import agent_config

#把Checkpointer里存的“老数据”，转换成每轮Agent需要的“新上下文”。
async def thread_values(
    agent: Any,
    thread_id: str,
) -> dict[str, object]:
    snapshot = await agent.aget_state(
        agent_config(thread_id)
    )

    values = snapshot.values

    if not isinstance(values, dict):
        return {}

    return values


async def context_for_thread(
    agent: Any,
    thread_id: str,
) -> ResearchContext:
    values = await thread_values(
        agent,
        thread_id,
    )

    raw_next_number = values.get(
        "next_source_number",
        1,
    )

    next_source_number = (
        raw_next_number
        if isinstance(raw_next_number, int)
        and raw_next_number >= 1
        else 1
    )

    return ResearchContext.from_settings(
        next_source_number=next_source_number,
    )


async def sources_for_thread(
    agent: Any,
    thread_id: str,
) -> dict[str, object]:
    values = await thread_values(
        agent,
        thread_id,
    )

    sources = values.get("sources", {})

    if not isinstance(sources, dict):
        return {}

    return sources