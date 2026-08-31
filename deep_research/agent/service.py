from typing import Any

from langchain_core.messages import AIMessage

from ..citations import append_verified_sources
from .messages import content_to_text
from ..state.runtime import agent_config, normalize_question
from ..state.access import (
    context_for_thread,
    sources_for_thread,
)


async def run_research_with_agent(
    agent: Any,
    question: str,
    thread_id: str,
) -> str:
    question = normalize_question(question)

    context = await context_for_thread(
        agent,
        thread_id,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config=agent_config(thread_id),
        context=context,
    )

    messages = result.get("messages", [])

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            answer = content_to_text(message.content)

            if answer:
                sources = await sources_for_thread(
                    agent,
                    thread_id,
                )

                return append_verified_sources(
                    answer,
                    sources,
                )

    raise RuntimeError("Agent did not return a final answer.")
