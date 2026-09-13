from typing import Any

from ..citations import append_verified_sources
from ..state.access import context_for_thread, sources_for_thread
from ..state.runtime import agent_config, normalize_question
from .image_context import (
    image_context_instruction,
    user_message_with_image_ids,
)

# 来源引用的流式回答生成器
async def stream_research_with_agent(
    agent: Any,
    question: str,
    thread_id: str,
    *,
    selected_attachment_ids: tuple[str, ...] = (),
    attachment_selection_confirmed: bool = False,
):
    question = normalize_question(question)
    context = await context_for_thread(
        agent,
        thread_id,
        selected_attachment_ids=selected_attachment_ids,
        attachment_selection_confirmed=attachment_selection_confirmed,
    )
    answer_chunks: list[str] = []
    image_instruction = image_context_instruction(context)
    model_question = (
        f"{question}\n\n{image_instruction}"
        if image_instruction
        else question
    )

    agent_input = {
        "messages": [
            user_message_with_image_ids(
                model_question,
                selected_attachment_ids,
                display_content=question,
            )
        ],
        "image_attachment_ids": list(selected_attachment_ids),
    }
    if attachment_selection_confirmed:
        agent_input["publication_attachment_ids"] = list(
            selected_attachment_ids
        )

    async for token, metadata in agent.astream(
        agent_input,
        config=agent_config(thread_id),
        context=context,
        stream_mode="messages",
    ):
        if not isinstance(metadata, dict):
            continue

        if metadata.get("lc_agent_name") == "researcher":
            continue

        if metadata.get("langgraph_node") != "model":
            continue

        text = getattr(token, "text", "")

        if text:
            answer_chunks.append(text)
            yield text

    raw_answer = "".join(answer_chunks)
    sources = await sources_for_thread(agent, thread_id)
    verified_answer = append_verified_sources(raw_answer, sources)
    base_answer = raw_answer.rstrip()

    if verified_answer.startswith(base_answer):
        suffix = verified_answer[len(base_answer):]

        if suffix:
            yield suffix
