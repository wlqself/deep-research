from typing import Any

from langchain_core.messages import AIMessage

from ..citations import append_verified_sources
from .messages import content_to_text
from ..state.runtime import agent_config, normalize_question
from ..state.access import (
    context_for_thread,
    sources_for_thread,
)
from .image_context import (
    image_context_instruction,
    user_message_with_image_ids,
)

"""
调用 agent 完成全部执行流程后，从消息历史中逆序提取最后一条 AI 最终回答，
追加来源引用后一次性返回，适合不需要实时流式输出、只需要最终结果的场景（如后台任务、测试、批量处理）。
"""
async def run_research_with_agent(
    agent: Any,
    question: str,
    thread_id: str,
    *,
    selected_attachment_ids: tuple[str, ...] = (),
    attachment_selection_confirmed: bool = False,
) -> str:
    question = normalize_question(question)

    context = await context_for_thread(
        agent,
        thread_id,
        selected_attachment_ids=selected_attachment_ids,
        attachment_selection_confirmed=attachment_selection_confirmed,
    )

    image_instruction = image_context_instruction(context)
    model_question = (
        f"{question}\n\n{image_instruction}"
        if image_instruction
        else question
    )

    result = await agent.ainvoke(
        {
            "messages": [
                user_message_with_image_ids(
                    model_question,
                    selected_attachment_ids,
                    display_content=question,
                )
            ],
            # This reducer-backed field is outside the summarizable message
            # history, so image references survive context compression.
            "image_attachment_ids": list(selected_attachment_ids),
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
