from typing import Any

from ..config import settings


def agent_config(thread_id: str) -> dict[str, Any]:
    normalized_thread_id = thread_id.strip()

    if not normalized_thread_id:
        raise ValueError("thread_id must not be empty.")

    return {
        "recursion_limit": settings.agent_recursion_limit,
        "configurable": {
            "thread_id": normalized_thread_id,
        },
    }


def normalize_question(question: str) -> str:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("Research question must not be empty.")

    return normalized_question