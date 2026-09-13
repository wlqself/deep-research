from typing import Any

from langchain_openai import ChatOpenAI

from ..config import settings


def build_chat_model(*, enable_thinking: bool) -> ChatOpenAI:
    """Build one role-specific SiliconFlow-compatible chat model."""

    model_kwargs: dict[str, Any] = {
        "model": settings.model_name,
        "api_key": settings.model_api_key,
        "temperature": 0,
        "timeout": 60,
        "max_retries": 5,
        "extra_body": {
            "enable_thinking": enable_thinking,
        },
    }

    if settings.model_base_url:
        model_kwargs["base_url"] = settings.model_base_url

    return ChatOpenAI(**model_kwargs)


model = build_chat_model(
    enable_thinking=settings.main_agent_enable_thinking,
)
researcher_model = build_chat_model(
    enable_thinking=settings.researcher_enable_thinking,
)


__all__ = ["build_chat_model", "model", "researcher_model"]
