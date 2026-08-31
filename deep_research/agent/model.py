from typing import Any

from langchain_openai import ChatOpenAI

from ..config import settings


model_kwargs: dict[str, Any] = {
    "model": settings.model_name,
    "api_key": settings.model_api_key,
    "temperature": 0,
    "timeout": 60,
    "max_retries": 5,
}

if settings.model_base_url:
    model_kwargs["base_url"] = settings.model_base_url


model = ChatOpenAI(**model_kwargs)