"""Summarization policy based only on locally estimated message size."""

from __future__ import annotations

from functools import partial
from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages.utils import count_tokens_approximately


local_message_token_counter = partial(
    count_tokens_approximately,
    use_usage_metadata_scaling=False,
)


class LocalTokenSummarizationMiddleware(SummarizationMiddleware):
    """Ignore unreliable provider usage totals when deciding to summarize."""

    def __init__(self, model: Any, **kwargs: Any) -> None:
        kwargs["token_counter"] = local_message_token_counter
        super().__init__(model, **kwargs)

    def _should_summarize_based_on_reported_tokens(
        self,
        messages: list[Any],
        threshold: float,
    ) -> bool:
        return False


__all__ = [
    "LocalTokenSummarizationMiddleware",
    "local_message_token_counter",
]
