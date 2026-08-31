import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .review_types import (
    MemoryExtractionResult,
    MemoryReviewInput,
)
from ..prompts.memory import MEMORY_EXTRACTION_PROMPT
"""
MemoryExtractor.extract() 把 MemoryReviewInput（对话+旧记忆+产物）序列化成 JSON
连同系统提示词一起发给 LLM，并强制 LLM 严格按 MemoryExtractionResult schema 返回
最后再校验一遍确保数据合法。
"""
class MemoryExtractor:
    def __init__(
        self,
        model: BaseChatModel,
    ) -> None:
        # 把 Pydantic 模型 MemoryExtractionResult 绑定到模型上
        self._model = model.with_structured_output(
            MemoryExtractionResult
        )

    async def extract(
        self,
        review_input: MemoryReviewInput,
    ) -> MemoryExtractionResult:
        # 序列化输入
        payload = review_input.model_dump(
            mode="json",
        )
        # 异步调用
        response: Any = await self._model.ainvoke(
            [
                SystemMessage(
                    content=MEMORY_EXTRACTION_PROMPT.strip()
                ),
                HumanMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                ),
            ]
        )

        return MemoryExtractionResult.model_validate(response)