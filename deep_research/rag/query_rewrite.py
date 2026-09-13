"""Controlled query rewriting for the production lexical retrieval stage."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .lexical import build_lexical_query


QUERY_REWRITE_PROMPT = """
你是检索查询改写器，不是问答助手。

输入是一条用户检索问题。只输出用于 BM25/关键词检索的短词或短语列表，
不要回答问题，不要总结文档，不要生成不存在的事实。

规则：
1. 必须保留原问题中的专有名词、模型名、缩写、数字、公式、实验编号和文件名片段。
2. 可以补充原问题明确暗示的中英文术语、同义术语或技术表达，例如：
   “随机种子”可补充为 “random seed” 或 “training seed”。
3. 可以把口语化表达转换为文档中更可能出现的技术短语，例如：
   “多轮长任务”可补充为 “multi-turn” 和 “long-horizon task”。
4. 不要加入问题答案中的具体数值、结论或未经输入支持的实体。
5. 每个词或短语尽量短，最多输出 24 个，按重要性排序。
Output must be a JSON object with exactly this shape:
{"terms": ["term 1", "term 2"]}
""".strip()


class RetrievalQueryRewrite(BaseModel):
    """The only model output allowed in the retrieval rewrite stage."""

    model_config = ConfigDict(extra="forbid")

    terms: list[str] = Field(default_factory=list, max_length=24)

    @field_validator("terms")
    @classmethod
    def normalize_terms(cls, terms: list[str]) -> list[str]:
        normalized: list[str] = []
        for term in terms:
            value = term.strip()
            if not value or len(value) > 80:
                continue
            if value not in normalized:
                normalized.append(value)
        return normalized


class QueryRewriter:
    """Synchronous, injectable wrapper around a structured-output chat model."""

    def __init__(self, model: BaseChatModel, *, structured_output: bool = True) -> None:
        self._structured_output = structured_output
        self._model = model.with_structured_output(RetrievalQueryRewrite) if structured_output else model

    def rewrite(self, query: str) -> RetrievalQueryRewrite:
        if not query.strip():
            raise ValueError("query must not be empty")
        response = self._model.invoke(
            [
                SystemMessage(content=QUERY_REWRITE_PROMPT),
                HumanMessage(content=query.strip()),
            ]
        )
        if self._structured_output:
            return RetrievalQueryRewrite.model_validate(response)
        return RetrievalQueryRewrite.model_validate(_parse_json_response(response))


def _parse_json_response(response: Any) -> dict[str, object]:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        raise ValueError("Query rewrite response must contain text content")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned.removeprefix("json").strip()
        cleaned = cleaned.removesuffix("```").strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Query rewrite response must be a JSON object")
    return parsed


def build_rewritten_lexical_query(
    original_query: str,
    rewrite_terms: Iterable[str],
) -> str:
    """Merge original lexical signals with model-provided search terms."""
    base_query = build_lexical_query(original_query)
    terms = [base_query, *(term.strip() for term in rewrite_terms)]
    return " ".join(
        term
        for index, term in enumerate(terms)
        if term and term not in terms[:index]
    )
