"""Helpers that are only used by offline reranker experiments."""

from __future__ import annotations

from collections.abc import Iterable

"""
build_reranker_query 是一个重排序器查询构造工具：
它在保留原始查询意图的基础上，将去重后的检索扩展词以结构化格式附加到查询文本中，
帮助重排序器同时利用用户本意和扩展语义做出更精准的相关性打分。
"""
def build_reranker_query(original_query: str, rewrite_terms: Iterable[str]) -> str:
    """Keep the original intent while exposing rewrite terms to a reranker."""
    # 规范化原始查询并校验
    normalized_query = original_query.strip()
    if not normalized_query:
        raise ValueError("original_query must not be empty")
    # 处理扩展词
    terms = [term.strip() for term in rewrite_terms if term.strip()]
    if not terms:
        return normalized_query
    return (
        f"原始问题：{normalized_query}\n"
        f"检索扩展词：{'；'.join(dict.fromkeys(terms))}"
    )
