"""Reusable runners for evaluating different retrieval implementations."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from .rag_metrics import RetrievalCase, evaluate_retrieval

"""
load_jsonl 是一个严格模式的 JSONL 文件加载器：
它逐行读取文件，跳过空行，将每行解析为 JSON 对象并校验类型，
遇到非对象行立即抛错定位问题，最终返回所有合法字典对象的列表，
为审计日志、持久化数据等场景提供可靠的数据加载能力。
"""
def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load a JSONL file and fail early when a row is not an object."""
    # 初始化结果列表
    rows: list[dict[str, object]] = []
    # 打开文件
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            rows.append(row)
    return rows

"""
load_eval_queries 是一个评估查询的加载与校验函数：
它在 load_jsonl 的基础上进一步校验每条记录必须包含非空的 _id 和 text 字段，
确保评估数据集的完整性，为后续的检索/回答质量评测提供可靠的"考题"输入。
"""
def load_eval_queries(path: str | Path) -> list[dict[str, object]]:
    """Load generated evaluation queries without mixing in relevance labels."""
    # 加载 JSONL 文件
    queries = load_jsonl(path)
    # 遍历校验每条查询
    for row in queries:
        query_id = row.get("_id")
        query_text = row.get("text")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("Each query must have a non-empty string '_id'")
        if not isinstance(query_text, str) or not query_text:
            raise ValueError(
                f"Query {query_id!r} must have a non-empty string 'text'"
            )
    return queries

"""
load_qrels 是一个BEIR 风格查询相关性判断文件的加载器：
它严格校验 TSV 文件的表头和每行数据的完整性，
将 query-id、corpus-id、score 解析为 query_id -> chunk_id -> grade 的嵌套字典，
为检索系统的评估流程提供标准化的相关性标注数据。
"""
def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load BEIR-style qrels into ``query_id -> chunk_id -> grade``."""
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected_fields:
            raise ValueError(
                "Qrels must have exactly these tab-separated columns: "
                "query-id, corpus-id, score"
            )

        for row in reader:
            query_id = row["query-id"]
            chunk_id = row["corpus-id"]
            if not query_id or not chunk_id:
                raise ValueError("Qrels query and corpus IDs cannot be empty")
            try:
                grade = int(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Qrels score must be an integer: {row['score']!r}"
                ) from exc
            qrels.setdefault(query_id, {})[chunk_id] = grade
    return qrels


Search = Callable[[str, int], Sequence[str]]

"""
evaluate_retriever 是一个检索器评估的编排函数：
它遍历查询集合逐个调用 search 接口执行检索并计时，
把结果和 qrels 标注交给 evaluate_retrieval 计算标准信息检索指标，
最后把检索器名称、top_k、每查询的召回列表和延迟补充到结果中，
为不同检索策略（稠密/稀疏/混合）提供统一、可对比的评估输出。
"""
def evaluate_retriever(
    queries: Iterable[Mapping[str, object]],
    qrels: Mapping[str, Mapping[str, int]],
    search: Search,
    *,
    top_k: int = 8,
    k_values: Sequence[int] = (1, 3, 5, 8),
    retriever_name: str = "unknown",
) -> dict[str, object]:
    """Run retrieval and calculate the shared metric contract.

    ``search`` is deliberately small: it receives query text and top-k, and
    returns chunk IDs in ranked order. Dense, BM25, hybrid, and reranker
    implementations can all be adapted to this contract.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    # 初始化结果收集列表
    cases: list[RetrievalCase] = []
    retrieved_by_query: list[tuple[list[str], float]] = []
    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]
        if not isinstance(query_id, str) or not isinstance(query_text, str):
            raise ValueError("Each query must contain string '_id' and 'text'")

        started = time.perf_counter()
        ranked_chunk_ids = list(search(query_text, top_k))
        elapsed_ms = (time.perf_counter() - started) * 1000
        cases.append(
            RetrievalCase(
                query_id=query_id,
                ranked_chunk_ids=ranked_chunk_ids,
                relevance=qrels.get(query_id, {}),
            )
        )
        retrieved_by_query.append((ranked_chunk_ids, elapsed_ms))

    result = evaluate_retrieval(cases, k_values=k_values)
    result["retriever"] = retriever_name
    result["top_k"] = top_k
    per_query_results = result["per_query"]
    if len(per_query_results) != len(retrieved_by_query):
        raise RuntimeError("Metric and retrieval result counts do not match")
    for per_query, (ranked_chunk_ids, elapsed_ms) in zip(
        per_query_results, retrieved_by_query
    ):
        per_query["retrieved_chunk_ids"] = ranked_chunk_ids
        per_query["latency_ms"] = round(elapsed_ms, 3)
    return result
