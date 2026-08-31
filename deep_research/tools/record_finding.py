import json
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..findings.types import (
    FindingStatus,
    ResearchFinding,
)

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ..context import ResearchContext

MAX_EVIDENCE_SUMMARY_CHARS = 4000

def _error_result(
        error: str,
        message: str,
) -> dict[str, object]:
    return {
        "ok": False,
        "error": error,
        "message": message,
    }

@tool
def record_research_finding(
    claim: str,
    evidence_summary: str,
    source_ids: list[str],
    status: FindingStatus,
    uncertainty: str,
    conflicts: str,
    runtime: ToolRuntime[ResearchContext],
) -> dict[str, object] | Command:
    """Record a structured research finding for the current thread."""
    normalized_claim = claim.strip()
    normalized_evidence = evidence_summary.strip()


    if not normalized_claim:
        return _error_result(
            "invalid_claim",
            "Claim must not be empty.",
        )

    if not normalized_evidence:
        return _error_result(
            "invalid_evidence_summary",
            "Evidence summary must not be empty.",
        )

    if len(normalized_evidence) > MAX_EVIDENCE_SUMMARY_CHARS:
        return _error_result(
            "evidence_summary_too_long",
            "Evidence summary is too long.",
        )

    if not isinstance(source_ids, list):
        return _error_result(
            "invalid_source_ids",
            "source_ids must be a list of strings.",
        )

    if not all(
        isinstance(source_id, str)
        for source_id in source_ids
    ):
        return _error_result(
            "invalid_source_ids",
            "source_ids must be a list of strings.",
        )

    if not source_ids:
        return _error_result(
            "empty_source_ids",
            "At least one source_id is required.",
        )

    if len(set(source_ids)) != len(source_ids):
        return _error_result(
            "duplicate_source_id",
            "source_ids must not contain duplicates.",
        )
    # 从 State 里拿“白名单来源”
    raw_sources = runtime.state.get(
        "sources",
        {},
    )

    sources = (
        raw_sources
        if isinstance(raw_sources, dict)
        else {}
    )

    unknown_source_ids = [
        source_id
        for source_id in source_ids
        if source_id not in sources
    ]

    if unknown_source_ids:
        return _error_result(
            "unknown_source_id",
            (
                "The following source_id values were not "
                "registered by the current thread: "
                + ", ".join(unknown_source_ids)
            ),
        )

    if status not in {
        "supported",
        "conflicted",
        "insufficient",
    }:
        return _error_result(
            "invalid_status",
            "status must be supported, conflicted, or insufficient.",
        )

    normalized_uncertainty = uncertainty.strip()
    normalized_conflicts = conflicts.strip()

    if status == "conflicted" and not normalized_conflicts:
        return _error_result(
            "missing_conflicts",
            "conflicts is required for conflicted findings.",
        )

    if not runtime.tool_call_id:
        return _error_result(
            "missing_tool_call_id",
            "A tool_call_id is required to record a finding.",
        )

    created_at = datetime.now(
        timezone.utc,
    ).isoformat()
    # 生成一个“不可伪造”的 finding_id
    finding_id = uuid4().hex
    # 把“研究发现”变成结构化状态
    finding: ResearchFinding = {
        "finding_id": finding_id,
        "claim": normalized_claim,
        "evidence_summary": normalized_evidence,
        "source_ids": list(source_ids),
        "status": status,
        "uncertainty": normalized_uncertainty,
        "conflicts": normalized_conflicts,
        "created_at": created_at,
        "updated_at": created_at,
    }

    payload = {
        "ok": True,
        "finding_id": finding_id,
        "status": status,
        "source_ids": list(source_ids),
    }
    # 用 Command 把 finding 写进持久化 State
    return Command(
        update={
            "findings": {
                finding_id: finding,
            },
            "messages": [
                ToolMessage( #  用 ToolMessage 告诉 LLM“我记下来了”
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
