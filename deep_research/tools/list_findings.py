from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ..context import ResearchContext
from ..findings.types import FindingStatus

@tool
def list_research_findings(
    runtime: ToolRuntime[ResearchContext],
    status: FindingStatus | None = None,
) -> dict[str, object]:
    """List structured research findings from the current thread."""

    if status is not None and status not in {
        "supported",
        "conflicted",
        "insufficient",
    }:
        return {
            "ok": False,
            "error": "invalid_status",
            "message": (
                "status must be supported, conflicted, or insufficient."
            ),
            "findings": [],
        }

    raw_findings = runtime.state.get(
        "findings",
        {},
    )

    if not isinstance(raw_findings, dict):
        return {
            "ok": True,
            "findings": [],
        }

    findings: list[dict[str, object]] = []

    for finding_id, finding in raw_findings.items():
        if not isinstance(finding_id, str):
            continue

        if not isinstance(finding, dict):
            continue

        if status is not None and finding.get("status") != status:
            continue

        findings.append(
            {
                "finding_id": finding_id,
                "claim": finding.get("claim", ""),
                "evidence_summary": finding.get(
                    "evidence_summary",
                    "",
                ),
                "source_ids": finding.get(
                    "source_ids",
                    [],
                ),
                "status": finding.get(
                    "status",
                    "",
                ),
                "uncertainty": finding.get(
                    "uncertainty",
                    "",
                ),
                "conflicts": finding.get(
                    "conflicts",
                    "",
                ),
                "created_at": finding.get(
                    "created_at",
                    "",
                ),
                "updated_at": finding.get(
                    "updated_at",
                    "",
                ),
            }
        )

    findings.sort(
        key=lambda item: (
            str(item["created_at"]),
            str(item["finding_id"]),
        )
    )

    return {
        "ok": True,
        "findings": findings,
    }
