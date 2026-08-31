from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ..context import ResearchContext


def _clean_items(values: list[str]) -> list[str]:
    return [
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip()
    ]


@tool
def assess_research(
    covered_points: list[str],
    knowledge_gaps: list[str],
    source_conflicts: list[str],
    next_action: Literal["search", "read", "answer"],
    runtime: ToolRuntime[ResearchContext],
) -> dict[str, object]:
    """Record explicit research coverage, gaps, conflicts, and next action."""

    context = runtime.context

    sources = runtime.state.get(
        "sources",
        {},
    )

    source_count = (
        len(sources)
        if isinstance(sources, dict)
        else 0
    )
    return {
        "ok": True,
        "type": "research_assessment",
        "covered_points": _clean_items(covered_points),
        "knowledge_gaps": _clean_items(knowledge_gaps),
        "source_conflicts": _clean_items(source_conflicts),
        "next_action": next_action,
        "research_state": {
            "search_count": context.search_count,
            "page_read_count": context.page_read_count,
            "source_count": source_count,
        },
    }
