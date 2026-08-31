import json
from typing import Any

def tool_input_preview(
    name: str,
    event_data: Any,
) -> dict[str, object]:
    if not isinstance(event_data, dict):
        return {}

    raw_input = event_data.get("input", {})

    if not isinstance(raw_input, dict):
        return {}

    if name == "web_search":
        return {
            "query": str(raw_input.get("query", ""))[:240],
        }

    if name == "read_page":
        return {
            "source_id": str(
                raw_input.get("source_id", "")
            ),
        }

    if name == "assess_research":
        gaps = raw_input.get("knowledge_gaps", [])
        conflicts = raw_input.get("source_conflicts", [])

        return {
            "next_action": raw_input.get("next_action"),
            "knowledge_gap_count": (
                len(gaps) if isinstance(gaps, list) else 0
            ),
            "conflict_count": (
                len(conflicts)
                if isinstance(conflicts, list)
                else 0
            ),
        }

    return {}

TODO_STATUSES = {
    "pending",
    "in_progress",
    "completed",
}


def todo_preview(raw_todos: Any) -> list[dict[str, str]]:
    if not isinstance(raw_todos, list):
        return []

    cleaned: list[dict[str, str]] = []

    for item in raw_todos:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        status = item.get("status")

        if not isinstance(content, str):
            continue

        if status not in TODO_STATUSES:
            continue

        cleaned.append(
            {
                "content": content[:240],
                "status": status,
            }
        )

    return cleaned

def artifact_preview(
        event_data: Any,
) -> dict[str, str]:
    if not isinstance(event_data, dict):
        return {}

    output = event_data.get("output")

    if hasattr(output, "content"):
        output = output.content

    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return {}

    if not isinstance(output, dict):
        return {}

    if output.get("ok") is not True:
        return {}

    fields = (
        "artifact_id",
        "filename",
        "download_url",
    )

    if not all(
        isinstance(output.get(field), str)
        and output[field]
        for field in fields
    ):
        return {}

    return {
        field: output[field]
        for field in fields
    }

