import json
from typing import Any


_PUBLISHING_CARD_CHANNELS = {
    "local_static_site",
    "wechat_official_account",
    "xiaohongshu",
    "douyin",
}
# 工具输入的"预览摘要"写入日志。
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


def tool_result_outcome(
    event_data: Any,
) -> tuple[str, str | None, bool | None]:
    """Distinguish transport completion from a tool's business outcome."""

    if not isinstance(event_data, dict):
        return "completed", None, None

    output = event_data.get("output")
    message_status = getattr(output, "status", None)
    if hasattr(output, "content"):
        output = output.content

    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            output = None

    if message_status == "error":
        return "failed", "tool_execution_failed", None

    if not isinstance(output, dict):
        return "completed", None, None

    resolution_status = output.get("resolution_status")
    if resolution_status in {"not_found", "ambiguous"}:
        raw_error_code = output.get("error_code")
        error_code = (
            raw_error_code
            if isinstance(raw_error_code, str) and raw_error_code
            else f"publication_target_{resolution_status}"
        )
        return "failed", error_code, False

    if output.get("ok") is not False:
        return "completed", None, None

    raw_error_code = output.get("error_code", output.get("error"))
    error_code = (
        raw_error_code
        if isinstance(raw_error_code, str) and raw_error_code
        else "tool_business_failed"
    )
    retryable = output.get("retryable")
    return (
        "failed",
        error_code,
        retryable if isinstance(retryable, bool) else None,
    )

TODO_STATUSES = {
    "pending",
    "in_progress",
    "completed",
}

# 把原始的待办事项列表清洗、裁剪成一个安全的预览版本，用于日志记录或 API 响应。
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

# 从 artifact（产物/制品）相关的事件数据中提取一个安全、精简的预览信息，用于日志记录或接口返回。
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


def generated_image_preview(
    event_data: Any,
) -> dict[str, object]:
    """Expose generated attachment metadata without provider URLs or paths."""
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
    if not isinstance(output, dict) or output.get("ok") is not True:
        return {}

    images: list[dict[str, object]] = []
    raw_images = output.get("images", [])
    if isinstance(raw_images, list):
        for image in raw_images[:30]:
            if not isinstance(image, dict):
                continue
            attachment_id = image.get("attachment_id")
            filename = image.get("filename")
            content_type = image.get("content_type")
            if not all(
                isinstance(value, str) and value
                for value in (attachment_id, filename, content_type)
            ):
                continue
            images.append({
                "attachment_id": attachment_id,
                "filename": filename,
                "content_type": content_type,
                "size_bytes": image.get("size_bytes", 0),
            })
    if not images:
        return {}
    return {
        "model": output.get("model") if isinstance(output.get("model"), str) else None,
        "attachment_ids": [image["attachment_id"] for image in images],
        "images": images,
    }


def publishing_intent_preview(
    event_data: Any,
) -> dict[str, object]:
    """Expose only safe intent-resolution data to the private frontend."""

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

    if not isinstance(output, dict) or output.get("ok") is not True:
        return {}

    raw_action = output.get("action")
    # The initial tool request is a request for human approval.  The private
    # frontend handles that request with the same explicit approve action as
    # the intent-resolution tool.
    action = (
        "approve"
        if raw_action == "request_publication"
        else raw_action
    )
    resolution_status = output.get("resolution_status")
    if action not in {"none", "status", "approve", "reject", "resume"}:
        return {}
    if resolution_status not in {
        "read_only",
        "resolved",
        "not_found",
        "ambiguous",
    }:
        return {}

    def safe_target(value: Any) -> dict[str, object] | None:
        if not isinstance(value, dict):
            return None

        required = (
            "approval_id",
            "article_id",
            "article_title",
            "article_slug",
            "article_version",
            "channel",
            "approval_status",
        )
        if not all(
            isinstance(value.get(field), str) and value[field]
            for field in required[:4]
        ):
            return None
        if not isinstance(value.get("article_version"), int):
            return None
        if value.get("channel") not in _PUBLISHING_CARD_CHANNELS:
            return None
        if not isinstance(value.get("approval_status"), str):
            return None

        publication_status = value.get("publication_status")
        if publication_status is not None and not isinstance(
            publication_status,
            str,
        ):
            return None

        publication_error_code = value.get("publication_error_code")
        if publication_error_code is not None and not isinstance(
            publication_error_code,
            str,
        ):
            return None

        raw_target_attachment_ids = value.get("attachment_ids", [])
        target_attachment_ids = (
            list(
                dict.fromkeys(
                    item.strip()
                    for item in raw_target_attachment_ids
                    if isinstance(item, str) and item.strip()
                )
            )[:30]
            if isinstance(raw_target_attachment_ids, list)
            else []
        )

        result = {
            field: value.get(field)
            for field in (
                "approval_id",
                "article_id",
                "article_title",
                "article_slug",
                "article_version",
                "channel",
                "approval_status",
                "workflow_status",
                "interaction_id",
                "interaction_status",
                "state_source",
                "requires_user_confirmation",
                "publication_status",
                "publication_id",
                "publication_error_code",
            )
        }
        result["attachment_ids"] = target_attachment_ids
        return result

    target = safe_target(output.get("target"))
    candidates = []
    raw_candidates = output.get("candidates", [])
    if isinstance(raw_candidates, list):
        for candidate in raw_candidates:
            safe_candidate = safe_target(candidate)
            if safe_candidate is not None:
                candidates.append(safe_candidate)

    raw_attachment_ids = output.get("attachment_ids", [])
    attachment_ids = []
    if isinstance(raw_attachment_ids, list):
        attachment_ids = [
            item.strip()
            for item in raw_attachment_ids
            if isinstance(item, str) and item.strip()
        ][:30]

    return {
        "action": action,
        "resolution_status": resolution_status,
        "target": target,
        "candidates": candidates,
        "requires_user_confirmation": (
            output.get("requires_user_confirmation") is True
        ),
        "interaction_id": (
            output.get("interaction_id")
            if isinstance(output.get("interaction_id"), str)
            and output.get("interaction_id")
            else None
        ),
        "interaction_status": (
            output.get("interaction_status")
            if isinstance(output.get("interaction_status"), str)
            else None
        ),
        "attachment_ids": attachment_ids,
    }


def publication_attachment_selection_preview(
    event_data: Any,
) -> dict[str, object]:
    """Expose a safe conversation card for selecting publication images."""
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
    if output.get("ok") is not True or output.get("attachment_selection_required") is not True:
        return {}

    target = output.get("target")
    if not isinstance(target, dict):
        return {}
    required = ("article_id", "article_title", "article_version", "channel")
    if not all(isinstance(target.get(field), (str, int)) for field in required):
        return {}
    if not isinstance(target.get("article_title"), str) or not target["article_title"]:
        return {}
    if not isinstance(target.get("article_id"), str) or not target["article_id"]:
        return {}
    if not isinstance(target.get("article_version"), int):
        return {}
    if target.get("channel") not in {
        "wechat_official_account",
        "xiaohongshu",
        "douyin",
    }:
        return {}

    images: list[dict[str, object]] = []
    raw_images = output.get("attachments", [])
    if isinstance(raw_images, list):
        for image in raw_images:
            if not isinstance(image, dict):
                continue
            attachment_id = image.get("attachment_id")
            filename = image.get("filename")
            content_type = image.get("content_type")
            size_bytes = image.get("size_bytes")
            if not all(
                isinstance(value, str) and value
                for value in (attachment_id, filename, content_type)
            ):
                continue
            if not isinstance(size_bytes, int) or size_bytes < 0:
                continue
            images.append(
                {
                    "attachment_id": attachment_id,
                    "filename": filename,
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                }
            )

    raw_ids = output.get("attachment_ids", [])
    attachment_ids = [
        item.strip()
        for item in raw_ids
        if isinstance(item, str) and item.strip()
    ][:30] if isinstance(raw_ids, list) else []

    return {
        "article_id": target["article_id"],
        "article_title": target["article_title"],
        "article_version": target["article_version"],
        "channel": target["channel"],
        "attachments": images,
        "attachment_ids": attachment_ids,
    }


def publishing_interrupt_preview(value: Any) -> dict[str, object]:
    """Expose a safe publication card from a native Graph interrupt."""

    if not isinstance(value, dict):
        return {}
    if value.get("kind") != "publication_approval":
        return {}

    preview = publishing_intent_preview({"output": value})
    if preview:
        preview["native_interrupt"] = True
    return preview


def publishing_approval_status_preview(
    event_data: Any,
) -> dict[str, object]:
    """Expose recoverable HITL cards from the read-only status tool."""

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
    if not isinstance(output, dict) or output.get("ok") is not True:
        return {}

    cards: list[dict[str, object]] = []
    raw_cards = output.get("recovery_cards", [])
    if isinstance(raw_cards, list):
        for raw_card in raw_cards:
            preview = publishing_intent_preview(
                {"output": {"ok": True, **raw_card}}
            )
            if preview:
                preview["recovery"] = True
                cards.append(preview)

    return {"cards": cards}

