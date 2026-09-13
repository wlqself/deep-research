from datetime import datetime, timezone
from uuid import uuid4

from typing_extensions import TypedDict


MAX_ACTIVITY_EVENTS = 200


class ActivityEvent(TypedDict):
    id: str # 核心标识字段
    run_id: str # 关联的运行 ID
    timestamp: str # 时间字段
    kind: str # 分类与来源字段
    actor: str 
    status: str # 状态与展示字段
    label: str
    summary: str
    detail: str
    elapsed_ms: float | None # 性能字段

# 时间
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )

# 文本压缩与截断工具
def compact_text(value: str, *, max_length: int = 140) -> str:
    compacted = " ".join(value.split())

    if len(compacted) <= max_length:
        return compacted

    return f"{compacted[: max_length - 1]}…"


def build_activity_event(
    *,
    run_id: str, # 关联的运行 ID
    kind: str, # 分类与来源字段
    actor: str,
    status: str, # 状态与展示字段
    label: str,
    summary: str | None = None,
    detail: str = "",
    elapsed_ms: float | None = None, # 性能字段
) -> ActivityEvent:
    return {
        "id": uuid4().hex,
        "run_id": run_id,
        "timestamp": utc_now(),
        "kind": kind,
        "actor": actor,
        "status": status,
        "label": label,
        "summary": summary if summary is not None else label,
        "detail": detail,
        "elapsed_ms": elapsed_ms,
    }


def build_question_activity(
    question: str,
    *,
    run_id: str,
) -> ActivityEvent:
    return build_activity_event(
        run_id=run_id,
        kind="question",
        actor="user",
        status="completed",
        label="用户提出问题",
        summary=compact_text(question.strip()),
        detail=question.strip(),
    )

# 流式响应中的各种内部事件翻译成结构化的 ActivityEvent 对象，为前端提供统一的活动时间线/进度展示数据
def activity_from_stream_event(
    event: dict[str, object],
    *,
    run_id: str,
) -> ActivityEvent | None:
    event_type = event.get("type")

    if event_type == "plan_update":
        todos = event.get("todos")
        count = len(todos) if isinstance(todos, list) else 0
        return build_activity_event(
            run_id=run_id,
            kind="plan",
            actor="main",
            status="completed",
            label="研究计划已更新",
            detail=f"当前包含 {count} 个步骤",
        )

    if event_type == "subagent_start":
        agent_name = str(event.get("agent_name") or "researcher")
        description = str(event.get("description") or "")
        return build_activity_event(
            run_id=run_id,
            kind="agent",
            actor=agent_name,
            status="running",
            label=f"已分配给 {agent_name} 任务",
            summary=compact_text(description) or "等待任务描述",
            detail=description,
        )

    if event_type == "subagent_end":
        agent_name = str(event.get("agent_name") or "researcher")
        status = str(event.get("status") or "completed")
        source_ids = event.get("source_ids")
        source_count = (
            len(source_ids)
            if isinstance(source_ids, list)
            else 0
        )
        detail = (
            f"关联 {source_count} 个来源"
            if status == "completed"
            else str(event.get("error") or "任务未完成")
        )
        return build_activity_event(
            run_id=run_id,
            kind="agent",
            actor=agent_name,
            status=status,
            label=(
                f"{agent_name} 已完成任务"
                if status == "completed"
                else f"{agent_name} 任务{status}"
            ),
            summary=compact_text(detail),
            detail=detail,
        )

    if event_type in {"tool_start", "tool_end"}:
        name = str(event.get("name") or "unknown_tool")
        status = (
            "running"
            if event_type == "tool_start"
            else str(event.get("status") or "completed")
        )
        elapsed_ms = event.get("elapsed_ms")

        if name == "prepare_article_for_publication":
            if event_type == "tool_start":
                label = "开始整理文章草稿"
                detail = "正在根据研究 Artifact 准备 Article 草稿"
            elif status == "failed":
                label = "文章草稿整理失败"
                detail = "文章草稿未能准备完成"
            else:
                label = "文章草稿整理完成"
                detail = "草稿已准备完成，等待用户人工审批"

            return build_activity_event(
                run_id=run_id,
                kind="publishing",
                actor="main",
                status=status,
                label=label,
                detail=detail,
                elapsed_ms=(
                    float(elapsed_ms)
                    if isinstance(elapsed_ms, (int, float))
                    and not isinstance(elapsed_ms, bool)
                    else None
                ),
            )

        return build_activity_event(
            run_id=run_id,
            kind="tool",
            actor="agent",
            status=status,
            label=(
                f"开始使用 {name}"
                if event_type == "tool_start"
                else f"{name} 执行结束"
            ),
            elapsed_ms=(
                float(elapsed_ms)
                if isinstance(elapsed_ms, (int, float))
                and not isinstance(elapsed_ms, bool)
                else None
            ),
        )

    if event_type == "memory_compacted":
        return build_activity_event(
            run_id=run_id,
            kind="memory",
            actor="system",
            status="completed",
            label="上下文已整理",
        )

    if event_type == "artifact_saved":
        filename = str(event.get("filename") or "")
        return build_activity_event(
            run_id=run_id,
            kind="artifact",
            actor="agent",
            status="completed",
            label="研究报告已保存",
            detail=filename,
        )

    if event_type == "done":
        goal_status = str(event.get("goal_status") or "completed")
        if goal_status == "waiting_for_user":
            return build_activity_event(
                run_id=run_id,
                kind="research",
                actor="main",
                status="waiting_for_user",
                label="等待用户确认",
            )
        if goal_status == "completed_with_failure":
            error_codes = event.get("error_codes")
            detail = (
                ", ".join(str(code) for code in error_codes)
                if isinstance(error_codes, list)
                else ""
            )
            return build_activity_event(
                run_id=run_id,
                kind="research",
                actor="main",
                status="completed_with_failure",
                label="处理结束，但目标未完成",
                detail=detail,
            )
        return build_activity_event(
            run_id=run_id,
            kind="research",
            actor="main",
            status="completed",
            label="处理完成",
        )

    if event_type == "cancelled":
        return build_activity_event(
            run_id=run_id,
            kind="research",
            actor="main",
            status="cancelled",
            label="研究已取消",
        )

    if event_type == "error":
        return build_activity_event(
            run_id=run_id,
            kind="research",
            actor="main",
            status="failed",
            label="研究失败",
        )

    return None


def merge_activity_events(
    current: list[ActivityEvent] | None,
    update: list[ActivityEvent] | None,
) -> list[ActivityEvent]:
    merged: dict[str, ActivityEvent] = {}
    # 把 current 和 update 两个数组合并成一个可迭代序列
    for event in [*(current or []), *(update or [])]:
        if not isinstance(event, dict):
            continue

        event_id = event.get("id")
        timestamp = event.get("timestamp")

        if not isinstance(event_id, str) or not event_id:
            continue

        if not isinstance(timestamp, str) or not timestamp:
            continue
        # 存入字典（去重）
        merged[event_id] = dict(event)
    # 排序
    ordered = sorted(
        merged.values(),
        key=lambda item: (item["timestamp"], item["id"]),
    )
    return ordered[-MAX_ACTIVITY_EVENTS:]
