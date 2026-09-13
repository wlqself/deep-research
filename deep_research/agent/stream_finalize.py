import logging
from typing import Any

from .stream_state import (
    elapsed_ms,
    persist_terminal_tasks,
    terminal_task_record,
)
from .stream_output import verified_answer_suffix
from ..state.research import TaskRecord
from ..log.logging_utils import log_event
from ..state.access import sources_for_thread

# 流式研究的收尾调度
async def finalize_stream(
    *,
    agent: Any,
    thread_id: str,
    answer_chunks: list[str],
    terminal_tasks: dict[str, TaskRecord],
    unresolved_tool_failures: dict[str, str],
    waiting_for_user: bool,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    if terminal_tasks and not waiting_for_user:
        await persist_terminal_tasks(
            agent,
            thread_id,
            terminal_tasks,
            logger=logger,
        )
    elif terminal_tasks and waiting_for_user:
        # Updating an interrupted LangGraph checkpoint clears its native
        # interrupt. Defer terminal-task persistence until the graph resumes
        # so the HITL endpoint can still find the pending task.
        logger.info(
            "defer terminal task persistence while native HITL is waiting"
        )

    raw_answer = "".join(answer_chunks)
    # 获取来源信息
    sources = await _sources_for_thread(agent, thread_id)
    suffix = verified_answer_suffix(raw_answer, sources)
    # 初始化事件列表
    events: list[dict[str, object]] = []

    if suffix:
        events.append({"type": "text", "text": suffix})

    task_error_codes = [
        str(task.get("error") or "subagent_task_failed")
        for task in terminal_tasks.values()
        if task.get("status") == "failed"
    ]
    error_codes = list(
        dict.fromkeys(
            [
                *unresolved_tool_failures.values(),
                *task_error_codes,
            ]
        )
    )
    goal_status = (
        "waiting_for_user"
        if waiting_for_user
        else "completed_with_failure"
        if error_codes
        else "completed"
    )
    events.append(
        {
            "type": "done",
            "goal_status": goal_status,
            "error_codes": error_codes,
        }
    )
    return events

# 内外转化
async def _sources_for_thread(agent: Any, thread_id: str) -> dict[str, object]:
    return await sources_for_thread(agent, thread_id)


async def cancelled_task_records(
    *,
    agent: Any,
    thread_id: str,
    active_subagents: dict[str, dict[str, str]],
    terminal_tasks: dict[str, TaskRecord],
    tool_started_at: dict[str, float],
    logger: logging.Logger,
) -> dict[str, TaskRecord]:
    # 构造取消任务记录字典
    cancelled_tasks = {
        task_id: terminal_task_record(
            task_id,
            info,
            status="cancelled",
            finding_ids=[],
            source_ids=[],
            error="user_cancelled",
        )
        for task_id, info in active_subagents.items()
    }
    # 合并任务字典
    merged_tasks = {
        **terminal_tasks,
        **cancelled_tasks,
    }
    # 记录取消日志
    for task_id, info in cancelled_tasks.items():
        log_event(
            logger,
            logging.WARNING,
            "subagent.task.cancelled",
            task_id=task_id,
            agent_name=info.get("agent_name", "unknown"),
            status="cancelled",
            error_code="user_cancelled",
        )

        started = tool_started_at.pop(task_id, None)
        log_event(
            logger,
            logging.WARNING,
            "tool.cancelled",
            tool_name="task",
            task_id=task_id,
            status="cancelled",
            error_code="user_cancelled",
            elapsed_ms=elapsed_ms(started),
        )
    # 持久化终态任务
    await persist_terminal_tasks(
        agent,
        thread_id,
        merged_tasks,
        logger=logger,
    )
    return merged_tasks
