from typing import Any

from .stream_state import (
    researcher_result_from_task_output,
    terminal_task_record,
    utc_now,
)
from ..state.research import TaskRecord

# 子任务启动信息的构造工厂
def build_task_start(
    run_id: str,
    input_data: dict[str, object],
) -> tuple[str, dict[str, str], dict[str, object]]:
    # 在 LangGraph 中，每次工具调用都有唯一的 run_id，天然适合做任务 ID，所以把run_id 直接当作 task_id
    task_id = run_id
    agent_name = str(input_data.get("subagent_type", "unknown"))
    description = str(input_data.get("description", ""))
    now = utc_now()

    info = {
        "agent_name": agent_name,
        "description": description,
        "created_at": now,
        "started_at": now,
    }

    event = {
        "type": "subagent_start",
        "task_id": task_id,
        "agent_name": agent_name,
        "description": description,
        "status": "running",
    }

    return task_id, info, event

"""
build_task_terminal 是一个子任务终态信息构造工厂：
它解析 task 工具的输出，根据成功或失败分别构造终态任务记录和 subagent_end 前端事件，
并提取产出的发现/来源 ID，为上层调度提供完整的收尾数据。
"""
def build_task_terminal(
    task_id: str,
    info: dict[str, str],
    event_data: object,
) -> tuple[TaskRecord, dict[str, object], str, list[str], list[str]]:
    result, error = researcher_result_from_task_output(event_data)

    if result is None:
        record = terminal_task_record(
            task_id,
            info,
            status="failed",
            finding_ids=[],
            source_ids=[],
            error=error,
        )
        event = {
            "type": "subagent_end",
            "task_id": task_id,
            "agent_name": info.get("agent_name", "unknown"),
            "status": "failed",
            "finding_ids": [],
            "source_ids": [],
            "error": error,
        }
        return record, event, error, [], []

    finding_ids = list(result.finding_ids)
    source_ids = list(result.source_ids)
    record = terminal_task_record(
        task_id,
        info,
        status="completed",
        finding_ids=finding_ids,
        source_ids=source_ids,
        error="",
    )
    event = {
        "type": "subagent_end",
        "task_id": task_id,
        "agent_name": info.get("agent_name", "unknown"),
        "status": "completed",
        "finding_ids": finding_ids,
        "source_ids": source_ids,
        "error": "",
    }
    return record, event, "", finding_ids, source_ids

# 子任务失败信息的处理
def build_task_failure(
    task_id: str,
    info: dict[str, str],
    error: str,
) -> tuple[TaskRecord, dict[str, object]]:
    record = terminal_task_record(
        task_id,
        info,
        status="failed",
        finding_ids=[],
        source_ids=[],
        error=error,
    )
    event = {
        "type": "subagent_end",
        "task_id": task_id,
        "agent_name": info.get("agent_name", "unknown"),
        "status": "failed",
        "finding_ids": [],
        "source_ids": [],
        "error": error,
    }
    return record, event
