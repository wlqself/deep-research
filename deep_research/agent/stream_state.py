import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import ValidationError

from .results import ResearcherResult
from ..state.research import TaskRecord
from ..state.runtime import agent_config


TERMINAL_TASK_PERSIST_TIMEOUT_SECONDS = 2.0

# 获取时间
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# 获取执行时间
def elapsed_ms(started_at: float | None) -> float | None:
    if started_at is None:
        return None

    return round((time.perf_counter() - started_at) * 1000, 3)

# 从 task 工具（子 agent 调度）的事件数据中提取工具消息。
def task_tool_message(event_data: object) -> ToolMessage | None:
    if not isinstance(event_data, dict):
        return None

    output = event_data.get("output")

    if isinstance(output, ToolMessage):
        return output

    if not isinstance(output, Command) or not isinstance(output.update, dict):
        return None

    messages = output.update.get("messages", [])

    if not isinstance(messages, list):
        return None

    return next(
        (message for message in messages if isinstance(message, ToolMessage)),
        None,
    )

# 子任务结果解析与错误分类器
def researcher_result_from_task_output(
    event_data: object,
) -> tuple[ResearcherResult | None, str]:
    message = task_tool_message(event_data)

    if message is None or not isinstance(message.content, str):
        return None, "invalid_researcher_result"

    if message.status == "error":
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            return None, "invalid_researcher_result"

        if (
            isinstance(payload, dict)
            and payload.get("error") == "parallel_task_limit_reached"
        ):
            return None, "parallel_task_limit_reached"

        return None, "invalid_researcher_result"

    try:
        return ResearcherResult.model_validate_json(message.content), ""
    except (TypeError, ValueError, ValidationError):
        return None, "invalid_researcher_result"

# 记录终态任务
def terminal_task_record(
    task_id: str,
    info: dict[str, str],
    *,
    status: str,
    finding_ids: list[str],
    source_ids: list[str],
    error: str,
) -> TaskRecord:
    now = utc_now()

    return {
        "task_id": task_id,
        "agent_name": info.get("agent_name", "unknown"),
        "description": info.get("description", ""),
        "status": status,
        "created_at": info.get("created_at", now),
        "started_at": info.get("started_at", ""),
        "completed_at": now,
        "updated_at": now,
        "finding_ids": list(finding_ids),
        "source_ids": list(source_ids),
        "error": error,
    }

# 任务安全取消与清理工具，会等待任务到达终态
async def cancel_and_drain(
    task: asyncio.Task[Any],
    *,
    logger: logging.Logger,
) -> None:
    if not task.done():
        task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("terminal task persistence failed while cancelling")

# 带超时与取消保护的持久化函数
async def persist_terminal_tasks(
    agent: Any,
    thread_id: str,
    tasks: dict[str, TaskRecord],
    *,
    logger: logging.Logger,
) -> None:
    if not tasks:
        return
    
    # 启动后台任务
    persist_task = asyncio.create_task(
        agent.aupdate_state(
            agent_config(thread_id),
            {"tasks": tasks},
        )
    )

    try:
        await asyncio.wait_for(
            asyncio.shield(persist_task),
            timeout=TERMINAL_TASK_PERSIST_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        # 这里再给一次机会：重新 wait_for(shield(persist_task), timeout)，让持久化尽量完成
        try:
            await asyncio.wait_for(
                asyncio.shield(persist_task),
                timeout=TERMINAL_TASK_PERSIST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            await cancel_and_drain(persist_task, logger=logger)
            logger.exception(
                "timed out while persisting cancelled research tasks"
            )
        except asyncio.CancelledError:
            await cancel_and_drain(persist_task, logger=logger)
            logger.exception(
                "cancelled again while persisting research tasks"
            )
        except Exception:
            logger.exception(
                "failed while finishing cancelled research task persistence"
            )

        raise

    except asyncio.TimeoutError:
        await cancel_and_drain(persist_task, logger=logger)
        logger.error("timed out while persisting research tasks")
    except Exception:
        logger.exception("failed to persist terminal research tasks")
