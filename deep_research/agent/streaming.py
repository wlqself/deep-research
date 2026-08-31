import json
import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import ValidationError

from .results import ResearcherResult
from .tool_events import artifact_preview, todo_preview, tool_input_preview
from ..citations import append_verified_sources
from ..state.access import context_for_thread, sources_for_thread
from ..state.research import TaskRecord
from ..state.runtime import agent_config, normalize_question

_TERMINAL_TASK_PERSIST_TIMEOUT_SECONDS = 2.0
logger = logging.getLogger(__name__)

async def _cancel_and_drain(
    task: asyncio.Task[Any],
) -> None:
    if not task.done():
        task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception(
            "terminal task persistence failed while cancelling"
        )

async def _persist_terminal_tasks(
    agent: Any,
    thread_id: str,
    tasks: dict[str, TaskRecord],
) -> None:
    if not tasks:
        return
    # 创建独立保存任务
    persist_task = asyncio.create_task(
        agent.aupdate_state(
            agent_config(thread_id),
            {"tasks": tasks},
        )
    )

    try: # 正常等待保存
        await asyncio.wait_for(
            asyncio.shield(persist_task),
            timeout=_TERMINAL_TASK_PERSIST_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError: # 外层等待时收到取消
        try:
            await asyncio.wait_for(
                asyncio.shield(persist_task),
                timeout=_TERMINAL_TASK_PERSIST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError: # 保存超时
            await _cancel_and_drain(persist_task)
            logger.exception(
                "timed out while persisting cancelled research tasks"
            )
        except asyncio.CancelledError:
            await _cancel_and_drain(persist_task)
            logger.exception(
                "cancelled again while persisting research tasks"
            )
        except Exception: # 保存本身失败
            logger.exception(
                "failed while finishing cancelled research task persistence"
            )

        raise

    except asyncio.TimeoutError:
        await _cancel_and_drain(persist_task)
        logger.error("timed out while persisting research tasks")
    except Exception:
        logger.exception("failed to persist terminal research tasks")
            
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _task_tool_message(event_data: object) -> ToolMessage | None:
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
        (
            message
            for message in messages
            if isinstance(message, ToolMessage)
        ),
        None,
    )


def _researcher_result_from_task_output(
    event_data: object,
) -> tuple[ResearcherResult | None, str]:
    message = _task_tool_message(event_data)

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


def _terminal_task_record(
    task_id: str,
    info: dict[str, str],
    *,
    status: str,
    finding_ids: list[str],
    source_ids: list[str],
    error: str,
) -> TaskRecord:
    now = _utc_now()

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


async def stream_research_with_agent(
    agent: Any,
    question: str,
    thread_id: str,
):
    question = normalize_question(question)
    context = await context_for_thread(agent, thread_id)
    answer_chunks: list[str] = []

    async for token, metadata in agent.astream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config=agent_config(thread_id),
        context=context,
        stream_mode="messages",
    ):
        if not isinstance(metadata, dict):
            continue

        if metadata.get("lc_agent_name") == "researcher":
            continue

        if metadata.get("langgraph_node") != "model":
            continue

        text = getattr(token, "text", "")

        if text:
            answer_chunks.append(text)
            yield text

    raw_answer = "".join(answer_chunks)
    sources = await sources_for_thread(agent, thread_id)
    verified_answer = append_verified_sources(raw_answer, sources)
    base_answer = raw_answer.rstrip()

    if verified_answer.startswith(base_answer):
        suffix = verified_answer[len(base_answer):]

        if suffix:
            yield suffix


async def stream_research_events_with_agent(
    agent: Any,
    question: str,
    thread_id: str,
) -> AsyncIterator[dict[str, object]]:
    question = normalize_question(question)
    context = await context_for_thread(agent, thread_id)
    answer_chunks: list[str] = []
    last_todos: list[dict[str, str]] = []
    active_subagents: dict[str, dict[str, str]] = {}
    terminal_tasks: dict[str, TaskRecord] = {}
    try:
        async for event in agent.astream_events(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            config=agent_config(thread_id),
            context=context,
            version="v2",
        ):
            event_metadata = event.get("metadata", {})

            if not isinstance(event_metadata, dict):
                event_metadata = {}

            if event_metadata.get("lc_agent_name") == "researcher":
                continue

            event_name = event.get("event")
            tool_name = str(event.get("name", ""))
            event_data = event.get("data", {})

            if (
                event_name == "on_chat_model_end"
                and event_metadata.get("lc_source") == "summarization"
            ):
                yield {
                    "type": "memory_compacted",
                    "message": "当前会话已进行一次上下文整理。",
                }
                continue

            if event_name == "on_chat_model_stream":
                if not isinstance(event_data, dict):
                    continue

                if event_metadata.get("langgraph_node") != "model":
                    continue

                chunk = event_data.get("chunk")
                text = getattr(chunk, "text", "")

                if text:
                    answer_chunks.append(text)
                    yield {
                        "type": "text",
                        "text": text,
                    }

            elif event_name == "on_tool_start":
                input_data = (
                    event_data.get("input", {})
                    if isinstance(event_data, dict)
                    else {}
                )

                if tool_name == "task":
                    task_id = str(event.get("run_id", ""))
                    agent_name = str(input_data.get("subagent_type", "unknown"))
                    description = str(input_data.get("description", ""))
                    now = _utc_now()

                    active_subagents[task_id] = {
                        "agent_name": agent_name,
                        "description": description,
                        "created_at": now,
                        "started_at": now,
                    }

                    yield {
                        "type": "subagent_start",
                        "task_id": task_id,
                        "agent_name": agent_name,
                        "description": description,
                        "status": "running",
                    }
                    continue

                if tool_name == "write_todos":
                    raw_todos = (
                        input_data.get("todos", [])
                        if isinstance(input_data, dict)
                        else []
                    )
                    todos = todo_preview(raw_todos)

                    if todos and todos != last_todos:
                        last_todos = todos
                        yield {
                            "type": "plan_update",
                            "todos": todos,
                        }

                yield {
                    "type": "tool_start",
                    "name": tool_name,
                    "input": tool_input_preview(tool_name, event_data),
                }

            elif event_name == "on_tool_end":
                if tool_name == "task":
                    task_id = str(event.get("run_id", ""))
                    info = active_subagents.pop(task_id, {})
                    result, error = _researcher_result_from_task_output(event_data)

                    if result is None:
                        terminal_tasks[task_id] = _terminal_task_record(
                            task_id,
                            info,
                            status="failed",
                            finding_ids=[],
                            source_ids=[],
                            error= error,
                        )
                        yield {
                            "type": "subagent_end",
                            "task_id": task_id,
                            "agent_name": info.get("agent_name", "unknown"),
                            "status": "failed",
                            "finding_ids": [],
                            "source_ids": [],
                            "error": error,
                        }
                        continue

                    finding_ids = list(result.finding_ids)
                    source_ids = list(result.source_ids)
                    terminal_tasks[task_id] = _terminal_task_record(
                        task_id,
                        info,
                        status="completed",
                        finding_ids=finding_ids,
                        source_ids=source_ids,
                        error="",
                    )
                    yield {
                        "type": "subagent_end",
                        "task_id": task_id,
                        "agent_name": info.get("agent_name", "unknown"),
                        "status": "completed",
                        "finding_ids": finding_ids,
                        "source_ids": source_ids,
                        "error": "",
                    }
                    continue

                yield {
                    "type": "tool_end",
                    "name": tool_name,
                    "status": "completed",
                }

                if tool_name == "save_report":
                    artifact = artifact_preview(event_data)

                    if artifact:
                        yield {
                            "type": "artifact_saved",
                            **artifact,
                        }

            elif event_name == "on_tool_error":
                if tool_name == "task":
                    task_id = str(event.get("run_id", ""))
                    info = active_subagents.pop(task_id, {})
                    terminal_tasks[task_id] = _terminal_task_record(
                        task_id,
                        info,
                        status="failed",
                        finding_ids=[],
                        source_ids=[],
                        error="task_execution_failed",
                    )
                    yield {
                        "type": "subagent_end",
                        "task_id": task_id,
                        "agent_name": info.get("agent_name", "unknown"),
                        "status": "failed",
                        "finding_ids": [],
                        "source_ids": [],
                        "error": "task_execution_failed",
                    }
                    continue

                yield {
                    "type": "tool_end",
                    "name": tool_name,
                    "status": "failed",
                }

        if terminal_tasks:
            await _persist_terminal_tasks(
                agent,
                thread_id,
                terminal_tasks,
            )

        raw_answer = "".join(answer_chunks)
        sources = await sources_for_thread(agent, thread_id)
        verified_answer = append_verified_sources(raw_answer, sources)
        base_answer = raw_answer.rstrip()

        if verified_answer.startswith(base_answer):
            suffix = verified_answer[len(base_answer):]

            if suffix:
                yield {
                    "type": "text",
                    "text": suffix,
                }

        yield {
            "type": "done",
        }

    except asyncio.CancelledError:
        cancelled_tasks = {}

        for task_id, info in active_subagents.items():
            if task_id in terminal_tasks:
                continue

            cancelled_tasks[task_id] = _terminal_task_record(
                task_id,
                info,
                status="cancelled",
                finding_ids=[],
                source_ids=[],
                error="user_cancelled",
            )

        terminal_tasks = {
            **terminal_tasks,
            **cancelled_tasks,
        }

        await _persist_terminal_tasks(
            agent,
            thread_id,
            terminal_tasks,
        )

        raise
