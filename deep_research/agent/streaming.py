import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from langgraph.errors import GraphInterrupt

from .tool_events import (
    artifact_preview,
    generated_image_preview,
    publication_attachment_selection_preview,
    publishing_approval_status_preview,
    publishing_interrupt_preview,
    publishing_intent_preview,
    todo_preview,
    tool_input_preview,
    tool_result_outcome,
)
from ..state.access import context_for_thread, sources_for_thread
from ..state.research import TaskRecord
from ..state.runtime import agent_config, normalize_question
from ..context import ResearchContext
from ..log.logging_utils import log_event
from .stream_state import (
    cancel_and_drain,
    elapsed_ms,
    persist_terminal_tasks,
    researcher_result_from_task_output,
    terminal_task_record,
    task_tool_message,
    utc_now,
)
from .message_stream import (
    stream_research_with_agent as _stream_research_with_agent,
)
from .event_dispatch import unpack_stream_event
from .stream_output import verified_answer_suffix
from .task_events import (
    build_task_failure,
    build_task_start,
    build_task_terminal,
)
from .model_events import (
    is_summarization_end,
    model_text_from_stream,
)
from .tool_timing import mark_tool_started, take_tool_elapsed
from .stream_finalize import (
    cancelled_task_records,
    finalize_stream,
)
from .repetition_guard import StreamOutputGuard
from ..config import settings
from .observability import LifecycleTracker, safe_model_error_code
from .image_context import (
    image_context_instruction,
    user_message_with_image_ids,
)

# 兼容旧的模块级私有导入路径；事件主循环仍使用旧签名。
async def _cancel_and_drain(
    task: asyncio.Task[Any],
) -> None:
    await cancel_and_drain(task, logger=logger)


async def _persist_terminal_tasks(
    agent: Any,
    thread_id: str,
    tasks: dict[str, TaskRecord],
) -> None:
    await persist_terminal_tasks(
        agent,
        thread_id,
        tasks,
        logger=logger,
    )


_utc_now = utc_now
_elapsed_ms = elapsed_ms
_task_tool_message = task_tool_message
_researcher_result_from_task_output = researcher_result_from_task_output
_terminal_task_record = terminal_task_record

_TERMINAL_TASK_PERSIST_TIMEOUT_SECONDS = 2.0
logger = logging.getLogger(__name__)

# Only these publication failures have a deterministic repair contract.  The
# approval tool returns the affected field and constraint to the model; all
# other non-retryable failures stop the turn immediately.
_PUBLICATION_REPAIRABLE_ERRORS = {
    "xiaohongshu_title_too_long",
    "xiaohongshu_content_too_long",
    "douyin_title_too_long",
    "wechat_title_too_long",
}


def _repair_key(tool_name: str, error_code: str | None) -> str:
    return f"{tool_name}:{error_code or 'tool_business_failed'}"


def _raw_interrupts_from_event(
    event_name: object,
    event_data: object,
) -> object | None:
    """Extract raw interrupt records from a v2 stream event."""

    raw_interrupts: object | None = None
    if event_name == "on_chain_stream" and isinstance(event_data, dict):
        chunk = event_data.get("chunk")
        if isinstance(chunk, dict):
            raw_interrupts = chunk.get("__interrupt__")
    elif event_name == "on_tool_error" and isinstance(event_data, dict):
        error = event_data.get("error")
        raw_interrupts = getattr(error, "interrupts", None)
        if raw_interrupts is None and isinstance(error, GraphInterrupt):
            raw_interrupts = error.args[0] if error.args else None

    return raw_interrupts


def _interrupt_values_from_event(
    event_name: object,
    event_data: object,
) -> list[Any]:
    """Extract LangGraph interrupt payloads from a v2 stream event."""

    raw_interrupts = _raw_interrupts_from_event(event_name, event_data)
    if raw_interrupts is None:
        return []
    records = (
        raw_interrupts
        if isinstance(raw_interrupts, (list, tuple))
        else (raw_interrupts,)
    )
    values: list[Any] = []
    for record in records:
        value = getattr(record, "value", None)
        if value is None and isinstance(record, dict):
            value = record.get("value")
        if value is not None:
            values.append(value)
    return values


def _interrupt_ids_from_event(
    event_name: object,
    event_data: object,
) -> list[str]:
    """Extract stable interrupt ids for stream/snapshot de-duplication."""

    raw_interrupts = _raw_interrupts_from_event(event_name, event_data)
    if raw_interrupts is None:
        return []
    records = (
        raw_interrupts
        if isinstance(raw_interrupts, (list, tuple))
        else (raw_interrupts,)
    )
    return [
        str(record.id)
        for record in records
        if getattr(record, "id", None)
    ]

# 兼容旧导入路径；纯文本 token 流已移动到 message_stream.py。
async def stream_research_with_agent(
    agent: Any,
    question: str,
    thread_id: str,
    *,
    selected_attachment_ids: tuple[str, ...] = (),
    attachment_selection_confirmed: bool = False,
):
    async for chunk in _stream_research_with_agent(
        agent,
        question,
        thread_id,
        selected_attachment_ids=selected_attachment_ids,
        attachment_selection_confirmed=attachment_selection_confirmed,
    ):
        yield chunk

"""
它监听 agent 执行过程中的所有 v2 事件，按类型过滤和解析，
实时向前端推送文本流、工具调用状态、子任务进度、待办更新和产物信息，
同时在后台收集终态任务用于持久化，最后追加来源引用并发送完成信号，
实现了"研究过程全透明、可观测、可恢复"的流式交互体验.
"""
async def stream_research_events_with_agent(
    agent: Any,
    question: str,
    thread_id: str,
    *,
    selected_attachment_ids: tuple[str, ...] = (),
    attachment_selection_confirmed: bool = False,
    observability: bool = False,
) -> AsyncIterator[dict[str, object]]:
    question = normalize_question(question)
    lifecycle = LifecycleTracker(thread_id=thread_id)
    if observability:
        yield lifecycle.event(
            "memory_recall_started",
            phase="memory_recall",
            agent_name="main",
            status="started",
            source="automatic",
        )
    try:
        context = await context_for_thread(
            agent,
            thread_id,
            selected_attachment_ids=selected_attachment_ids,
            attachment_selection_confirmed=attachment_selection_confirmed,
        )
    except Exception:
        context = ResearchContext.from_settings(
            selected_attachment_ids=selected_attachment_ids,
            attachment_selection_confirmed=attachment_selection_confirmed,
        )
        if observability:
            yield lifecycle.event(
                "memory_recall_completed",
                phase="memory_recall",
                agent_name="main",
                status="failed",
                source="automatic",
                recalled_count=0,
                error_code="memory_recall_failed",
            )
    else:
        if observability:
            yield lifecycle.event(
                "memory_recall_completed",
                phase="memory_recall",
                agent_name="main",
                status="completed",
                source="automatic",
                recalled_count=0,
            )
    answer_chunks: list[str] = [] # 收集模型生成的文本块
    last_todos: list[dict[str, str]] = [] # 上一次的待办列表
    active_subagents: dict[str, dict[str, str]] = {} # 正在运行的子 agent 任务信息
    terminal_tasks: dict[str, TaskRecord] = {} # 已进入终态（完成/失败）的任务记录，用于最后统一持久化
    tool_started_at: dict[str, float] = {} # 记录每个工具调用的开始时间（用 perf_counter 算耗时）
    # 用方通过 async for 接收一个个结构化的事件字典（如 {"type": "text", "text": "..."}），用于 SSE 推送到前端。
    seen_interrupt_ids: set[str] = set()
    interrupt_detected = False
    # Interaction cards are intentionally held until the model stream has
    # drained and the assistant message has been finalized.  Emitting them
    # from an on_tool_end/on_chain_stream handler makes the UI show a HITL
    # card while the same assistant turn is still writing.
    deferred_interaction_events: list[dict[str, object]] = []
    deferred_interaction_keys: set[str] = set()

    def defer_interaction(event: dict[str, object]) -> None:
        target = event.get("target")
        if isinstance(target, dict):
            identity = (
                target.get("approval_id")
                or target.get("article_id")
                or target.get("article_title")
            )
            version = target.get("article_version")
        else:
            identity = event.get("article_id") or event.get("article_title")
            version = event.get("article_version")
        key = ":".join(
            str(part or "")
            for part in (
                event.get("type"),
                event.get("interaction_id"),
                identity,
                version,
                event.get("action"),
            )
        )
        if key in deferred_interaction_keys:
            return
        deferred_interaction_keys.add(key)
        deferred_interaction_events.append(event)

    unresolved_tool_failures: dict[str, str] = {}
    repair_attempts: dict[str, int] = {}
    retry_attempts: dict[str, int] = {}
    output_guard = StreamOutputGuard(
        max_chars=settings.main_stream_max_chars,
        repetition_min_chars=settings.main_stream_repetition_min_chars,
        repetition_count=settings.main_stream_repetition_count,
    )
    image_instruction = image_context_instruction(context)
    model_question = (
        f"{question}\n\n{image_instruction}"
        if image_instruction
        else question
    )

    agent_input = {
        "messages": [
            user_message_with_image_ids(
                model_question,
                selected_attachment_ids,
                display_content=question,
            )
        ],
        "image_attachment_ids": list(selected_attachment_ids),
    }
    if attachment_selection_confirmed:
        agent_input["publication_attachment_ids"] = list(
            selected_attachment_ids
        )

    event_stream = agent.astream_events(
        agent_input,
        config=agent_config(thread_id),
        context=context,
        version="v2",
    )
    terminal_tasks_persisted = False
    try:
        async for event in event_stream:
            # 提取事件公共字段
            (
                event_metadata,
                is_researcher_event,
                event_name,
                tool_name,
                event_data,
            ) = unpack_stream_event(event)

            interrupt_values = _interrupt_values_from_event(
                event_name,
                event_data,
            )
            if interrupt_values:
                interrupt_detected = True
                interrupt_ids = _interrupt_ids_from_event(
                    event_name,
                    event_data,
                )
                is_new_interrupt = not interrupt_ids or any(
                    interrupt_id not in seen_interrupt_ids
                    for interrupt_id in interrupt_ids
                )
                if is_new_interrupt:
                    for value in interrupt_values:
                        interrupt_preview = publishing_interrupt_preview(value)
                        if interrupt_preview:
                            defer_interaction({
                                "type": "publishing_intent",
                                **interrupt_preview,
                            })
                seen_interrupt_ids.update(interrupt_ids)

                # Drain the native LangGraph event stream. The interrupt has
                # already stopped the graph; consuming the remaining terminal
                # events is required for the checkpointer to persist the
                # pending task so the later HITL resume can find it.
                continue

            #  处理记忆压缩事件
            if is_summarization_end(event_name, event_metadata):
                yield {
                    "type": "memory_compacted",
                    "message": "当前会话已进行一次上下文整理。",
                }
                continue

            # Public lifecycle events contain state only; model payloads stay internal.
            is_summary = event_metadata.get("lc_source") == "summarization"
            agent_name = "researcher" if is_researcher_event else "main"
            model_role = "researcher" if is_researcher_event else "main"
            if observability and event_name == "on_chat_model_start" and not is_summary:
                attempt = event_metadata.get("attempt", event_metadata.get("retry_attempt"))
                if isinstance(attempt, int) and attempt > 1:
                    yield lifecycle.event(
                        "model_retrying",
                        attempt=attempt,
                        max_attempts=event_metadata.get("max_attempts"),
                        reason_code="model_retry",
                        agent_name=agent_name,
                        model_role=model_role,
                        phase="model",
                        status="retrying",
                    )
                yield lifecycle.event(
                    "model_started",
                    agent_name=agent_name,
                    model_role=model_role,
                    phase="model",
                    status="started",
                    run_id=str(event.get("run_id") or lifecycle.run_id),
                )
                yield lifecycle.event(
                    "model_reasoning",
                    agent_name=agent_name,
                    model_role=model_role,
                    phase="model",
                    status="reasoning",
                    message=("主 Agent 正在分析请求" if agent_name == "main" else "Researcher 正在整理来源"),
                )
                continue

            if observability and event_name == "on_chat_model_end" and not is_summary:
                output = event_data.get("output") if isinstance(event_data, dict) else None
                output_type = "tool_call" if getattr(output, "tool_calls", None) else "text"
                if output is None:
                    output_type = "empty"
                yield lifecycle.event(
                    "model_completed",
                    agent_name=agent_name,
                    model_role=model_role,
                    phase="model",
                    status="completed",
                    output_type=output_type,
                    run_id=str(event.get("run_id") or lifecycle.run_id),
                )
                continue

            if observability and event_name in {"on_chat_model_error", "on_model_error"} and not is_summary:
                yield lifecycle.event(
                    "model_failed",
                    agent_name=agent_name,
                    model_role=model_role,
                    phase="model",
                    status="failed",
                    error_code=safe_model_error_code(event_data.get("error") if isinstance(event_data, dict) else None),
                    retryable=True,
                    run_id=str(event.get("run_id") or lifecycle.run_id),
                )
                continue

            # 处理模型流式输出
            if event_name == "on_chat_model_stream":
                text = model_text_from_stream(
                    event_data,
                    is_researcher_event=is_researcher_event,
                    event_metadata=event_metadata,
                )

                if text:
                    if observability and not is_researcher_event and text.strip():
                        first_token = lifecycle.first_token()
                        if first_token is not None:
                            yield first_token
                    answer_chunks.append(text)
                    yield {
                        "type": "text",
                        "text": text,
                    }
                    guard_error = output_guard.add(text)
                    if guard_error is not None:
                        unresolved_tool_failures["model_output"] = guard_error
                        log_event(
                            logger,
                            logging.WARNING,
                            "model.output.stopped",
                            status="failed",
                            error_code=guard_error,
                        )
                        yield {
                            "type": "error",
                            "code": guard_error,
                            "message": (
                                "模型输出出现重复，已停止本轮生成。"
                                if guard_error == "model_repetition_detected"
                                else "模型输出超过安全长度，已停止本轮生成。"
                            ),
                        }
                        close = getattr(event_stream, "aclose", None)
                        if callable(close):
                            await close()
                        break
            # 处理工具开始
            elif event_name == "on_tool_start":
                run_id = str(event.get("run_id", ""))
                mark_tool_started(tool_started_at, run_id)
                input_data = (
                    event_data.get("input", {})
                    if isinstance(event_data, dict)
                    else {}
                )

                if observability and tool_name == "recall_memories":
                    yield lifecycle.event(
                        "memory_recall_started",
                        phase="memory_recall",
                        agent_name=agent_name,
                        status="started",
                        source="tool",
                    )

                # 启动子 agent 任务
                if tool_name == "task":
                    task_id, info, start_event = build_task_start(
                        str(event.get("run_id", "")),
                        input_data,
                    )
                    active_subagents[task_id] = info

                    log_event(
                        logger,
                        logging.INFO,
                        "subagent.task.started",
                        task_id=task_id,
                        agent_name=info.get("agent_name", "unknown"),
                        status="started",
                    )

                    yield start_event
                    continue

                # 更新待办列表
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

                log_event(
                    logger,
                    logging.INFO,
                    "tool.started",
                    tool_name=tool_name,
                    agent_name=("researcher" if is_researcher_event else None),
                    status="started",
                )

                if is_researcher_event:
                    continue

                yield {
                    "type": "tool_start",
                    "name": tool_name,
                    "input": tool_input_preview(tool_name, event_data),
                }

            # 处理工具结束
            elif event_name == "on_tool_end":
                run_id = str(event.get("run_id", ""))
                elapsed_ms = take_tool_elapsed(
                    tool_started_at,
                    run_id,
                )
                if observability and tool_name == "recall_memories":
                    output = event_data.get("output") if isinstance(event_data, dict) else None
                    count = len(output) if isinstance(output, list) else 0
                    yield lifecycle.event(
                        "memory_recall_completed",
                        phase="memory_recall",
                        agent_name=agent_name,
                        status="completed",
                        source="tool",
                        recalled_count=count,
                    )
                if tool_name == "task":
                    task_id = str(event.get("run_id", ""))
                    info = active_subagents.pop(task_id, {})
                    (
                        terminal_record,
                        terminal_event,
                        error,
                        finding_ids,
                        source_ids,
                    ) = build_task_terminal(
                        task_id,
                        info,
                        event_data,
                    )
                    terminal_tasks[task_id] = terminal_record

                    if error:

                        log_event(
                            logger,
                            logging.ERROR,
                            "subagent.task.failed",
                            task_id=task_id,
                            agent_name=info.get(
                                "agent_name",
                                "unknown",
                            ),
                            status="failed",
                            error_code=error,
                        )

                        log_event(
                            logger,
                            logging.ERROR,
                            "tool.failed",
                            tool_name=tool_name,
                            task_id=task_id,
                            status="failed",
                            error_code=error,
                            elapsed_ms=elapsed_ms,
                        )

                        yield terminal_event
                        continue
                    log_event(
                        logger,
                        logging.INFO,
                        "tool.completed",
                        tool_name=tool_name,
                        task_id=task_id,
                        agent_name=info.get("agent_name"),
                        status="completed",
                        elapsed_ms=elapsed_ms,
                    )

                    log_event(
                        logger,
                        logging.INFO,
                        "subagent.task.completed",
                        task_id=task_id,
                        agent_name=info.get("agent_name", "unknown"),
                        status="completed",
                        elapsed_ms=elapsed_ms,
                    )

                    yield terminal_event
                    continue

                tool_status, error_code, retryable = tool_result_outcome(
                    event_data
                )
                if tool_status == "failed":
                    unresolved_tool_failures[tool_name] = (
                        error_code or "tool_business_failed"
                    )
                    log_event(
                        logger,
                        logging.WARNING,
                        "tool.failed",
                        tool_name=tool_name,
                        agent_name=(
                            "researcher" if is_researcher_event else None
                        ),
                        status="failed",
                        error_code=error_code,
                        elapsed_ms=elapsed_ms,
                    )
                else:
                    unresolved_tool_failures.pop(tool_name, None)
                    log_event(
                        logger,
                        logging.INFO,
                        "tool.completed",
                        tool_name=tool_name,
                        agent_name=(
                            "researcher" if is_researcher_event else None
                        ),
                        status="completed",
                        elapsed_ms=elapsed_ms,
                    )
                if is_researcher_event:
                    continue

                yield {
                    "type": "tool_end",
                    "name": tool_name,
                    "status": tool_status,
                    "elapsed_ms": elapsed_ms,
                    **(
                        {"error_code": error_code}
                        if error_code is not None
                        else {}
                    ),
                    **(
                        {"retryable": retryable}
                        if retryable is not None
                        else {}
                    ),
                }

                if tool_status == "failed":
                    failure_key = _repair_key(tool_name, error_code)
                    if error_code in _PUBLICATION_REPAIRABLE_ERRORS:
                        attempts = repair_attempts.get(failure_key, 0) + 1
                        repair_attempts[failure_key] = attempts
                        if attempts <= 1:
                            # The structured tool result is now in the model
                            # conversation. Give Main Agent one chance to
                            # read, revise, and resubmit the Article.
                            continue
                        stop_code = "publication_repair_exhausted"
                        stop_message = (
                            "文章已自动修复并重新校验，但仍未通过发布平台限制；"
                            "本轮已停止，请修改文章后重试。"
                        )
                    elif retryable is True:
                        attempts = retry_attempts.get(failure_key, 0) + 1
                        retry_attempts[failure_key] = attempts
                        if attempts <= 1:
                            continue
                        stop_code = "tool_retry_exhausted"
                        stop_message = (
                            "工具连续重试仍未成功，本轮已停止，请稍后重试。"
                        )
                    else:
                        # A non-retryable error without an explicit repair
                        # contract must terminate immediately. Letting the
                        # model guess parameters caused the old recursion loop.
                        stop_code = error_code or "tool_business_failed"
                        stop_message = (
                            "工具返回不可自动修复的错误，本轮已停止："
                            f"{stop_code}"
                        )

                    unresolved_tool_failures["tool_loop"] = stop_code
                    log_event(
                        logger,
                        logging.WARNING,
                        "agent.tool_loop.stopped",
                        status="failed",
                        error_code=stop_code,
                    )
                    yield {
                        "type": "error",
                        "code": stop_code,
                        "message": stop_message,
                        "tool_name": tool_name,
                        **(
                            {"original_error_code": error_code}
                            if error_code and error_code != stop_code
                            else {}
                        ),
                    }
                    close = getattr(event_stream, "aclose", None)
                    if callable(close):
                        await close()
                    break

                if (
                    tool_status == "completed"
                    and tool_name == "revise_article_for_publication"
                ):
                    # A successful revision starts a fresh repair attempt for
                    # the next validation result.
                    repair_attempts.clear()

                if tool_name == "save_report":
                    artifact = artifact_preview(event_data)

                    if artifact:
                        yield {
                            "type": "artifact_saved",
                            **artifact,
                        }

                if tool_name == "generate_image":
                    generated = generated_image_preview(event_data)
                    if generated:
                        yield {
                            "type": "image_generated",
                            **generated,
                        }

                if tool_name == "resolve_publication_intent":
                    intent = publishing_intent_preview(event_data)
                    if intent:
                        defer_interaction({
                            "type": "publishing_intent",
                            **intent,
                        })

                if tool_name == "request_publication_approval":
                    selection = publication_attachment_selection_preview(event_data)
                    if selection:
                        defer_interaction({
                            "type": "publication_attachment_selection",
                            **selection,
                        })

                if tool_name == "get_publication_approval_status":
                    recovery = publishing_approval_status_preview(event_data)
                    for card in recovery.get("cards", []):
                        defer_interaction({
                            "type": "publishing_intent",
                            **card,
                        })

            # 处理工具错误
            elif event_name == "on_tool_error":

                run_id = str(event.get("run_id", ""))
                elapsed_ms = take_tool_elapsed(
                    tool_started_at,
                    run_id,
                )

                log_event(
                    logger,
                    logging.ERROR,
                    "tool.failed",
                    tool_name=tool_name,
                    agent_name=("researcher" if is_researcher_event else None),
                    task_id=(
                        run_id
                        if tool_name == "task"
                        else None
                    ),
                    status="failed",
                    error_code="tool_execution_failed",
                    elapsed_ms=elapsed_ms,
                )

                if tool_name == "task":
                    task_id = str(event.get("run_id", ""))
                    info = active_subagents.pop(task_id, {})
                    terminal_record, terminal_event = build_task_failure(
                        task_id,
                        info,
                        "task_execution_failed",
                    )
                    terminal_tasks[task_id] = terminal_record
                    log_event(
                        logger,
                        logging.ERROR,
                        "subagent.task.failed",
                        task_id=task_id,
                        agent_name=info.get(
                            "agent_name",
                            "unknown",
                        ),
                        status="failed",
                        error_code="task_execution_failed",
                    )
                    yield terminal_event
                    continue

                if is_researcher_event:
                    continue

                unresolved_tool_failures[tool_name] = (
                    "tool_execution_failed"
                )

                yield {
                    "type": "tool_end",
                    "name": tool_name,
                    "status": "failed",
                    "elapsed_ms": elapsed_ms,
                }

        try:
            snapshot = await agent.aget_state(agent_config(thread_id))
            for interrupt_record in getattr(snapshot, "interrupts", ()):
                interrupt_id = getattr(interrupt_record, "id", None)
                if interrupt_id and str(interrupt_id) in seen_interrupt_ids:
                    continue
                interrupt_detected = True
                interrupt_preview = publishing_interrupt_preview(
                    getattr(interrupt_record, "value", None),
                )
                if interrupt_preview:
                    defer_interaction({
                        "type": "publishing_intent",
                        **interrupt_preview,
                    })
        except Exception:
            log_event(
                logger,
                logging.WARNING,
                "hitl.interrupt_preview_failed",
                thread_id=thread_id,
                status="failed",
                error_code="hitl_interrupt_preview_failed",
            )

        final_events = await finalize_stream(
            agent=agent,
            thread_id=thread_id,
            answer_chunks=answer_chunks,
            terminal_tasks=terminal_tasks,
            unresolved_tool_failures=unresolved_tool_failures,
            waiting_for_user=interrupt_detected,
            logger=logger,
        )
        terminal_tasks_persisted = True
        # Keep `done` terminal for SSE consumers.  The assistant text and
        # final suffix are sent first, then the deferred HITL card, and only
        # then the turn is marked complete/waiting.
        done_events: list[dict[str, object]] = []
        for final_event in final_events:
            if final_event.get("type") == "done":
                done_events.append(final_event)
                continue
            if observability and final_event.get("type") == "done":
                final_event = lifecycle.done(
                    goal_status=str(final_event.get("goal_status") or "completed"),
                    error_codes=[
                        str(code)
                        for code in final_event.get("error_codes", [])
                        if isinstance(code, (str, int, float))
                    ],
                )
            yield final_event
        for interaction_event in deferred_interaction_events:
            yield interaction_event
        for final_event in done_events:
            if observability:
                final_event = lifecycle.done(
                    goal_status=str(final_event.get("goal_status") or "completed"),
                    error_codes=[
                        str(code)
                        for code in final_event.get("error_codes", [])
                        if isinstance(code, (str, int, float))
                    ],
                )
            yield final_event

    except asyncio.CancelledError:
        await cancelled_task_records(
            agent=agent,
            thread_id=thread_id,
            active_subagents=active_subagents,
            terminal_tasks=terminal_tasks,
            tool_started_at=tool_started_at,
            logger=logger,
        )
        terminal_tasks_persisted = True

        raise
    finally:
        if not terminal_tasks_persisted:
            try:
                await cancelled_task_records(
                    agent=agent,
                    thread_id=thread_id,
                    active_subagents=active_subagents,
                    terminal_tasks=terminal_tasks,
                    tool_started_at=tool_started_at,
                    logger=logger,
                )
            except (asyncio.CancelledError, Exception):
                log_event(
                    logger,
                    logging.ERROR,
                    "subagent.tasks.cancel_persistence_failed",
                    status="failed",
                    error_code="task_persistence_failed",
                )
        close = getattr(event_stream, "aclose", None)
        if callable(close):
            try:
                await close()
            except (asyncio.CancelledError, Exception):
                pass
