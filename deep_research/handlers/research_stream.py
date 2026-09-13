import json
import logging
import time
import asyncio

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from uuid import uuid4

from ..activity import (
    activity_from_stream_event,
    build_activity_event,
    build_question_activity,
)
from ..log.logging_utils import log_event
from ..state.runtime import agent_config
from ..agent.observability import LifecycleTracker, safe_model_error_code
from ..config import settings
from ..log.log_context import get_request_id
from .research_memory import review_successful_research

# 把一个事件字典序列化成特定格式的字符串
def encode_event(event: dict[str, object]) -> str:
    return (
        json.dumps(
            event,
            ensure_ascii=False,
        )
        + "\n"
    )


def encode_activity_event(event: dict[str, object]) -> str:
    return encode_event({
        "type": "activity",
        "activity": event,
    })


async def _stream_with_heartbeat(
    source: AsyncIterator[dict[str, object]],
    tracker: LifecycleTracker,
) -> AsyncIterator[dict[str, object]]:
    """Multiplex the existing stream with a cancellable heartbeat task."""
    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
    active = True

    async def pump() -> None:
        try:
            async for item in source:
                await queue.put(("source", item))
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            await queue.put(("error", error))
        finally:
            await queue.put(("eof", None))

    async def heartbeat() -> None:
        try:
            while active:
                await asyncio.sleep(settings.heartbeat_interval_seconds)
                if active:
                    await queue.put(("heartbeat", tracker.heartbeat()))
        except asyncio.CancelledError:
            raise

    source_task = asyncio.create_task(pump())
    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "heartbeat":
                yield payload  # type: ignore[misc]
            elif kind == "source":
                event = payload  # type: ignore[assignment]
                if isinstance(event, dict) and event.get("type") == "done":
                    event = {
                        **event,
                        "heartbeat_count": tracker.heartbeat_count,
                    }
                yield event  # type: ignore[misc]
            elif kind == "error":
                raise payload  # type: ignore[misc]
            else:
                break
    finally:
        active = False
        heartbeat_task.cancel()
        source_task.cancel()
        await asyncio.gather(heartbeat_task, source_task, return_exceptions=True)
        close = getattr(source, "aclose", None)
        if callable(close):
            try:
                await close()
            except BaseException:
                pass


async def _persist_activity(
    agent: Any,
    thread_id: str,
    events: list[dict[str, object]],
    *,
    logger: logging.Logger,
) -> None:
    if not events or not hasattr(agent, "aupdate_state"):
        return

    try:
        await agent.aupdate_state(
            agent_config(thread_id),
            {"activity": events},
        )
    except Exception:
        logger.warning("failed to persist user activity")


async def _record_cancelled_run(
    *,
    agent: Any,
    thread_id: str,
    run_id: str,
    activity_events: list[dict[str, object]],
    logger: logging.Logger,
    started_at: float,
) -> None:
    if not any(
        event.get("run_id") == run_id
        and event.get("status") == "cancelled"
        for event in activity_events
    ):
        activity_events.append(
            build_activity_event(
                run_id=run_id,
                kind="research",
                actor="main",
                status="cancelled",
                label="研究已取消",
            )
        )

    log_event(
        logger,
        logging.WARNING,
        "research.cancelled",
        thread_id=thread_id,
        status="cancelled",
        elapsed_ms=round(
            (time.perf_counter() - started_at) * 1000,
            3,
        ),
    )
    await _persist_activity(
        agent,
        thread_id,
        activity_events,
        logger=logger,
    )
"""
stream_answer 是整个研究系统的外层流调度入口：
它包装了事件流生成器，在转发事件给前端的同时收集完整答案，
在研究结束时触发记忆审查回调，并统一处理取消和错误情况，
确保每一次研究交互都有完整的日志追踪和记忆更新。
"""
async def stream_answer(
    question: str,
    thread_id: str,
    *,
    memory_service: Any = None,
    memory_extractor: Any = None,
    agent: Any,
    stream_events: Callable[..., AsyncIterator[dict[str, object]]],
    review_callback: Callable[..., Awaitable[object]], # 新的 review_successful_research helper
    review_runner: Callable[..., Awaitable[object]], # 旧的 review_after_success 实际执行函数
    has_explicit_correction: Callable[[str], bool],
    has_confirmed_project_decision: Callable[[str], bool],
    has_memory_rule_signal: Callable[[str], bool],
    logger: logging.Logger,
    attachment_ids: tuple[str, ...] = (),
    attachment_selection_confirmed: bool = False,
) -> AsyncIterator[str]:
    run_id = uuid4().hex
    activity_events: list[dict[str, object]] = [
        build_question_activity(
            question,
            run_id=run_id,
        ),
        build_activity_event(
            run_id=run_id,
            kind="research",
            actor="main",
            status="running",
            label="开始处理问题",
        ),
    ]

    # 初始化与开始日志
    started_at = time.perf_counter()
    terminal_recorded = False
    try:
        completion_logged = False
        answer_parts: list[str] = []
        report_saved = False
        model_completed_sent = False
        first_token_sent = False
        waiting_for_user = False

        log_event(
            logger,
            logging.INFO,
            "research.started",
            thread_id=thread_id,
            status="started",
        )

        for activity in activity_events:
            yield encode_activity_event(activity)
        # 消费事件流
        lifecycle = LifecycleTracker(thread_id=thread_id, run_id=run_id)
        source_events = (
            stream_events(
                question,
                thread_id,
                attachment_ids,
                attachment_selection_confirmed,
            )
            if attachment_ids or attachment_selection_confirmed
            else stream_events(question, thread_id)
        )
        event_source = (
            _stream_with_heartbeat(source_events, lifecycle)
            if get_request_id() is not None
            else source_events
        )
        async for event in event_source:
            observability_enabled = get_request_id() is not None
            if event.get("type") == "model_reasoning":
                event = {
                    **event,
                    "message": (
                        "Researcher 正在整理来源"
                        if event.get("agent_name") == "researcher"
                        else "主 Agent 正在分析请求"
                    ),
                }
            if event.get("type") == "model_completed":
                model_completed_sent = True
            if event.get("type") == "model_started":
                lifecycle.model_attempts += 1
            if event.get("type") == "first_token":
                first_token_sent = True
                lifecycle.first_token_received = True
            if observability_enabled and event.get("type") == "text":
                text_value = event.get("text")
                if (not first_token_sent and isinstance(text_value, str)
                        and text_value.strip()):
                    first_token = lifecycle.first_token()
                    if first_token is not None:
                        if lifecycle.model_attempts == 0:
                            lifecycle.model_attempts = 1
                        first_token_sent = True
                        yield encode_event(first_token)

            if observability_enabled and event.get("type") == "error":
                yield encode_event(
                    lifecycle.event(
                        "model_failed",
                        agent_name="main",
                        model_role="main",
                        phase="model",
                        status="failed",
                        error_code=safe_model_error_code(event.get("code")),
                        retryable=False,
                    )
                )

            if observability_enabled and event.get("type") == "done":
                if not model_completed_sent:
                    yield encode_event(
                        lifecycle.event(
                            "model_completed",
                            agent_name="main",
                            model_role="main",
                            phase="model",
                            status="completed",
                            output_type="text" if lifecycle.first_token_received else "empty",
                        )
                    )
                event = {
                    **event,
                    "elapsed_ms": event.get("elapsed_ms", lifecycle.event("done")["elapsed_ms"]),
                    "first_token_received": event.get(
                        "first_token_received", lifecycle.first_token_received
                    ),
                    "model_attempts": event.get("model_attempts", lifecycle.model_attempts),
                    "heartbeat_count": event.get("heartbeat_count", lifecycle.heartbeat_count),
                }
            activity = activity_from_stream_event(
                event,
                run_id=run_id,
            )

            if activity is not None:
                activity_events.append(activity)
                yield encode_activity_event(activity)

            if event.get("type") == "artifact_saved":
                report_saved = True

            if event.get("type") == "text":
                text = event.get("text", "")

                if isinstance(text, str):
                    answer_parts.append(text)

            if event.get("type") == "done":
                terminal_recorded = True
                answer = "".join(answer_parts)
                goal_status = str(
                    event.get("goal_status") or "completed"
                )
                waiting_for_user = goal_status == "waiting_for_user"

                if observability_enabled and goal_status == "waiting_for_user":
                    yield encode_event(
                        lifecycle.event(
                            "waiting_for_user",
                            phase="hitl",
                            agent_name="main",
                            status="waiting",
                            error_code="hitl_waiting",
                        )
                    )

                if goal_status == "completed" and answer.strip():
                    await review_callback(
                        agent=agent,
                        thread_id=thread_id,
                        question=question,
                        answer=answer,
                        memory_service=memory_service,
                        memory_extractor=memory_extractor,
                        report_saved=report_saved,
                        review_after_success=review_runner,
                        has_explicit_correction=has_explicit_correction,
                        has_confirmed_project_decision=(
                            has_confirmed_project_decision
                        ),
                        has_memory_rule_signal=has_memory_rule_signal,
                    )

                if not completion_logged:
                    log_event(
                        logger,
                        logging.INFO,
                        "research.completed",
                        thread_id=thread_id,
                        status=goal_status,
                        elapsed_ms=round(
                            (
                                time.perf_counter()
                                - started_at
                            )
                            * 1000,
                            3,
                        ),
                    )
                    completion_logged = True

            yield encode_event(event)

        if not waiting_for_user:
            await _persist_activity(
                agent,
                thread_id,
                activity_events,
                logger=logger,
            )
        else:
            # Do not call aupdate_state after a native LangGraph interrupt.
            # Updating an interrupted checkpoint creates a new checkpoint and
            # clears the task-level interrupt, leaving the pending tool task
            # impossible to resume from the HITL endpoint.
            logger.info(
                "defer activity persistence while native HITL is waiting"
            )

    except asyncio.CancelledError:
        terminal_recorded = True
        await _record_cancelled_run(
            agent=agent,
            thread_id=thread_id,
            run_id=run_id,
            activity_events=activity_events,
            logger=logger,
            started_at=started_at,
        )
        raise

    except Exception as error:
        terminal_recorded = True
        activity_events.append(
            build_activity_event(
                run_id=run_id,
                kind="research",
                actor="main",
                status="failed",
                label="研究失败",
                detail="research_stream_failed",
            )
        )
        await _persist_activity(
            agent,
            thread_id,
            activity_events,
            logger=logger,
        )

        log_event(
            logger,
            logging.ERROR,
            "research.failed",
            thread_id=thread_id,
            status="failed",
            error_code="research_stream_failed",
            exception_type=type(error).__name__,
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
            exc_info=True,
        )
        raise
    finally:
        if not terminal_recorded:
            try:
                await _record_cancelled_run(
                    agent=agent,
                    thread_id=thread_id,
                    run_id=run_id,
                    activity_events=activity_events,
                    logger=logger,
                    started_at=started_at,
                )
            except Exception:
                log_event(
                    logger,
                    logging.ERROR,
                    "research.cancel_persistence_failed",
                    thread_id=thread_id,
                    status="failed",
                    error_code="cancel_persistence_failed",
                )
