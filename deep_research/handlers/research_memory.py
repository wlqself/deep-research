import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..log.logging_utils import log_event

# 从某个 agent 的指定 thread 中读取所有 artifact（产物）的 ID
async def artifact_ids_for_thread(
    agent: Any,
    thread_id: str,
    *,
    thread_values: Callable[
        [Any, str],
        Awaitable[dict[str, object]],
    ],
    logger: logging.Logger,
) -> set[str] | None:
    try:
        values = await thread_values(agent, thread_id) # 异步调用回调函数，拿到该线程的所有状态数据（一个字典）；
    except Exception as error:
        log_event(
            logger,
            logging.ERROR,
            "research.artifacts.read.failed",
            status="failed",
            error_code="artifact_read_failed",
            exception_type=type(error).__name__,
            exc_info=True,
        )
        return None

    raw_artifacts = values.get("artifacts", {})

    if not isinstance(raw_artifacts, dict):
        return set()

    return {
        artifact_id
        for artifact_id in raw_artifacts
        if isinstance(artifact_id, str)
    }

# 在研究（research）成功完成后，触发一轮"后置审查"，把对话内容交给记忆系统去提取和存储。
async def review_successful_research(
    *,
    agent: Any,
    thread_id: str,
    question: str,
    answer: str,
    memory_service: Any,
    memory_extractor: Any,
    report_saved: bool,
    review_after_success: Callable[..., Awaitable[object]],
    has_explicit_correction: Callable[[str], bool],
    has_confirmed_project_decision: Callable[[str], bool],
    has_memory_rule_signal: Callable[[str], bool],
) -> None:
    # 前置守卫（早退检查）
    if (
        memory_service is None
        or memory_extractor is None
        or not answer.strip()
    ):
        return

    await review_after_success(
        agent,
        thread_id,
        answer=answer,
        memory_service=memory_service,
        extractor=memory_extractor,
        explicit_correction=has_explicit_correction(question),
        project_decision_confirmed=(
            has_confirmed_project_decision(question)
        ),
        report_saved=report_saved,
        rule_triggered=has_memory_rule_signal(question),
    )