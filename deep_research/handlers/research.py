import logging
import asyncio
import time
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..agent import run_research, stream_research_events
from ..agent.hitl_resume import pending_native_hitl_interrupts
from .. import agent as agent_module
from ..memory.review_runner import review_after_success
from ..memory.triggers import has_explicit_correction,has_confirmed_project_decision,has_memory_rule_signal
from ..state.access import thread_values
from ..log.log_context import set_thread_id
from ..log.logging_utils import log_event
from .research_memory import (
    artifact_ids_for_thread,
    review_successful_research,
)
from .research_stream import stream_answer

logger = logging.getLogger(__name__)
router = APIRouter()


async def _reject_new_turn_while_interrupted(
    http_request: Request,
    thread_id: str,
) -> None:
    """Prevent an ordinary message from bypassing a pending HITL pause."""

    agent = getattr(http_request.app.state, "agent", agent_module.agent)
    interrupts = await pending_native_hitl_interrupts(
        agent,
        thread_id=thread_id,
    )
    if not interrupts:
        return

    first = interrupts[0]
    value = getattr(first, "value", None)
    if isinstance(first, dict):
        value = first.get("value")
    interaction_id = (
        value.get("interaction_id")
        if isinstance(value, dict)
        else None
    )
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "error_code": "hitl_pending",
            "message": "当前会话正在等待审批，请先处理会话中的确认卡片。",
            "interaction_id": interaction_id,
        },
    )


class ResearchRequest(BaseModel):
    question: str
    thread_id: UUID | None = None
    attachment_ids: list[str] = Field(default_factory=list, max_length=30)
    attachment_selection_confirmed: bool = False

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()

        if not question:
            raise ValueError("question must not be empty")

        return question

    @field_validator("attachment_ids")
    @classmethod
    def validate_attachment_ids(cls, value: list[str]) -> list[str]:
        normalized = []
        for attachment_id in value:
            item = attachment_id.strip()
            if not item:
                raise ValueError("attachment_ids must contain non-empty strings")
            normalized.append(item)
        return list(dict.fromkeys(normalized))


class ResearchResponse(BaseModel):
    answer: str
    thread_id: UUID | None = None


async def _artifact_ids_for_thread(
    thread_id: str,
) -> set[str] | None:
    return await artifact_ids_for_thread(
        agent_module.agent,
        thread_id,
        thread_values=thread_values,
        logger=logger,
    )

"""
run_research()
  -> 得到非空最终答案
  -> 调用 review_after_success()
  -> 记忆流程失败则记录日志
  -> 返回原答案
"""
@router.post("/research", response_model=ResearchResponse)
async def research(
    request: ResearchRequest,
    http_request: Request,
) -> ResearchResponse:

    thread_id = request.thread_id or uuid4()
    set_thread_id(str(thread_id))

    await _reject_new_turn_while_interrupted(
        http_request,
        str(thread_id),
    )

    started_at = time.perf_counter()

    log_event(
        logger,
        logging.INFO,
        "research.started",
        thread_id=str(thread_id),
        status="started",
    )

    try:
        memory_service = getattr(
            http_request.app.state,
            "memory_service",
            None,
        )

        memory_extractor = getattr(
            http_request.app.state,
            "memory_extractor",
            None,
        )

        before_artifact_ids = None

        if memory_service is not None and memory_extractor is not None:
            before_artifact_ids = await _artifact_ids_for_thread(
                str(thread_id)
            )

        answer = await run_research(
            request.question,
            str(thread_id),
            selected_attachment_ids=tuple(request.attachment_ids),
            attachment_selection_confirmed=request.attachment_selection_confirmed,
        )

        after_artifact_ids = None

        if before_artifact_ids is not None:
            after_artifact_ids = await _artifact_ids_for_thread(
                str(thread_id)
            )

        report_saved = (
            before_artifact_ids is not None
            and after_artifact_ids is not None
            and bool(after_artifact_ids - before_artifact_ids)
        )

        await review_successful_research(
            agent=agent_module.agent,
            thread_id=str(thread_id),
            question=request.question,
            answer=answer,
            memory_service=memory_service,
            memory_extractor=memory_extractor,
            report_saved=report_saved,
            review_after_success=review_after_success,
            has_explicit_correction=has_explicit_correction,
            has_confirmed_project_decision=has_confirmed_project_decision,
            has_memory_rule_signal=has_memory_rule_signal,
        )
        elapsed_ms = round(
            (
                time.perf_counter()
                - started_at
            )
            * 1000,
            3,
        )

        log_event(
            logger,
            logging.INFO,
            "research.completed",
            thread_id=str(thread_id),
            status="completed",
            elapsed_ms=elapsed_ms,
        )

        return ResearchResponse(
            answer=answer,
            thread_id=thread_id,
        )

    except asyncio.CancelledError:
        log_event(
            logger,
            logging.WARNING,
            "research.cancelled",
            thread_id=str(thread_id),
            status="cancelled",
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
        )
        raise

    except Exception as error:
        log_event(
            logger,
            logging.ERROR,
            "research.failed",
            thread_id=str(thread_id),
            status="failed",
            error_code="research_failed",
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
# 引用定义的stream_answer
async def _stream_answer(
    question: str,
    thread_id: str,
    *,
    memory_service=None,
    memory_extractor=None,
    report_saved=False,
    attachment_ids: tuple[str, ...] = (),
    attachment_selection_confirmed: bool = False,
) -> AsyncIterator[str]:
    async for chunk in stream_answer(
        question,
        thread_id,
        memory_service=memory_service,
        memory_extractor=memory_extractor,
        agent=agent_module.agent,
        stream_events=stream_research_events,
        attachment_ids=attachment_ids,
        attachment_selection_confirmed=attachment_selection_confirmed,
        review_callback=review_successful_research,
        review_runner=review_after_success,
        has_explicit_correction=has_explicit_correction,
        has_confirmed_project_decision=(
            has_confirmed_project_decision
        ),
        has_memory_rule_signal=has_memory_rule_signal,
        logger=logger,
    ):
        yield chunk

@router.post(
    "/research/stream",
    response_class=StreamingResponse,
)
async def research_stream(
    request: ResearchRequest,
    http_request: Request,
) -> StreamingResponse:
    thread_id = request.thread_id or uuid4()
    set_thread_id(str(thread_id))
    await _reject_new_turn_while_interrupted(
        http_request,
        str(thread_id),
    )
    memory_service = getattr(
        http_request.app.state,
        "memory_service",
        None,
    )
    memory_extractor = getattr(
        http_request.app.state,
        "memory_extractor",
        None,
    )

    return StreamingResponse(
        _stream_answer(
            request.question,
            str(thread_id),
            memory_service=memory_service,
            memory_extractor=memory_extractor,
            attachment_ids=tuple(request.attachment_ids),
            attachment_selection_confirmed=request.attachment_selection_confirmed,
        ),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "X-Thread-ID": str(thread_id),
        },
    )
