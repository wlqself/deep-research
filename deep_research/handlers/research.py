import json
import logging
from collections.abc import AsyncIterator
from urllib import request
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from ..agent import run_research, stream_research_events
from .. import agent as agent_module
from ..memory.review_runner import review_after_success
from ..memory.triggers import has_explicit_correction,has_confirmed_project_decision,has_memory_rule_signal
from ..state.access import thread_values

logger = logging.getLogger(__name__)
router = APIRouter()


class ResearchRequest(BaseModel):
    question: str
    thread_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()

        if not question:
            raise ValueError("question must not be empty")

        return question


class ResearchResponse(BaseModel):
    answer: str
    thread_id: UUID | None = None


async def _artifact_ids_for_thread(
    thread_id: str,
) -> set[str] | None:
    try:
        values = await thread_values(
            agent_module.agent,
            thread_id,
        )
    except Exception:
        logger.exception(
            "failed to read artifacts for memory review trigger"
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

    if (
        memory_service is not None
        and memory_extractor is not None
        and answer.strip()
    ):
        await review_after_success(
            agent_module.agent,
            str(thread_id),
            answer=answer,
            memory_service=memory_service,
            extractor=memory_extractor,
            explicit_correction=has_explicit_correction(request.question),
            project_decision_confirmed=(
                has_confirmed_project_decision(request.question)
            ),
            report_saved=report_saved,
            rule_triggered=has_memory_rule_signal(request.question),
        )
    return ResearchResponse(
        answer=answer,
        thread_id=thread_id,
    )


async def _stream_answer(
    question: str,
    thread_id: str,
    *,
    memory_service=None,
    memory_extractor=None,
    report_saved = False
) -> AsyncIterator[str]:
    answer_parts: list[str] = []
    report_saved = False

    async for event in stream_research_events(
        question,
        thread_id,
    ):
        if event.get("type") == "artifact_saved":
            report_saved = True

        if event.get("type") == "text":
            text = event.get("text", "")

            if isinstance(text, str):
                answer_parts.append(text)

        if event.get("type") == "done":
            answer = "".join(answer_parts)

            if (
                memory_service is not None
                and memory_extractor is not None
                and answer.strip()
            ):
                await review_after_success(
                    agent_module.agent,
                    thread_id,
                    answer=answer,
                    memory_service=memory_service,
                    extractor=memory_extractor,
                    report_saved=report_saved,
                    explicit_correction=has_explicit_correction(question),
                    project_decision_confirmed=(
                        has_confirmed_project_decision(question)
                    ),
                    rule_triggered=has_memory_rule_signal(question),
                )

        yield (
            json.dumps(
                event,
                ensure_ascii=False,
            )
            + "\n"
        )


@router.post(
    "/research/stream",
    response_class=StreamingResponse,
)
async def research_stream(
    request: ResearchRequest,
    http_request: Request,
) -> StreamingResponse:
    thread_id = request.thread_id or uuid4()
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
        ),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "X-Thread-ID": str(thread_id),
        },
    )
