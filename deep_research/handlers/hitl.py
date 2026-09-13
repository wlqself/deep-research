"""Private HTTP boundary for durable human-in-the-loop decisions."""

from __future__ import annotations

import logging
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from typing import Literal

from ..agent.hitl_resume import (
    checkpoint_id,
    interrupt_id,
    native_hitl_snapshot,
    resume_agent_after_hitl,
)
from ..hitl.models import (
    HITLInteraction,
    HITLStatus,
    InvalidHITLTransitionError,
    InvalidHITLValueError,
)
from ..hitl.decision_service import HITLDecisionError
from ..hitl.publication_state import resolve_publication_hitl_state
from ..hitl.service import HITLService, HITLServiceError
from ..log.logging_utils import log_event


router = APIRouter()
logger = logging.getLogger("deep_research.hitl.api")


class DecideHITLRequest(BaseModel):
    decision_reason: str | None = Field(default=None, max_length=500)


class RejectHITLRequest(BaseModel):
    decision_reason: str | None = Field(default=None, max_length=500)


class UnifiedHITLDecisionRequest(BaseModel):
    decision: Literal["approve", "reject", "request_changes"]
    decision_reason: str | None = Field(default=None, max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=128)


class HITLInteractionResponse(BaseModel):
    interaction_id: str
    thread_id: str
    run_id: str
    action: str
    target_type: str
    target_id: str
    target_version: int
    status: str
    created_at: str
    updated_at: str
    resolved_at: str | None
    decision_actor: str | None
    decision_reason: str | None
    error_code: str | None
    interrupt_id: str | None = None
    checkpoint_id: str | None = None
    expires_at: str | None = None
    publication_approval_status: str | None = None
    target: dict[str, Any] | None = None
    requires_user_confirmation: bool = True


class HITLResumeResponse(BaseModel):
    interaction_id: str
    resumed: bool
    answer: str | None = None
    error_code: str | None = None
    next_intent: dict[str, Any] | None = None
    interrupt_id: str | None = None
    checkpoint_id: str | None = None


class HITLDecisionResponse(BaseModel):
    interaction: HITLInteractionResponse
    resume: HITLResumeResponse
    replayed: bool = False


def _timestamp(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _response(
    interaction: HITLInteraction,
    *,
    publication_approval_status: str | None = None,
    target: dict[str, Any] | None = None,
    requires_user_confirmation: bool | None = None,
) -> HITLInteractionResponse:
    if requires_user_confirmation is None:
        requires_user_confirmation = interaction.status in {
            HITLStatus.PENDING,
            HITLStatus.APPROVED,
            HITLStatus.FAILED,
        }
    return HITLInteractionResponse(
        interaction_id=interaction.interaction_id,
        thread_id=interaction.thread_id,
        run_id=interaction.run_id,
        action=interaction.action.value,
        target_type=interaction.target_type,
        target_id=interaction.target_id,
        target_version=interaction.target_version,
        status=interaction.status.value,
        created_at=_timestamp(interaction.created_at),
        updated_at=_timestamp(interaction.updated_at),
        resolved_at=_timestamp(interaction.resolved_at),
        decision_actor=interaction.decision_actor,
        decision_reason=interaction.decision_reason,
        error_code=interaction.error_code,
        interrupt_id=interaction.interrupt_id,
        checkpoint_id=interaction.checkpoint_id,
        expires_at=_timestamp(interaction.expires_at),
        publication_approval_status=publication_approval_status,
        target=target,
        requires_user_confirmation=requires_user_confirmation,
    )


def _publication_target(request: Request, interaction: HITLInteraction) -> dict[str, Any] | None:
    if interaction.target_type != "publication_approval":
        return None
    publishing_service = getattr(request.app.state, "publishing_service", None)
    if publishing_service is None:
        return None
    try:
        approval = publishing_service.repository.get_approval_request(interaction.target_id)
        article = publishing_service.get_article(approval.article_id)
        publication_state = resolve_publication_hitl_state(
            approval,
            getattr(request.app.state, "hitl_service", None),
        )
        return {
            "approval_id": approval.approval_id,
            "article_id": approval.article_id,
            "article_title": article.title,
            "article_slug": article.slug,
            "article_version": approval.article_version,
            "channel": approval.channel.value,
            "attachment_ids": list(approval.attachment_ids),
            "approval_status": publication_state.approval_status,
            "workflow_status": publication_state.workflow_status,
            "interaction_id": publication_state.interaction_id,
            "interaction_status": publication_state.interaction_status,
            "state_source": publication_state.state_source,
            "requires_user_confirmation": (
                publication_state.requires_user_confirmation
            ),
        }
    except Exception:
        return None


def _interaction_response(request: Request, interaction: HITLInteraction) -> HITLInteractionResponse:
    target = _publication_target(request, interaction)
    return _response(
        interaction,
        publication_approval_status=(target or {}).get("approval_status"),
        target=target,
        requires_user_confirmation=(target or {}).get(
            "requires_user_confirmation"
        ),
    )


def _service(request: Request) -> HITLService:
    service = getattr(request.app.state, "hitl_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "hitl_service_unavailable"},
        )
    return service


def _thread_mismatch() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error_code": "hitl_interaction_not_found"},
    )


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, HITLDecisionError):
        code = error.error_code
        http_status = {
            "hitl_interaction_not_found": status.HTTP_404_NOT_FOUND,
            "hitl_publication_target_not_found": status.HTTP_404_NOT_FOUND,
            "hitl_publication_target_stale": status.HTTP_409_CONFLICT,
            "hitl_decision_conflict": status.HTTP_409_CONFLICT,
            "hitl_publication_decision_conflict": status.HTTP_409_CONFLICT,
            "hitl_decision_target_invalid": status.HTTP_409_CONFLICT,
            "hitl_decision_record_failed": status.HTTP_503_SERVICE_UNAVAILABLE,
            "hitl_graph_not_interrupted": status.HTTP_200_OK,
            "hitl_graph_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
            "hitl_interaction_not_resolved": status.HTTP_409_CONFLICT,
            "hitl_resume_in_progress": status.HTTP_409_CONFLICT,
            "hitl_interaction_stale": status.HTTP_409_CONFLICT,
            "hitl_interaction_expired": status.HTTP_409_CONFLICT,
            "hitl_interrupt_mismatch": status.HTTP_409_CONFLICT,
            "hitl_idempotency_conflict": status.HTTP_409_CONFLICT,
            "hitl_decision_in_progress": status.HTTP_409_CONFLICT,
        }.get(code, status.HTTP_400_BAD_REQUEST)
    elif isinstance(error, HITLServiceError):
        code = (
            "hitl_interaction_not_found"
            if "not found" in str(error).lower()
            else "hitl_operation_failed"
        )
        http_status = (
            status.HTTP_404_NOT_FOUND
            if code == "hitl_interaction_not_found"
            else status.HTTP_400_BAD_REQUEST
        )
    elif isinstance(error, InvalidHITLTransitionError):
        code, http_status = (
            "hitl_invalid_state_transition",
            status.HTTP_409_CONFLICT,
        )
    elif isinstance(error, InvalidHITLValueError):
        code, http_status = "hitl_invalid_request", status.HTTP_422_UNPROCESSABLE_ENTITY
    else:
        code, http_status = "hitl_operation_failed", status.HTTP_500_INTERNAL_SERVER_ERROR

    log_event(
        logger,
        logging.WARNING if http_status < 500 else logging.ERROR,
        "hitl.api.failed",
        status="failed",
        error_code=code,
    )
    return HTTPException(
        status_code=http_status,
        detail={"error_code": code},
    )


def _get_thread_interaction(
    service: HITLService,
    interaction_id: str,
    thread_id: str,
) -> HITLInteraction:
    interaction = service.get_interaction(interaction_id)
    if interaction.thread_id != thread_id:
        raise _thread_mismatch()
    return interaction


def _ensure_not_expired(
    service: HITLService,
    interaction: HITLInteraction,
) -> HITLInteraction:
    if interaction.is_expired and interaction.status is HITLStatus.PENDING:
        service.expire(interaction.interaction_id)
        raise HITLDecisionError(
            "hitl_interaction_expired",
            "HITL interaction has expired",
        )
    if interaction.status is HITLStatus.EXPIRED:
        raise HITLDecisionError(
            "hitl_interaction_expired",
            "HITL interaction has expired",
        )
    return interaction


async def _reconcile_thread_interactions(
    request: Request,
    thread_id: str,
) -> None:
    """Bind durable interactions to, or invalidate them against, the graph."""

    agent = getattr(request.app.state, "agent", None)
    if agent is None or not hasattr(agent, "aget_state"):
        return

    service = _service(request)
    interactions = service.list_recoverable_for_thread(thread_id)
    if not interactions:
        return

    try:
        snapshot, interrupts = await native_hitl_snapshot(
            agent,
            thread_id=thread_id,
        )
    except Exception:
        logger.warning(
            "hitl reconciliation could not read graph state",
            exc_info=True,
        )
        return

    records_by_interaction: dict[str, Any] = {}
    for record in interrupts:
        value = getattr(record, "value", None)
        if isinstance(record, dict):
            value = record.get("value")
        if (
            isinstance(value, dict)
            and isinstance(value.get("interaction_id"), str)
        ):
            records_by_interaction[value["interaction_id"]] = record

    current_checkpoint_id = checkpoint_id(snapshot)
    for interaction in interactions:
        record = records_by_interaction.get(interaction.interaction_id)
        if record is not None:
            current_interrupt_id = interrupt_id(record)
            if current_interrupt_id and (
                interaction.interrupt_id != current_interrupt_id
                or interaction.checkpoint_id != current_checkpoint_id
            ):
                service.bind_runtime(
                    interaction.interaction_id,
                    interrupt_id=current_interrupt_id,
                    checkpoint_id=current_checkpoint_id,
                )
        elif interaction.status in {
            HITLStatus.APPROVED,
            HITLStatus.REJECTED,
            HITLStatus.FAILED,
        }:
            # PENDING can be observed in the small window before LangGraph
            # persists its first interrupt checkpoint, so leave it untouched.
            service.mark_stale(interaction.interaction_id)


@router.get(
    "/hitl/interactions",
    response_model=list[HITLInteractionResponse],
)
async def list_interactions(
    request: Request,
    thread_id: str = Query(min_length=1, max_length=128),
) -> list[HITLInteractionResponse]:
    try:
        await _reconcile_thread_interactions(request, thread_id)
        interactions = _service(request).list_recoverable_for_thread(thread_id)
        return [_interaction_response(request, interaction) for interaction in interactions]
    except HTTPException:
        raise
    except Exception as error:
        raise _http_error(error) from error


@router.get(
    "/hitl/interactions/{interaction_id}",
    response_model=HITLInteractionResponse,
)
async def get_interaction(
    request: Request,
    interaction_id: str,
    thread_id: str = Query(min_length=1, max_length=128),
) -> HITLInteractionResponse:
    try:
        await _reconcile_thread_interactions(request, thread_id)
        return _interaction_response(
            request,
            _get_thread_interaction(
                _service(request), interaction_id, thread_id
            ),
        )
    except HTTPException:
        raise
    except Exception as error:
        raise _http_error(error) from error


@router.get("/hitl/interactions/{interaction_id}/audit")
def list_interaction_audit(
    request: Request,
    interaction_id: str,
    thread_id: str = Query(min_length=1, max_length=128),
) -> list[dict[str, object]]:
    """Return the append-only decision/resume audit for one thread-scoped card."""

    service = _service(request)
    interaction = _get_thread_interaction(service, interaction_id, thread_id)
    return service.repository.list_audit(interaction.interaction_id)


@router.post(
    "/hitl/interactions/{interaction_id}/approve",
    response_model=HITLInteractionResponse,
)
def approve_interaction(
    request: Request,
    interaction_id: str,
    payload: DecideHITLRequest | None = None,
    thread_id: str = Query(min_length=1, max_length=128),
) -> HITLInteractionResponse:
    try:
        service = _service(request)
        interaction = _get_thread_interaction(service, interaction_id, thread_id)
        _ensure_not_expired(service, interaction)
        decision_service = getattr(
            request.app.state,
            "hitl_decision_service",
            None,
        )
        if decision_service is not None and interaction.target_type == (
            "publication_approval"
        ):
            result = decision_service.approve_publication(
                interaction_id,
                thread_id=thread_id,
                decision_actor="user",
                decision_reason=(
                    payload.decision_reason if payload is not None else None
                ),
            )
            return _response(
                result.interaction,
                publication_approval_status=(
                    result.publication_approval.status.value
                ),
            )

        return _response(
            service.approve(
                interaction_id,
                decision_actor="user",
                decision_reason=(
                    payload.decision_reason if payload is not None else None
                ),
            )
        )
    except HTTPException:
        raise
    except Exception as error:
        raise _http_error(error) from error


@router.post(
    "/hitl/interactions/{interaction_id}/reject",
    response_model=HITLInteractionResponse,
)
def reject_interaction(
    request: Request,
    interaction_id: str,
    payload: RejectHITLRequest,
    thread_id: str = Query(min_length=1, max_length=128),
) -> HITLInteractionResponse:
    try:
        service = _service(request)
        interaction = _get_thread_interaction(service, interaction_id, thread_id)
        _ensure_not_expired(service, interaction)
        decision_service = getattr(
            request.app.state,
            "hitl_decision_service",
            None,
        )
        if decision_service is not None and interaction.target_type == (
            "publication_approval"
        ):
            result = decision_service.reject_publication(
                interaction_id,
                thread_id=thread_id,
                decision_actor="user",
                decision_reason=payload.decision_reason,
            )
            return _response(
                result.interaction,
                publication_approval_status=(
                    result.publication_approval.status.value
                ),
            )

        return _response(
            service.reject(
                interaction_id,
                decision_actor="user",
                decision_reason=payload.decision_reason,
            )
        )
    except HTTPException:
        raise
    except Exception as error:
        raise _http_error(error) from error


@router.post(
    "/hitl/interactions/{interaction_id}/decide",
    response_model=HITLDecisionResponse,
)
async def decide_interaction(
    request: Request,
    interaction_id: str,
    payload: UnifiedHITLDecisionRequest,
    thread_id: str = Query(min_length=1, max_length=128),
) -> HITLDecisionResponse:
    """Apply one decision and resume the exact native interrupt.

    The old approve/reject/resume endpoints remain available for clients that
    have not migrated. New clients use this single idempotent boundary so a
    browser retry cannot create a second decision or resume the graph twice.
    """

    service = _service(request)
    interaction = _get_thread_interaction(service, interaction_id, thread_id)
    _ensure_not_expired(service, interaction)
    repository = service.repository
    existing = repository.get_idempotency(interaction_id, payload.idempotency_key)
    if existing is not None:
        if existing.get("decision") != payload.decision:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error_code": "hitl_idempotency_conflict"},
            )
        response_json = existing.get("response_json")
        if isinstance(response_json, str) and response_json:
            return HITLDecisionResponse.model_validate(
                {**json.loads(response_json), "replayed": True},
                strict=False,
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "hitl_decision_in_progress"},
        )

    repository.create_idempotency(
        interaction_id=interaction_id,
        idempotency_key=payload.idempotency_key,
        decision=payload.decision,
    )
    repository.append_audit(
        interaction_id=interaction_id,
        thread_id=thread_id,
        event_type="decision_requested",
        actor="user",
        decision=payload.decision,
        idempotency_key=payload.idempotency_key,
        status="started",
        interrupt_id=interaction.interrupt_id,
        checkpoint_id=interaction.checkpoint_id,
        metadata={"decision_reason": payload.decision_reason},
    )
    try:
        if payload.decision == "approve":
            decision_response = approve_interaction(
                request,
                interaction_id,
                DecideHITLRequest(decision_reason=payload.decision_reason),
                thread_id,
            )
            native_decision = "approved"
        else:
            decision_response = reject_interaction(
                request,
                interaction_id,
                RejectHITLRequest(decision_reason=payload.decision_reason),
                thread_id,
            )
            native_decision = "rejected"

        repository.append_audit(
            interaction_id=interaction_id,
            thread_id=thread_id,
            event_type="decision_recorded",
            actor="user",
            decision=payload.decision,
            idempotency_key=payload.idempotency_key,
            status=decision_response.status,
            interrupt_id=decision_response.interrupt_id,
            checkpoint_id=decision_response.checkpoint_id,
        )
        resume_response = await resume_interaction(
            request,
            interaction_id,
            thread_id,
        )
        result = HITLDecisionResponse(
            interaction=_interaction_response(
                request,
                service.get_interaction(interaction_id),
            ),
            resume=resume_response,
        )
        response_data = result.model_dump(mode="json")
        repository.update_idempotency(
            interaction_id=interaction_id,
            idempotency_key=payload.idempotency_key,
            status="completed" if resume_response.resumed else "failed",
            response=response_data,
            error_code=resume_response.error_code,
        )
        repository.append_audit(
            interaction_id=interaction_id,
            thread_id=thread_id,
            event_type="resume_completed" if resume_response.resumed else "resume_failed",
            actor="user",
            decision=native_decision,
            idempotency_key=payload.idempotency_key,
            status="completed" if resume_response.resumed else "failed",
            error_code=resume_response.error_code,
            interrupt_id=service.get_interaction(interaction_id).interrupt_id,
            checkpoint_id=service.get_interaction(interaction_id).checkpoint_id,
        )
        return result
    except HTTPException as error:
        error_detail = error.detail if isinstance(error.detail, dict) else {}
        error_code = error_detail.get("error_code") if isinstance(error_detail, dict) else None
        repository.update_idempotency(
            interaction_id=interaction_id,
            idempotency_key=payload.idempotency_key,
            status="failed",
            error_code=error_code or "hitl_decision_failed",
        )
        repository.append_audit(
            interaction_id=interaction_id,
            thread_id=thread_id,
            event_type="decision_failed",
            actor="user",
            decision=payload.decision,
            idempotency_key=payload.idempotency_key,
            status="failed",
            error_code=error_code or "hitl_decision_failed",
        )
        raise
    except Exception as error:
        repository.update_idempotency(
            interaction_id=interaction_id,
            idempotency_key=payload.idempotency_key,
            status="failed",
            error_code="hitl_decision_failed",
        )
        repository.append_audit(
            interaction_id=interaction_id,
            thread_id=thread_id,
            event_type="decision_failed",
            actor="user",
            decision=payload.decision,
            idempotency_key=payload.idempotency_key,
            status="failed",
            error_code="hitl_decision_failed",
        )
        raise _http_error(error) from error


@router.post(
    "/hitl/interactions/{interaction_id}/resume",
    response_model=HITLResumeResponse,
)
async def resume_interaction(
    request: Request,
    interaction_id: str,
    thread_id: str = Query(min_length=1, max_length=128),
) -> HITLResumeResponse:
    """Resume a matching LangGraph-native HITL interrupt, if present."""

    try:
        service = _service(request)
        interaction = _get_thread_interaction(
            service,
            interaction_id,
            thread_id,
        )
        if interaction.status is HITLStatus.RESUMED:
            return HITLResumeResponse(
                interaction_id=interaction_id,
                resumed=True,
                answer="该审批已经恢复过，Agent 无需重复执行。",
            )

        if interaction.status is HITLStatus.RESUMING:
            # A second browser click can arrive while the first graph resume
            # is still running.  Treat it as an idempotent in-flight request
            # instead of attempting a second Command(resume).
            return HITLResumeResponse(
                interaction_id=interaction_id,
                resumed=True,
                answer="恢复请求正在处理中，请稍候查看会话中的后续反馈。",
            )

        if interaction.is_expired or interaction.status is HITLStatus.EXPIRED:
            if interaction.status is HITLStatus.PENDING:
                service.expire(interaction_id)
            raise HITLDecisionError(
                "hitl_interaction_expired",
                "HITL interaction has expired",
            )

        if interaction.status not in {
            HITLStatus.APPROVED,
            HITLStatus.REJECTED,
            HITLStatus.FAILED,
        }:
            raise HITLDecisionError(
                "hitl_interaction_not_resolved",
                "HITL interaction is not resolved",
            )

        agent = getattr(request.app.state, "agent", None)
        if agent is None:
            raise HITLDecisionError(
                "hitl_graph_unavailable",
                "The graph is not available for resume",
            )

        # Verify the exact native pause before claiming the durable record.
        # An approved record without its graph interrupt is an old/stale card,
        # not a reason to issue Command(resume) blindly.
        try:
            snapshot, interrupts = await native_hitl_snapshot(
                agent,
                thread_id=thread_id,
            )
        except Exception as error:
            raise HITLDecisionError(
                "hitl_graph_unavailable",
                "The graph state could not be inspected.",
            ) from error

        matching_record = None
        for record in interrupts:
            value = getattr(record, "value", None)
            if isinstance(record, dict):
                value = record.get("value")
            if (
                isinstance(value, dict)
                and value.get("kind") == "publication_approval"
                and value.get("interaction_id") == interaction_id
            ):
                matching_record = record
                break

        if matching_record is None:
            if interaction.status in {
                HITLStatus.APPROVED,
                HITLStatus.REJECTED,
                HITLStatus.FAILED,
            }:
                service.mark_stale(interaction_id)
            return HITLResumeResponse(
                interaction_id=interaction_id,
                resumed=False,
                error_code="hitl_interaction_stale",
            )

        current_interrupt_id = interrupt_id(matching_record)
        if (
            interaction.interrupt_id
            and current_interrupt_id
            and interaction.interrupt_id != current_interrupt_id
        ):
            service.mark_stale(interaction_id, error_code="hitl_interrupt_mismatch")
            return HITLResumeResponse(
                interaction_id=interaction_id,
                resumed=False,
                error_code="hitl_interaction_stale",
            )
        if current_interrupt_id:
            service.bind_runtime(
                interaction_id,
                interrupt_id=current_interrupt_id,
                checkpoint_id=checkpoint_id(snapshot),
            )

        # Keep the durable interaction lifecycle aligned with the graph
        # lifecycle.  The previous implementation called mark_resumed()
        # directly while the record was still APPROVED, which could leave a
        # successfully resumed graph recorded as APPROVED and make later
        # clicks report hitl_graph_not_interrupted.
        decision = (
            interaction.status.value
            if interaction.status in {
                HITLStatus.APPROVED,
                HITLStatus.REJECTED,
            }
            else HITLStatus.APPROVED.value
        )
        interaction, claimed = service.claim_resume(interaction_id)
        if not claimed:
            if interaction.status is HITLStatus.RESUMING:
                return HITLResumeResponse(
                    interaction_id=interaction_id,
                    resumed=True,
                    answer="恢复请求正在处理中，请稍候查看会话中的后续反馈。",
                )
            # The interaction may have changed between the initial read and
            # the atomic claim. Let the normal domain error explain the
            # unexpected state instead of issuing a second graph resume.
            interaction.start_resume()
        # A failed interaction is also retried with the original approval
        # decision.  The publication approval itself remains the source of
        # truth for that decision.
        try:
            result = await resume_agent_after_hitl(
                agent,
                thread_id=thread_id,
                interaction_id=interaction_id,
                decision=decision,
                expected_interrupt_id=(
                    current_interrupt_id or interaction.interrupt_id
                ),
            )
        except Exception as error:
            service.mark_failed(
                interaction_id,
                error_code="hitl_resume_failed",
            )
            raise HITLDecisionError(
                "hitl_resume_failed",
                "The Agent graph could not be resumed.",
            ) from error
        if result.get("resumed"):
            service.mark_resumed(interaction_id)
        else:
            error_code = str(
                result.get("error_code") or "hitl_resume_failed"
            )
            if error_code == "hitl_graph_not_interrupted":
                service.mark_stale(interaction_id)
                error_code = "hitl_interaction_stale"
            else:
                service.mark_failed(
                    interaction_id,
                    error_code=error_code,
                )
            log_event(
                logger,
                logging.WARNING,
                "hitl.resume.not_found",
                thread_id=thread_id,
                status="failed",
                error_code=str(
                    result.get("error_code") or "hitl_resume_failed"
                ),
            )
        return HITLResumeResponse(
            interaction_id=interaction_id,
            resumed=bool(result.get("resumed")),
            answer=(
                result.get("answer")
                if isinstance(result.get("answer"), str)
                else None
            ),
            error_code=(
                result.get("error_code")
                if isinstance(result.get("error_code"), str)
                else None
            ),
            next_intent=(
                result.get("next_intent")
                if isinstance(result.get("next_intent"), dict)
                else None
            ),
            interrupt_id=(
                result.get("interrupt_id")
                if isinstance(result.get("interrupt_id"), str)
                else None
            ),
            checkpoint_id=(
                result.get("checkpoint_id")
                if isinstance(result.get("checkpoint_id"), str)
                else None
            ),
        )
    except HTTPException:
        raise
    except Exception as error:
        raise _http_error(error) from error


__all__ = [
    "DecideHITLRequest",
    "HITLInteractionResponse",
    "HITLDecisionResponse",
    "HITLResumeResponse",
    "RejectHITLRequest",
    "UnifiedHITLDecisionRequest",
    "router",
]
