from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, HttpUrl

from ..config import settings
from ..memory.decision import decide_candidate
from ..memory.service import MemoryService
from ..memory.type import (
    MemoryCandidate,
    MemoryEntry,
    MemoryEntryKind,
    MemoryScope,
)


router = APIRouter()


class MemoryUpdateRequest(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    content: str = Field(min_length=1)
    keywords: list[str]
    url: HttpUrl | None = None
    verified_at: datetime | None = None
    incorrect: str | None = None
    correct: str | None = None
    applies_when: str | None = None


class PublicMemory(BaseModel):
    id: str
    memory_key: str
    scope: MemoryScope
    kind: MemoryEntryKind
    title: str
    summary: str
    content: str
    keywords: list[str]
    source_type: str
    url: HttpUrl | None = None
    verified_at: datetime | None = None
    incorrect: str | None = None
    correct: str | None = None
    applies_when: str | None = None
    created_at: datetime
    updated_at: datetime
    status: str


class MemoryListResponse(BaseModel):
    page: int
    page_size: int
    memories: list[PublicMemory]


class MemoryMutationResponse(BaseModel):
    ok: bool = True
    memory: PublicMemory | None = None
    deleted_memory_id: str | None = None
    deleted_count: int | None = None


def _memory_service(http_request: Request) -> MemoryService:
    service = getattr(
        http_request.app.state,
        "memory_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Memory service is unavailable.",
        )

    return service


def _public_memory(entry: MemoryEntry) -> PublicMemory:
    return PublicMemory.model_validate(
        entry.model_dump(
            mode="json",
            exclude={"source_thread_id"},
        )
    )


def _memory_not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail="Active memory not found.",
    )


async def _active_entry(
    service: MemoryService,
    kind: MemoryEntryKind,
    memory_id: str,
) -> MemoryEntry:
    entry = await service.get_entry(kind, memory_id)

    if entry is None or entry.status != "active":
        raise _memory_not_found()

    return entry


@router.get(
    "/memory",
    response_model=MemoryListResponse,
)
async def list_memories(
    http_request: Request,
    kind: MemoryEntryKind | None = None,
    keyword: str | None = None,
    page: int = Query(default=1, ge=1),
) -> MemoryListResponse:
    service = _memory_service(http_request)

    try:
        entries = await service.list_active_memories(
            kind=kind,
            keyword=keyword,
            page=page,
            page_size=settings.memory_page_size,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return MemoryListResponse(
        page=page,
        page_size=settings.memory_page_size,
        memories=[_public_memory(entry) for entry in entries],
    )


@router.put(
    "/memory/{kind}/{memory_id}",
    response_model=MemoryMutationResponse,
)
async def update_memory(
    kind: MemoryEntryKind,
    memory_id: str,
    payload: MemoryUpdateRequest,
    http_request: Request,
) -> MemoryMutationResponse:
    service = _memory_service(http_request)
    existing = await _active_entry(service, kind, memory_id)

    candidate_data: dict[str, Any] = {
        "kind": existing.kind,
        "memory_key": existing.memory_key,
        "scope": existing.scope,
        "title": payload.title,
        "summary": payload.summary,
        "content": payload.content,
        "keywords": payload.keywords,
        "source_type": "explicit_user_ui",
        "source_thread_id": existing.source_thread_id,
        "url": payload.url if payload.url is not None else existing.url,
        "verified_at": (
            payload.verified_at
            if payload.verified_at is not None
            else existing.verified_at
        ),
        "incorrect": (
            payload.incorrect
            if payload.incorrect is not None
            else existing.incorrect
        ),
        "correct": (
            payload.correct
            if payload.correct is not None
            else existing.correct
        ),
        "applies_when": (
            payload.applies_when
            if payload.applies_when is not None
            else existing.applies_when
        ),
    }

    try:
        candidate = MemoryCandidate.model_validate(candidate_data)
        decision = decide_candidate(
            candidate,
            [existing],
            intent="explicit_update",
        )
        saved = await service.apply_decision(
            candidate,
            decision,
            existing=existing,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return MemoryMutationResponse(
        memory=_public_memory(saved or existing),
    )


@router.delete(
    "/memory/{kind}/{memory_id}",  
    response_model=MemoryMutationResponse,
)
async def delete_memory(
    kind: MemoryEntryKind,
    memory_id: str,
    http_request: Request,
) -> MemoryMutationResponse:
    service = _memory_service(http_request)
    entry = await _active_entry(service, kind, memory_id)

    await service.forget_entry(
        entry.kind,
        entry.id,
        source_thread_id=None,
        reason="explicit_user_forget",
    )

    return MemoryMutationResponse(
        deleted_memory_id=entry.id,
    )


@router.delete(
    "/memory",
    response_model=MemoryMutationResponse,
)
async def clear_memories(
    http_request: Request,
    kind: MemoryEntryKind | None = None,
) -> MemoryMutationResponse:
    service = _memory_service(http_request)

    if kind is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Kind-scoped clear is not supported. "
                "Use clear all without a kind filter."
            ),
        )

    deleted_count = await service.clear_all_memories()

    return MemoryMutationResponse(
        deleted_count=deleted_count,
    )
