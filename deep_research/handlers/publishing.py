"""HTTP boundary for deterministic publishing operations."""

from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..log.logging_utils import log_event
from ..publishing.approvals import ApprovalRequest, ApprovalStatus
from ..publishing.artifacts import ArtifactIntegrityError, ArtifactNotFoundError
from ..publishing.models import (
    Article,
    ArticleStatus,
    Publication,
    PublicationChannel,
    PublicationStatus,
    InvalidStateTransitionError,
)
from ..publishing.repository import (
    IdempotencyConflictError,
    RecordNotFoundError,
)
from ..publishing.service import (
    ArticleImportConflictError,
    PublicationValidationError,
    PublisherConfigurationError,
    PublishingServiceError,
)
from ..hitl.models import HITLAction, HITLStatus
from ..hitl.publication_state import resolve_publication_hitl_state
from ..handlers.hitl import (
    HITLDecisionResponse,
    UnifiedHITLDecisionRequest,
    decide_interaction,
    resume_interaction,
)


router = APIRouter()
logger = logging.getLogger("deep_research.publishing.api")


def _sync_publication_hitl_decision(
    request: Request,
    approval: ApprovalRequest,
    *,
    approved: bool,
    decision_reason: str | None = None,
) -> None:
    """Keep a conversation HITL record in sync with center-side decisions."""

    hitl_service = getattr(request.app.state, "hitl_service", None)
    if hitl_service is None:
        return

    interactions = hitl_service.list_for_target(
        "publication_approval",
        approval.approval_id,
        target_version=approval.article_version,
        statuses=(HITLStatus.PENDING,),
    )
    for interaction in interactions:
        if interaction.action is not HITLAction.APPROVE:
            continue
        if approved:
            hitl_service.approve(
                interaction.interaction_id,
                decision_actor="user",
                decision_reason=decision_reason,
            )
        else:
            hitl_service.reject(
                interaction.interaction_id,
                decision_actor="user",
                decision_reason=decision_reason,
            )


def _publication_hitl_interaction(request: Request, approval: ApprovalRequest):
    """Find the conversation HITL record that owns this center approval."""

    hitl_service = getattr(request.app.state, "hitl_service", None)
    if hitl_service is None:
        return None
    interactions = hitl_service.list_for_target(
        "publication_approval",
        approval.approval_id,
        target_version=approval.article_version,
        statuses=(
            HITLStatus.PENDING,
            HITLStatus.APPROVED,
            HITLStatus.REJECTED,
            HITLStatus.RESUMING,
            HITLStatus.FAILED,
        ),
    )
    return next(
        (
            interaction
            for interaction in interactions
            if interaction.action is HITLAction.APPROVE
        ),
        None,
    )


async def _continue_publication_hitl_from_center(
    request: Request,
    approval: ApprovalRequest,
    *,
    decision: str,
    decision_reason: str | None = None,
) -> None:
    """Use the conversation HITL lifecycle when the center owns the click."""

    interaction = _publication_hitl_interaction(request, approval)
    if interaction is None:
        return

    thread_id = interaction.thread_id
    if interaction.status is HITLStatus.PENDING:
        # Some lightweight/legacy app embeddings expose only HITLService.
        # Keep those integrations compatible while the full application uses
        # HITLDecisionService to update both databases atomically-by-retry.
        if getattr(request.app.state, "hitl_decision_service", None) is None:
            if decision == "approve" and approval.status is ApprovalStatus.PENDING:
                request.app.state.publishing_service.approve_publication_request(
                    approval.approval_id,
                    decision_actor="user",
                    decision_reason=decision_reason,
                )
            elif decision != "approve" and approval.status is ApprovalStatus.PENDING:
                request.app.state.publishing_service.reject_publication_request(
                    approval.approval_id,
                    decision_actor="user",
                    decision_reason=decision_reason,
                )
        try:
            await decide_interaction(
                request,
                interaction.interaction_id,
                UnifiedHITLDecisionRequest(
                    decision=decision,
                    decision_reason=decision_reason,
                    idempotency_key=(
                        f"center-{approval.approval_id}-{decision}"
                    ),
                ),
                thread_id,
            )
        except HTTPException as error:
            # The publication decision has already been durably recorded.
            # A temporarily unavailable graph must not make the center report
            # that the approval itself failed; the conversation recovery card
            # can resume it later.
            if error.status_code >= 500:
                return
            raise
        return

    if interaction.status in {
        HITLStatus.APPROVED,
        HITLStatus.REJECTED,
        HITLStatus.FAILED,
    }:
        try:
            await resume_interaction(
                request,
                interaction.interaction_id,
                thread_id,
            )
        except HTTPException as error:
            if error.status_code >= 500:
                return
            raise


def _timestamp(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _thread_id(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error_code": "invalid_thread_id"},
        )
    return normalized

# 从制品导入文章
class CreateArticleRequest(BaseModel):
    """Client-editable metadata used when importing an Artifact."""

    thread_id: str = Field(min_length=1, max_length=128)
    artifact_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=240)
    slug: str | None = Field(default=None, max_length=120)
    excerpt: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = Field(default=None, max_length=32)

# 编辑草稿
class UpdateArticleRequest(BaseModel):
    """Fields that may be edited while an Article remains a draft."""

    title: str | None = Field(default=None, max_length=240)
    slug: str | None = Field(default=None, max_length=120)
    markdown_content: str | None = Field(default=None, max_length=1_000_000) # 允许最大 1,000,000 字符
    excerpt: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = Field(default=None, max_length=32)

# 发布命令
class PublishArticleRequest(BaseModel):
    """Explicit publish command; the client must provide an idempotency key."""

    channel: PublicationChannel = PublicationChannel.LOCAL_STATIC_SITE
    idempotency_key: str = Field(min_length=1, max_length=128) # 必填且非空（长度 1~128），用于保证发布操作的幂等性
    attachment_ids: list[str] | None = Field(default=None, max_length=9)


class RetryPublicationRequest(BaseModel):
    """Explicit retry command for a publication-center action."""

    idempotency_key: str = Field(min_length=1, max_length=128)
    confirm_delivery_unknown: bool = False

# 创建审核请求
class CreateApprovalRequest(BaseModel):
    """Channel selected for a human publication decision."""

    channel: PublicationChannel = PublicationChannel.LOCAL_STATIC_SITE
    attachment_ids: list[str] | None = Field(default=None, max_length=9)

# 批准决策
class DecideApprovalRequest(BaseModel):
    """Optional context for an explicit human approval decision."""

    decision_reason: str | None = Field(default=None, max_length=500)

# 拒绝决策
class RejectApprovalRequest(BaseModel):
    """Required explanation for rejecting a publication request."""

    decision_reason: str = Field(min_length=1, max_length=500)

# 文章完整响应
class ArticleResponse(BaseModel):
    article_id: str
    source_thread_id: str
    source_artifact_id: str
    source_artifact_sha256: str
    title: str
    slug: str
    excerpt: str
    tags: list[str]
    markdown_content: str
    status: ArticleStatus
    version: int
    created_at: str
    updated_at: str
    approved_at: str | None
    published_at: str | None

# 发布记录完整响应
class PublicationResponse(BaseModel):
    publication_id: str
    article_id: str
    article_version: int
    channel: PublicationChannel
    status: PublicationStatus
    idempotency_key: str
    content_sha256: str
    external_id: str | None
    public_url: str | None
    attachment_ids: list[str]
    attempt_count: int
    error_code: str | None
    created_at: str
    updated_at: str
    published_at: str | None


class ImageAttachmentResponse(BaseModel):
    attachment_id: str
    thread_id: str
    filename: str
    content_type: str
    size_bytes: int
    content_sha256: str
    created_at: str
    is_active: bool = False
    cover_asset_id: str | None = None
    analysis_status: str = "pending"
    analysis_type: str | None = None
    analysis_confidence: float | None = None
    analysis_summary: str | None = None
    analysis_ocr_text: str | None = None
    analysis_error_code: str | None = None


class DeleteImageAttachmentResponse(BaseModel):
    ok: bool
    attachment_id: str
    status: str
    message: str | None = None


class SetWeChatCoverRequest(BaseModel):
    """Optional source thread retained for backwards-compatible clients."""

    thread_id: str | None = Field(default=None, min_length=1, max_length=128)


class WeChatCoverResponse(BaseModel):
    ok: bool
    attachment_id: str
    cover_asset_id: str
    status: str
    message: str | None = None


class WeChatPreviewResponse(BaseModel):
    configured: bool
    publish_mode: str | None
    attachment_id: str | None = None
    cover_asset_id: str | None = None
    filename: str | None = None
    content_type: str | None = None
    content_sha256: str | None = None

# 审核请求响应
class ApprovalRequestResponse(BaseModel):
    approval_id: str
    action: str
    article_id: str
    article_version: int
    channel: PublicationChannel
    content_sha256: str
    attachment_ids: list[str]
    status: str
    created_at: str
    updated_at: str
    decided_at: str | None
    decision_actor: str | None
    decision_reason: str | None
    hitl_interaction_id: str | None = None
    hitl_status: str | None = None
    workflow_status: str
    state_source: str
    requires_user_confirmation: bool = False

# 将领域对象 Article 转换为 ArticleResponse DTO
def _article_response(article: Article) -> ArticleResponse:
    return ArticleResponse(
        article_id=article.article_id,
        source_thread_id=article.source_thread_id,
        source_artifact_id=article.source_artifact_id,
        source_artifact_sha256=article.source_artifact_sha256,
        title=article.title,
        slug=article.slug,
        excerpt=article.excerpt,
        tags=article.tags,
        markdown_content=article.markdown_content,
        status=article.status,
        version=article.version,
        created_at=_timestamp(article.created_at),
        updated_at=_timestamp(article.updated_at),
        approved_at=_timestamp(article.approved_at),
        published_at=_timestamp(article.published_at),
    )

# 将领域对象 Publication 转换为 PublicationResponse
def _publication_response(publication: Publication) -> PublicationResponse:
    return PublicationResponse(
        publication_id=publication.publication_id,
        article_id=publication.article_id,
        article_version=publication.article_version,
        channel=publication.channel,
        status=publication.status,
        idempotency_key=publication.idempotency_key,
        content_sha256=publication.content_sha256,
        external_id=publication.external_id,
        public_url=publication.public_url,
        attachment_ids=list(publication.attachment_ids),
        attempt_count=publication.attempt_count,
        error_code=publication.error_code,
        created_at=_timestamp(publication.created_at),
        updated_at=_timestamp(publication.updated_at),
        published_at=_timestamp(publication.published_at),
    )


def _attachment_response(
    attachment: Any,
    *,
    active_asset: Any = None,
    analysis: Any = None,
) -> ImageAttachmentResponse:
    is_active = bool(
        active_asset is not None
        and attachment.content_sha256 == active_asset.content_sha256
    )
    return ImageAttachmentResponse(
        attachment_id=attachment.attachment_id,
        thread_id=attachment.thread_id,
        filename=attachment.filename,
        content_type=attachment.content_type,
        size_bytes=attachment.size_bytes,
        content_sha256=attachment.content_sha256,
        created_at=_timestamp(attachment.created_at),
        is_active=is_active,
        cover_asset_id=active_asset.asset_id if is_active else None,
        analysis_status=(analysis.status if analysis is not None else "pending"),
        analysis_type=(analysis.image_type if analysis is not None else None),
        analysis_confidence=(analysis.confidence if analysis is not None else None),
        analysis_summary=(analysis.summary if analysis is not None else None),
        analysis_ocr_text=(analysis.ocr_text if analysis is not None else None),
        analysis_error_code=(analysis.error_code if analysis is not None else None),
    )

# 将 ApprovalRequest 领域对象转换为响应 DTO
def _approval_response(
    request: Request,
    approval: ApprovalRequest,
) -> ApprovalRequestResponse:
    state = resolve_publication_hitl_state(
        approval,
        getattr(request.app.state, "hitl_service", None),
    )
    return ApprovalRequestResponse(
        approval_id=approval.approval_id,
        action=approval.action.value,
        article_id=approval.article_id,
        article_version=approval.article_version,
        channel=approval.channel,
        content_sha256=approval.content_sha256,
        attachment_ids=list(approval.attachment_ids),
        # Keep the legacy approval status for compatibility.  The workflow
        # status is authoritative for HITL-aware clients.
        status=approval.status.value,
        created_at=_timestamp(approval.created_at),
        updated_at=_timestamp(approval.updated_at),
        decided_at=_timestamp(approval.decided_at),
        decision_actor=approval.decision_actor,
        decision_reason=approval.decision_reason,
        hitl_interaction_id=state.interaction_id,
        hitl_status=state.interaction_status,
        workflow_status=state.workflow_status,
        state_source=state.state_source,
        requires_user_confirmation=state.requires_user_confirmation,
    )

# 统一异常到 HTTP 响应的映射
def _http_error(
    error: Exception,
    *,
    not_found_code: str = "article_not_found",
) -> HTTPException:
    if isinstance(error, ArtifactNotFoundError):
        code, http_status = "artifact_not_found", status.HTTP_404_NOT_FOUND # 资源不存在（制品找不到、记录找不到）
    elif isinstance(error, ArtifactIntegrityError):
        code, http_status = "artifact_integrity_failed", status.HTTP_409_CONFLICT # 409：冲突（完整性失败、幂等冲突、非法状态流转）
    elif isinstance(error, RecordNotFoundError):
        code, http_status = not_found_code, status.HTTP_404_NOT_FOUND
    elif isinstance(error, IdempotencyConflictError):
        code, http_status = "idempotency_conflict", status.HTTP_409_CONFLICT
    elif isinstance(error, PublicationValidationError):
        code, http_status = (
            error.error_code,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    elif isinstance(error, InvalidStateTransitionError):
        code, http_status = "invalid_state_transition", status.HTTP_409_CONFLICT
    elif isinstance(error, PublisherConfigurationError):
        code, http_status = error.error_code, status.HTTP_400_BAD_REQUEST # 客户端请求参数有误（渠道配置错误、通用服务错误）
    elif isinstance(error, PublishingServiceError):
        code, http_status = "invalid_publishing_request", status.HTTP_400_BAD_REQUEST
    else:
        code, http_status = "publishing_operation_failed", status.HTTP_500_INTERNAL_SERVER_ERROR # 未预期的其他异常（服务端内部错误）
    # 记录结构化日志：4xx 记 WARNING（客户端问题），5xx 记 ERROR（服务端问题）
    log_event(
        logger,
        logging.WARNING if http_status < 500 else logging.ERROR,
        "publishing.api.failed",
        status="failed",
        error_code=code,
    )
    return HTTPException(
        status_code=http_status,
        detail={"error_code": code},
    )

# Pydantic 模型转字典
def _model_fields(payload: BaseModel) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_unset=True)
    return payload.dict(exclude_unset=True)


def _publisher_for_channel(
    request: Request,
    channel: PublicationChannel,
) -> Any:
    """Resolve a configured publisher without silently falling back."""

    if channel is PublicationChannel.LOCAL_STATIC_SITE:
        publisher = getattr(request.app.state, "local_static_publisher", None)
        error_code = "local_publisher_not_configured"
    elif channel is PublicationChannel.WECHAT_OFFICIAL_ACCOUNT:
        publisher = getattr(request.app.state, "wechat_publisher", None)
        error_code = "wechat_not_configured"
    elif channel is PublicationChannel.XIAOHONGSHU:
        publisher = getattr(request.app.state, "xiaohongshu_publisher", None)
        error_code = "xiaohongshu_not_configured"
    elif channel is PublicationChannel.DOUYIN:
        publisher = getattr(request.app.state, "douyin_publisher", None)
        error_code = "douyin_not_configured"
    else:
        raise PublisherConfigurationError("unsupported publication channel")

    if publisher is None:
        raise PublisherConfigurationError(
            "requested publication channel is not configured",
            error_code=error_code,
        )
    return publisher


@router.post(
    "/publishing/articles/from-artifact",
    response_model=ArticleResponse,
    status_code=status.HTTP_201_CREATED,
)
# 从制品创建文章
async def create_article_from_artifact(
    request: Request,
    payload: CreateArticleRequest,
) -> ArticleResponse:
    try:
        article = await request.app.state.publishing_service.create_article_from_artifact(
            thread_id=payload.thread_id,
            artifact_id=payload.artifact_id,
            title=payload.title,
            slug=payload.slug,
            excerpt=payload.excerpt,
            tags=payload.tags,
        )
        return _article_response(article)
    except Exception as error:
        raise _http_error(error) from error


@router.post(
    "/publishing/attachments/images",
    response_model=ImageAttachmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_image_attachment(
    request: Request,
    thread_id: str = Form(..., min_length=1, max_length=128),
    file: UploadFile = File(...),
) -> ImageAttachmentResponse:
    """Store one user-uploaded image in the shared library."""
    normalized_thread_id = _thread_id(thread_id)
    try:
        service = getattr(request.app.state, "image_attachment_service", None)
        if service is None:
            raise PublisherConfigurationError(
                "image attachments are not configured",
                error_code="image_attachment_not_configured",
            )
        content = await file.read()
        attachment = await asyncio.to_thread(
            service.save,
            thread_id=normalized_thread_id,
            filename=file.filename or "image",
            content_type=file.content_type or "application/octet-stream",
            content=content,
        )
        # Uploading only stores the shared image. Vision analysis is an
        # explicit, message-scoped Agent action and must not start while an
        # image is still sitting in the composer waiting to be sent.
        # Read an existing derived result for idempotent re-uploads, but do
        # not create or schedule a new analysis job here.
        analysis = await asyncio.to_thread(
            service.repository.get_image_attachment_analysis,
            attachment.attachment_id,
        )
        return _attachment_response(attachment, analysis=analysis)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": str(error) or "invalid_image_attachment"},
        ) from error
    except Exception as error:
        raise _http_error(error) from error
    finally:
        await file.close()


@router.get(
    "/publishing/attachments/images",
    response_model=list[ImageAttachmentResponse],
)
def list_image_attachments(
    request: Request,
    thread_id: str | None = Query(default=None, min_length=1, max_length=128),
) -> list[ImageAttachmentResponse]:
    service = getattr(request.app.state, "image_attachment_service", None)
    if service is None:
        raise _http_error(
            PublisherConfigurationError(
                "image attachments are not configured",
                error_code="image_attachment_not_configured",
            )
        )
    active_asset = service.repository.get_active_wechat_cover_asset()
    return [
        _attachment_response(
            item,
            active_asset=active_asset,
            analysis=service.repository.get_image_attachment_analysis(
                item.attachment_id,
            ),
        )
        for item in service.repository.list_image_attachments()
    ]


@router.get(
    "/publishing/attachments/images/{attachment_id}/content",
    response_class=FileResponse,
)
def get_image_attachment_content(
    request: Request,
    attachment_id: str,
    thread_id: str | None = Query(default=None, min_length=1, max_length=128),
) -> FileResponse:
    """Serve one image from the shared user-owned attachment library."""
    service = getattr(request.app.state, "image_attachment_service", None)
    if service is None:
        raise _http_error(
            PublisherConfigurationError(
                "image attachments are not configured",
                error_code="image_attachment_not_configured",
            )
        )
    try:
        attachment = service.get(attachment_id.strip())
        return FileResponse(
            attachment.storage_path,
            media_type=attachment.content_type,
            filename=attachment.filename,
        )
    except ValueError as error:
        code = str(error).strip() or "invalid_image_attachment"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": code},
        ) from error
    except RecordNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "image_attachment_not_found"},
        ) from error
    except Exception as error:
        raise _http_error(error, not_found_code="image_attachment_not_found") from error


@router.delete(
    "/publishing/attachments/{attachment_id}",
    response_model=DeleteImageAttachmentResponse,
)
def delete_image_attachment(
    request: Request,
    attachment_id: str,
) -> DeleteImageAttachmentResponse:
    service = getattr(request.app.state, "image_attachment_service", None)
    if service is None:
        raise _http_error(
            PublisherConfigurationError(
                "image attachments are not configured",
                error_code="image_attachment_not_configured",
            )
        )
    try:
        deleted = service.delete(attachment_id.strip())
        return DeleteImageAttachmentResponse(
            ok=True,
            attachment_id=deleted.attachment_id,
            status="deleted",
            message="图片已从共享图片库删除",
        )
    except RecordNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "image_attachment_not_found"},
        ) from error
    except Exception as error:
        raise _http_error(error, not_found_code="image_attachment_not_found") from error


@router.post(
    "/publishing/attachments/{attachment_id}/wechat-cover",
    response_model=WeChatCoverResponse,
)
async def set_wechat_cover(
    request: Request,
    attachment_id: str,
    payload: SetWeChatCoverRequest,
) -> WeChatCoverResponse:
    """Promote any shared-library image without exposing local or WeChat secrets."""
    attachment_service = getattr(
        request.app.state, "image_attachment_service", None
    )
    cover_service = getattr(request.app.state, "wechat_cover_service", None)
    if attachment_service is None:
        raise _http_error(
            PublisherConfigurationError(
                "image attachments are not configured",
                error_code="image_attachment_not_configured",
            )
        )
    if cover_service is None:
        raise _http_error(
            PublisherConfigurationError(
                "wechat cover service is not configured",
                error_code="wechat_cover_not_configured",
            )
        )

    try:
        attachment = attachment_service.get(attachment_id.strip())
        content = await asyncio.to_thread(
            lambda: Path(attachment.storage_path).read_bytes()
        )
        asset = await asyncio.to_thread(
            cover_service.upload_cover,
            content,
            filename=attachment.filename,
            content_type=attachment.content_type,
            make_active=True,
        )
        return WeChatCoverResponse(
            ok=True,
            attachment_id=attachment.attachment_id,
            cover_asset_id=asset.asset_id,
            status="active",
            message="当前图片已设为微信公众号封面",
        )
    except ValueError as error:
        code = str(error).strip() or "wechat_cover_attachment_failed"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": code},
        ) from error
    except RecordNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "image_attachment_not_found"},
        ) from error
    except Exception as error:
        raise _http_error(error, not_found_code="image_attachment_not_found") from error


@router.get(
    "/publishing/wechat/preview",
    response_model=WeChatPreviewResponse,
)
def get_wechat_preview(
    request: Request,
    thread_id: str | None = Query(default=None, min_length=1, max_length=128),
) -> WeChatPreviewResponse:
    """Return only safe cover and mode metadata for the publishing preview."""
    attachment_service = getattr(
        request.app.state, "image_attachment_service", None
    )
    cover_service = getattr(request.app.state, "wechat_cover_service", None)
    publisher = getattr(request.app.state, "wechat_publisher", None)
    if attachment_service is None:
        raise _http_error(
            PublisherConfigurationError(
                "image attachments are not configured",
                error_code="image_attachment_not_configured",
            )
        )

    active_asset = attachment_service.repository.get_active_wechat_cover_asset()
    active_attachment = None
    if active_asset is not None:
        active_attachment = next(
            (
                item
                for item in attachment_service.repository.list_image_attachments()
                if item.content_sha256 == active_asset.content_sha256
            ),
            None,
        )
    publish_mode = getattr(publisher, "publish_mode", None)
    configured = bool(cover_service is not None and publisher is not None)
    return WeChatPreviewResponse(
        configured=configured,
        publish_mode=publish_mode if publish_mode in {"draft", "publish"} else None,
        attachment_id=(
            active_attachment.attachment_id if active_attachment is not None else None
        ),
        cover_asset_id=active_asset.asset_id if active_asset is not None else None,
        filename=active_attachment.filename if active_attachment is not None else None,
        content_type=(
            active_attachment.content_type if active_attachment is not None else None
        ),
        content_sha256=(
            active_attachment.content_sha256
            if active_attachment is not None
            else None
        ),
    )


@router.get(
    "/publishing/articles",
    response_model=list[ArticleResponse],
)#  列出所有文章
def list_articles(request: Request) -> list[ArticleResponse]:
    try:
        return [
            _article_response(article)
            for article in request.app.state.publishing_service.list_articles()
        ]
    except Exception as error:
        raise _http_error(error) from error


@router.get(
    "/publishing/articles/{article_id}/publications",
    response_model=list[PublicationResponse],
)
# 列出发布记录
def list_publications(
    request: Request,
    article_id: str,
) -> list[PublicationResponse]:
    try:
        publications = request.app.state.publishing_service.list_publications(
            article_id
        )
        return [_publication_response(publication) for publication in publications]
    except Exception as error:
        raise _http_error(error) from error


@router.get(
    "/publishing/articles/{article_id}",
    response_model=ArticleResponse,
)
# 获取单篇文章
def get_article(request: Request, article_id: str) -> ArticleResponse:
    try:
        return _article_response(
            request.app.state.publishing_service.get_article(article_id)
        )
    except Exception as error:
        raise _http_error(error) from error


@router.patch(
    "/publishing/articles/{article_id}",
    response_model=ArticleResponse,
)
# 编辑文章
def update_article(
    request: Request,
    article_id: str,
    payload: UpdateArticleRequest,
) -> ArticleResponse:
    try:
        article = request.app.state.publishing_service.edit_article(
            article_id,
            **_model_fields(payload),
        )
        return _article_response(article)
    except Exception as error:
        raise _http_error(error) from error


@router.post(
    "/publishing/articles/{article_id}/approve",
    response_model=ArticleResponse,
)
# 审核文章
def approve_article(request: Request, article_id: str) -> ArticleResponse:
    try:
        return _article_response(
            request.app.state.publishing_service.approve_article(article_id)
        )
    except Exception as error:
        raise _http_error(error) from error


@router.post(
    "/publishing/articles/{article_id}/approval-requests",
    response_model=ApprovalRequestResponse,
    status_code=status.HTTP_201_CREATED,
)
# 创建审核请求
def create_approval_request(
    request: Request,
    article_id: str,
    payload: CreateApprovalRequest,
) -> ApprovalRequestResponse:
    try:
        approval = (
            request.app.state.publishing_service.request_publication_approval(
                article_id,
                channel=payload.channel,
                actor="application",
                attachment_ids=tuple(payload.attachment_ids or ()),
            )
        )
        return _approval_response(request, approval)
    except Exception as error:
        raise _http_error(error) from error


@router.get(
    "/publishing/articles/{article_id}/approval-requests",
    response_model=list[ApprovalRequestResponse],
)
# 列出审核请求
def list_approval_requests(
    request: Request,
    article_id: str,
) -> list[ApprovalRequestResponse]:
    try:
        approvals = request.app.state.publishing_service.repository.list_approval_requests(
            article_id=article_id,
        )
        return [_approval_response(request, approval) for approval in approvals]
    except Exception as error:
        raise _http_error(error) from error


@router.post(
    "/publishing/approval-requests/{approval_id}/approve",
    response_model=ApprovalRequestResponse,
)
# 批准审核
async def approve_publication_request(
    request: Request,
    approval_id: str,
    payload: DecideApprovalRequest | None = None,
) -> ApprovalRequestResponse:
    try:
        reason = payload.decision_reason if payload is not None else None
        repository = request.app.state.publishing_service.repository
        approval = repository.get_approval_request(approval_id)
        interaction = _publication_hitl_interaction(request, approval)
        if interaction is not None and interaction.status is HITLStatus.PENDING:
            await _continue_publication_hitl_from_center(
                request,
                approval,
                decision="approve",
                decision_reason=reason,
            )
            approval = repository.get_approval_request(approval_id)
        else:
            if approval.status is ApprovalStatus.PENDING:
                approval = request.app.state.publishing_service.approve_publication_request(
                    approval_id,
                    decision_actor="user",
                    decision_reason=reason,
                )
            _sync_publication_hitl_decision(
                request,
                approval,
                approved=True,
                decision_reason=reason,
            )
            if interaction is not None:
                await _continue_publication_hitl_from_center(
                    request,
                    approval,
                    decision="approve",
                    decision_reason=reason,
                )
            approval = repository.get_approval_request(approval_id)
        return _approval_response(request, approval)
    except Exception as error:
        raise _http_error(
            error,
            not_found_code="approval_request_not_found",
        ) from error


@router.post(
    "/publishing/approval-requests/{approval_id}/reject",
    response_model=ApprovalRequestResponse,
)
#  拒绝审核
async def reject_publication_request(
    request: Request,
    approval_id: str,
    payload: RejectApprovalRequest,
) -> ApprovalRequestResponse:
    try:
        repository = request.app.state.publishing_service.repository
        approval = repository.get_approval_request(approval_id)
        interaction = _publication_hitl_interaction(request, approval)
        if interaction is not None and interaction.status is HITLStatus.PENDING:
            await _continue_publication_hitl_from_center(
                request,
                approval,
                decision="reject",
                decision_reason=payload.decision_reason,
            )
            approval = repository.get_approval_request(approval_id)
        else:
            if approval.status is ApprovalStatus.PENDING:
                approval = request.app.state.publishing_service.reject_publication_request(
                    approval_id,
                    decision_actor="user",
                    decision_reason=payload.decision_reason,
                )
            _sync_publication_hitl_decision(
                request,
                approval,
                approved=False,
                decision_reason=payload.decision_reason,
            )
            if interaction is not None:
                await _continue_publication_hitl_from_center(
                    request,
                    approval,
                    decision="reject",
                    decision_reason=payload.decision_reason,
                )
            approval = repository.get_approval_request(approval_id)
        return _approval_response(request, approval)
    except Exception as error:
        raise _http_error(
            error,
            not_found_code="approval_request_not_found",
        ) from error


@router.post(
    "/publishing/publications/{publication_id}/refresh",
    response_model=PublicationResponse,
)
# 刷新外部平台发布状态
def refresh_publication(
    request: Request,
    publication_id: str,
) -> PublicationResponse:
    try:
        publication = request.app.state.publishing_service.repository.get_publication(
            publication_id
        )
        refreshed = request.app.state.publishing_service.refresh_publication(
            publication_id,
            publisher=_publisher_for_channel(request, publication.channel),
        )
        return _publication_response(refreshed)
    except Exception as error:
        raise _http_error(
            error,
            not_found_code="publication_not_found",
        ) from error


@router.post(
    "/publishing/publications/{publication_id}/retry",
    response_model=PublicationResponse,
)
# 发布中心重新发布
def retry_publication(
    request: Request,
    publication_id: str,
    payload: RetryPublicationRequest,
) -> PublicationResponse:
    try:
        publication = request.app.state.publishing_service.repository.get_publication(
            publication_id
        )
        retried = request.app.state.publishing_service.retry_publication(
            publication_id,
            idempotency_key=payload.idempotency_key,
            confirm_delivery_unknown=payload.confirm_delivery_unknown,
            publisher=_publisher_for_channel(request, publication.channel),
        )
        return _publication_response(retried)
    except Exception as error:
        raise _http_error(
            error,
            not_found_code="publication_not_found",
        ) from error


@router.post(
    "/publishing/articles/{article_id}/publish",
    response_model=PublicationResponse,
)
#  发布文章
def publish_article(
    request: Request,
    article_id: str,
    payload: PublishArticleRequest,
) -> PublicationResponse:
    try:
        publication = request.app.state.publishing_service.publish_article(
            article_id,
            channel=payload.channel,
            idempotency_key=payload.idempotency_key,
            publisher=_publisher_for_channel(request, payload.channel),
            attachment_ids=(
                tuple(payload.attachment_ids)
                if payload.attachment_ids is not None
                else None
            ),
        )
        return _publication_response(publication)
    except Exception as error:
        raise _http_error(error) from error


__all__ = [
    "ApprovalRequestResponse",
    "CreateApprovalRequest",
    "ArticleResponse",
    "CreateArticleRequest",
    "DecideApprovalRequest",
    "PublishArticleRequest",
    "RetryPublicationRequest",
    "PublicationResponse",
    "ImageAttachmentResponse",
    "DeleteImageAttachmentResponse",
    "SetWeChatCoverRequest",
    "WeChatCoverResponse",
    "WeChatPreviewResponse",
    "RejectApprovalRequest",
    "UpdateArticleRequest",
    "router",
]
