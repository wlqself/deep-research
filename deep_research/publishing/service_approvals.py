"""Human publication-approval workflow operations."""

from __future__ import annotations

import hashlib
import logging
from uuid import uuid4

from .approvals import ApprovalAction, ApprovalRequest, ApprovalStatus
from .formatting import check_article_markdown
from .models import (
    Article,
    ArticleStatus,
    InvalidStateTransitionError,
    PublicationChannel,
    PublicationStatus,
)
from .repository import DuplicateRecordError
from .service_support import (
    PublicationValidationError,
    PublisherConfigurationError,
    PublishingServiceError,
)
from .publishers.wechat.drafts import (
    WeChatDraftValidationError,
    validate_wechat_title,
)
from .publishers.xiaohongshu.formatter import format_xiaohongshu_content
from .publishers.douyin.formatter import format_douyin_content


class ApprovalWorkflowMixin:
    """Create and deterministically decide version-pinned approvals."""

    def list_publication_approvals_for_thread(
        self,
        thread_id: str,
        *,
        include_resolved: bool = False,
        limit: int = 20,
    ) -> list[ApprovalRequest]:
        """Return durable approval records relevant to one conversation."""

        statuses = None
        if not include_resolved:
            statuses = (
                ApprovalStatus.PENDING,
                ApprovalStatus.APPROVED,
            )
        return self.repository.list_approval_requests_for_thread(
            thread_id,
            statuses=statuses,
            limit=limit,
        )

    def request_publication_approval(
        self,
        article_id: str,
        *,
        channel: PublicationChannel,
        actor: str = "application",
        attachment_ids: tuple[str, ...] = (),
    ) -> ApprovalRequest:
        """Create or reuse a version-pinned approval request for publishing."""

        if channel not in {
            PublicationChannel.LOCAL_STATIC_SITE,
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            PublicationChannel.XIAOHONGSHU,
            PublicationChannel.DOUYIN,
        }:
            raise PublisherConfigurationError("unsupported publication channel")

        article = self.repository.get_article(article_id)
        if not article.can_request_publication_approval():
            self._record_event(
                event="publishing.approval.request_failed",
                status="failed",
                article=article,
                error_code="article_not_approval_ready",
                actor=actor,
                level=logging.WARNING,
            )
            raise InvalidStateTransitionError(
                "only draft, approved, or published articles can request publication approval"
            )

        format_report = check_article_markdown(article.markdown_content)
        if not format_report.is_publishable:
            error_code = format_report.error_code or "article_markdown_invalid"
            self._record_event(
                event="publishing.approval.request_failed",
                status="failed",
                article=article,
                error_code=error_code,
                actor=actor,
                level=logging.WARNING,
            )
            raise PublicationValidationError(
                error_code,
                "article Markdown failed publication preflight",
            )

        if channel is PublicationChannel.WECHAT_OFFICIAL_ACCOUNT:
            try:
                validate_wechat_title(article.title)
            except WeChatDraftValidationError as error:
                self._record_event(
                    event="publishing.approval.request_failed",
                    status="failed",
                    article=article,
                    error_code=error.error_code,
                    actor=actor,
                    level=logging.WARNING,
                )
                raise PublicationValidationError(
                    error.error_code,
                    str(error),
                ) from error
        elif channel is PublicationChannel.XIAOHONGSHU:
            format_xiaohongshu_content(article)
        elif channel is PublicationChannel.DOUYIN:
            format_douyin_content(article)

        if any(
            publication.article_version == article.version
            and publication.channel is channel
            and publication.status is PublicationStatus.PUBLISHED
            for publication in self.repository.list_publications(article.article_id)
        ):
            self._record_event(
                event="publishing.approval.request_failed",
                status="failed",
                article=article,
                error_code="article_already_published_to_channel",
                actor=actor,
                level=logging.WARNING,
            )
            raise InvalidStateTransitionError(
                "article version is already published to this channel"
            )

        content_sha256 = hashlib.sha256(
            article.markdown_content.encode("utf-8")
        ).hexdigest()
        pending = self.repository.get_pending_approval_request(
            action=ApprovalAction.PUBLISH,
            article_id=article.article_id,
            article_version=article.version,
            channel=channel,
        )
        if pending is not None:
            self._record_event(
                event="publishing.approval.request_replayed",
                status="replayed",
                article=article,
                actor=actor,
            )
            return pending

        approved = next(
            (
                request
                for request in self.repository.list_approval_requests(
                    article_id=article.article_id,
                )
                if request.article_version == article.version
                and request.channel is channel
                and request.status is ApprovalStatus.APPROVED
            ),
            None,
        )
        if approved is not None:
            self._record_event(
                event="publishing.approval.request_replayed",
                status="replayed",
                article=article,
                actor=actor,
            )
            return approved

        request = ApprovalRequest(
            approval_id=uuid4().hex,
            action=ApprovalAction.PUBLISH,
            article_id=article.article_id,
            article_version=article.version,
            channel=channel,
            content_sha256=content_sha256,
            attachment_ids=tuple(attachment_ids),
        )
        try:
            created = self.repository.create_approval_request(request)
        except DuplicateRecordError:
            pending = self.repository.get_pending_approval_request(
                action=ApprovalAction.PUBLISH,
                article_id=article.article_id,
                article_version=article.version,
                channel=channel,
            )
            if pending is not None:
                self._record_event(
                    event="publishing.approval.request_replayed",
                    status="replayed",
                    article=article,
                    actor=actor,
                )
                return pending
            self._record_event(
                event="publishing.approval.request_failed",
                status="failed",
                article=article,
                error_code="approval_request_conflict",
                actor=actor,
                level=logging.WARNING,
            )
            raise PublishingServiceError(
                "publication approval request could not be created"
            )

        self._record_event(
            event="publishing.approval.requested",
            status="waiting_approval",
            article=article,
            actor=actor,
            summary="文章已提交人工发布审批",
        )
        return created

    def approve_publication_request(
        self,
        approval_id: str,
        *,
        decision_actor: str,
        decision_reason: str | None = None,
    ) -> ApprovalRequest:
        """Approve one exact Article version for one publication channel."""

        request, article = self._get_current_approval_target(approval_id)
        try:
            if article.status is ArticleStatus.DRAFT:
                article = self.approve_article(article.article_id)
            request.approve(
                decision_actor=decision_actor,
                decision_reason=decision_reason,
            )
            approved = self.repository.update_approval_request(request)
        except InvalidStateTransitionError:
            self._record_event(
                event="publishing.approval.approve_failed",
                status="failed",
                article=article,
                error_code="approval_not_pending",
                actor=decision_actor,
                level=logging.WARNING,
            )
            raise

        self._record_event(
            event="publishing.approval.approved",
            status="approved",
            article=article,
            actor=decision_actor,
            summary="用户批准发布文章",
        )
        return approved

    def reject_publication_request(
        self,
        approval_id: str,
        *,
        decision_actor: str,
        decision_reason: str | None = None,
    ) -> ApprovalRequest:
        """Reject one pending publication approval request."""

        request, article = self._get_current_approval_target(approval_id)
        try:
            request.reject(
                decision_actor=decision_actor,
                decision_reason=decision_reason,
            )
            rejected = self.repository.update_approval_request(request)
        except InvalidStateTransitionError:
            self._record_event(
                event="publishing.approval.reject_failed",
                status="failed",
                article=article,
                error_code="approval_not_pending",
                actor=decision_actor,
                level=logging.WARNING,
            )
            raise

        self._record_event(
            event="publishing.approval.rejected",
            status="rejected",
            article=article,
            actor=decision_actor,
            summary="用户拒绝发布文章",
        )
        return rejected

    def _get_current_approval_target(
        self,
        approval_id: str,
    ) -> tuple[ApprovalRequest, Article]:
        request = self.repository.get_approval_request(approval_id)
        article = self.repository.get_article(request.article_id)
        content_sha256 = hashlib.sha256(
            article.markdown_content.encode("utf-8")
        ).hexdigest()
        if (
            article.version != request.article_version
            or not article.can_request_publication_approval()
            or content_sha256 != request.content_sha256
        ):
            self._record_event(
                event="publishing.approval.decision_failed",
                status="failed",
                article=article,
                error_code="approval_target_stale",
                level=logging.WARNING,
            )
            raise InvalidStateTransitionError(
                "approval request no longer matches the current article"
            )
        return request, article

__all__ = ["ApprovalWorkflowMixin"]
