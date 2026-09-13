"""Publication execution and publication-history operations."""

from __future__ import annotations

import hashlib
import inspect
import logging
from uuid import uuid4

from .approvals import ApprovalStatus
from .formatting import check_article_markdown
from .models import (
    Article,
    ArticleStatus,
    InvalidStateTransitionError,
    Publication,
    PublicationChannel,
    PublicationStatus,
)
from .repository import (
    DuplicateRecordError,
    IdempotencyConflictError,
)
from .service_support import (
    PublicationDeliveryUnknownError,
    PublicationExecutionError,
    PublicationPollingError,
    PublicationValidationError,
    PublicationPublisher,
    PublicationResult,
    PublisherConfigurationError,
    PublishingServiceError,
)
from .publishers.wechat.drafts import (
    WeChatDraftValidationError,
    validate_wechat_title,
)
from .publishers.xiaohongshu.formatter import format_xiaohongshu_content
from .publishers.douyin.formatter import format_douyin_content

"""
已批准但尚未创建 Publication 时，使用稳定幂等键创建
已存在 publishing 状态时，复用原 Publication
不重复创建发布记录
恢复时递增 attempt_count
发布成功或失败继续写入原有日志和 Audit Log
已发布记录直接返回，不重复执行
"""
class PublicationWorkflowMixin:
    """Execute channel publication only after durable approval."""

    def resume_publication_for_article(
        self,
        article_id: str,
        *,
        channel: PublicationChannel,
        publisher: PublicationPublisher,
    ) -> Publication:
        """Resume an approved publication without creating a new intent.

        The approval request supplies the stable idempotency key.  A missing
        publication starts one; an existing publishing publication is resumed
        in place.  No new approval is created by this method.
        """

        if channel not in {
            PublicationChannel.LOCAL_STATIC_SITE,
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            PublicationChannel.XIAOHONGSHU,
            PublicationChannel.DOUYIN,
        }:
            raise PublisherConfigurationError("unsupported publication channel")
        if publisher.channel is not channel:
            raise PublisherConfigurationError("publisher channel mismatch")

        article = self.repository.get_article(article_id)
        content_sha256 = hashlib.sha256(
            article.markdown_content.encode("utf-8")
        ).hexdigest()
        approved_request = next(
            (
                request
                for request in self.repository.list_approval_requests(
                    article_id=article.article_id,
                    status=ApprovalStatus.APPROVED,
                )
                if request.article_version == article.version
                and request.channel is channel
                and request.content_sha256 == content_sha256
            ),
            None,
        )
        if approved_request is None:
            self._record_event(
                event="publishing.article.resume_failed",
                status="failed",
                article=article,
                error_code="publication_approval_required",
                level=logging.WARNING,
            )
            raise InvalidStateTransitionError(
                "an approved publication request is required"
            )

        if channel is PublicationChannel.WECHAT_OFFICIAL_ACCOUNT:
            try:
                validate_wechat_title(article.title)
            except WeChatDraftValidationError as error:
                self._record_event(
                    event="publishing.article.publish_validation_failed",
                    status="failed",
                    article=article,
                    error_code=error.error_code,
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

        existing = next(
            (
                publication
                for publication in self.repository.list_publications(
                    article.article_id,
                )
                if publication.article_version == article.version
                and publication.channel is channel
            ),
            None,
        )
        if existing is None:
            return self.publish_article(
                article.article_id,
                channel=channel,
                idempotency_key=f"approval-{approved_request.approval_id}",
                publisher=publisher,
            )
        if existing.status in {
            PublicationStatus.PUBLISHED,
            PublicationStatus.DRAFTED,
            PublicationStatus.DELIVERY_UNKNOWN,
        }:
            return existing
        if existing.status is not PublicationStatus.PUBLISHING:
            raise InvalidStateTransitionError(
                "only an in-progress publication can be resumed"
            )
        return self._resume_existing_publication(
            article,
            existing,
            publisher,
        )

    def publish_article(
        self,
        article_id: str,
        *,
        channel: PublicationChannel,
        idempotency_key: str,
        publisher: PublicationPublisher,
        attachment_ids: tuple[str, ...] | None = None,
    ) -> Publication:
        """Publish through an explicit channel publisher exactly once per key."""

        if channel not in {
            PublicationChannel.LOCAL_STATIC_SITE,
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            PublicationChannel.XIAOHONGSHU,
            PublicationChannel.DOUYIN,
        }:
            raise PublisherConfigurationError("unsupported publication channel")
        if publisher.channel is not channel:
            raise PublisherConfigurationError("publisher channel mismatch")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise PublishingServiceError("idempotency key must not be empty")

        article = self.repository.get_article(article_id)
        content_sha256 = hashlib.sha256(
            article.markdown_content.encode("utf-8")
        ).hexdigest()
        existing = self.repository.get_publication_by_idempotency_key(
            channel=channel,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if (
                existing.article_id != article.article_id
                or existing.article_version != article.version
                or existing.content_sha256 != content_sha256
            ):
                raise IdempotencyConflictError(
                    "idempotency key is already used for another publication"
                )
            self._record_event(
                event="publishing.article.publish_replayed",
                status="replayed",
                article=article,
                publication=existing,
            )
            if existing.status is not PublicationStatus.PUBLISHING:
                return existing
            return self._resume_existing_publication(
                article,
                existing,
                publisher,
            )

        format_report = check_article_markdown(article.markdown_content)
        if not format_report.is_publishable:
            error_code = format_report.error_code or "article_markdown_invalid"
            self._record_event(
                event="publishing.article.publish_validation_failed",
                status="failed",
                article=article,
                error_code=error_code,
                level=logging.WARNING,
            )
            raise PublicationValidationError(
                error_code,
                "article Markdown failed publication preflight",
            )

        approved_request = next(
            (
                request
                for request in self.repository.list_approval_requests(
                    article_id=article.article_id,
                    status=ApprovalStatus.APPROVED,
                )
                if request.article_version == article.version
                and request.channel is channel
                and request.content_sha256 == content_sha256
            ),
            None,
        )
        if approved_request is None:
            self._record_event(
                event="publishing.article.publish_failed",
                status="failed",
                article=article,
                error_code="publication_approval_required",
                level=logging.WARNING,
            )
            raise InvalidStateTransitionError(
                "an approved publication request is required"
            )

        approved_attachment_ids = tuple(approved_request.attachment_ids)
        if attachment_ids is not None:
            requested_attachment_ids = tuple(
                dict.fromkeys(
                    attachment_id.strip()
                    for attachment_id in attachment_ids
                    if isinstance(attachment_id, str) and attachment_id.strip()
                )
            )
            if requested_attachment_ids != approved_attachment_ids:
                self._record_event(
                    event="publishing.article.publish_validation_failed",
                    status="failed",
                    article=article,
                    error_code="publication_attachment_mismatch",
                    level=logging.WARNING,
                )
                raise PublicationValidationError(
                    "publication_attachment_mismatch",
                    "publication images do not match the approved image selection",
                )

        if channel is PublicationChannel.WECHAT_OFFICIAL_ACCOUNT:
            try:
                validate_wechat_title(article.title)
            except WeChatDraftValidationError as error:
                self._record_event(
                    event="publishing.article.publish_validation_failed",
                    status="failed",
                    article=article,
                    error_code=error.error_code,
                    level=logging.WARNING,
                )
                raise PublicationValidationError(
                    error.error_code,
                    str(error),
                ) from error
        elif channel is PublicationChannel.XIAOHONGSHU:
            format_xiaohongshu_content(article)

        try:
            if article.status in {
                ArticleStatus.APPROVED,
                ArticleStatus.FAILED,
            }:
                article.start_publishing()
                self.repository.update_article(article)
            elif article.status not in {
                ArticleStatus.PUBLISHING,
                ArticleStatus.PUBLISHED,
            }:
                raise InvalidStateTransitionError(
                    "article is not in a publishable lifecycle state"
                )
        except InvalidStateTransitionError:
            self._record_event(
                event="publishing.article.publish_failed",
                status="failed",
                article=article,
                error_code="article_not_publishable",
                level=logging.WARNING,
            )
            raise

        publication = Publication(
            publication_id=self._new_publication_id(),
            article_id=article.article_id,
            article_version=article.version,
            channel=channel,
            status=PublicationStatus.PUBLISHING,
            idempotency_key=idempotency_key,
            content_sha256=content_sha256,
            attachment_ids=approved_attachment_ids,
            attempt_count=1,
        )
        try:
            self.repository.create_publication(publication)
        except DuplicateRecordError:
            existing = self.repository.get_publication_by_idempotency_key(
                channel=channel,
                idempotency_key=idempotency_key,
            )
            if existing is not None:
                self._record_event(
                    event="publishing.article.publish_replayed",
                    status="replayed",
                    article=article,
                    publication=existing,
                )
                return existing
            raise

        self._record_event(
            event="publishing.article.publish_requested",
            status="started",
            article=article,
            publication=publication,
        )

        try:
            result = self._publish_with_context(
                publisher,
                article,
                attachment_ids=publication.attachment_ids,
            )
        except PublicationDeliveryUnknownError as error:
            if error.external_id:
                publication.external_id = error.external_id
            publication.mark_delivery_unknown(error_code=error.error_code)
            unknown = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.delivery_unknown",
                status="delivery_unknown",
                article=article,
                publication=unknown,
                error_code=unknown.error_code,
                level=logging.WARNING,
            )
            return unknown
        except PublicationExecutionError as error:
            if article.status is ArticleStatus.PUBLISHING:
                article.mark_failed()
                self.repository.update_article(article)
            publication.mark_failed(error_code=error.error_code)
            failed = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.failed",
                status="failed",
                article=article,
                publication=failed,
                error_code=failed.error_code,
                level=logging.WARNING,
            )
            return failed
        except Exception as error:
            if article.status is ArticleStatus.PUBLISHING:
                article.mark_failed()
                self.repository.update_article(article)
            error_code = getattr(error, "error_code", "publisher_failed")
            publication.mark_failed(error_code=error_code)
            failed = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.failed",
                status="failed",
                article=article,
                publication=failed,
                error_code=error_code,
                level=logging.ERROR,
            )
            return failed

        return self._apply_publication_result(
            article,
            publication,
            result,
        )

    def refresh_publication(
        self,
        publication_id: str,
        *,
        publisher: PublicationPublisher,
    ) -> Publication:
        """Refresh one in-flight publication using its stable idempotency key."""

        publication = self.repository.get_publication(publication_id)
        if publisher.channel is not publication.channel:
            raise PublisherConfigurationError("publisher channel mismatch")
        if publication.status is PublicationStatus.DELIVERY_UNKNOWN:
            raise PublicationValidationError(
                "delivery_unknown_requires_manual_confirmation",
                "an ambiguous delivery must be checked manually before retrying",
            )
        if publication.status is not PublicationStatus.PUBLISHING:
            raise InvalidStateTransitionError(
                "only an in-progress publication can be refreshed"
            )

        article = self.repository.get_article(publication.article_id)
        status_reader = getattr(publisher, "get_publication_status", None)
        if not callable(status_reader) or not publication.external_id:
            raise InvalidStateTransitionError(
                "publication does not have an external status reader"
            )

        self._record_event(
            event="publishing.publication.status_check_requested",
            status="started",
            article=article,
            publication=publication,
            summary="发布中心主动查询外部平台状态",
        )
        try:
            result = status_reader(publication.external_id)
        except PublicationPollingError as error:
            self._record_event(
                event="publishing.publication.status_check_deferred",
                status="publishing",
                article=article,
                publication=publication,
                error_code=error.error_code,
                level=logging.WARNING,
            )
            return publication
        except PublicationExecutionError as error:
            if article.status is ArticleStatus.PUBLISHING:
                article.mark_failed()
                self.repository.update_article(article)
            publication.mark_failed(error_code=error.error_code)
            failed = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.failed",
                status="failed",
                article=article,
                publication=failed,
                error_code=failed.error_code,
                level=logging.WARNING,
            )
            return failed
        except Exception:
            if article.status is ArticleStatus.PUBLISHING:
                article.mark_failed()
                self.repository.update_article(article)
            publication.mark_failed(error_code="publisher_status_failed")
            failed = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.failed",
                status="failed",
                article=article,
                publication=failed,
                error_code=failed.error_code,
                level=logging.ERROR,
            )
            return failed

        return self._apply_publication_result(
            article,
            publication,
            result,
            summary="发布中心查询到外部平台最新状态",
        )

    def retry_publication(
        self,
        publication_id: str,
        *,
        idempotency_key: str,
        publisher: PublicationPublisher,
        confirm_delivery_unknown: bool = False,
    ) -> Publication:
        """Retry a failed publication or a manually confirmed ambiguous one."""

        publication = self.repository.get_publication(publication_id)
        if publisher.channel is not publication.channel:
            raise PublisherConfigurationError("publisher channel mismatch")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise PublishingServiceError("idempotency key must not be empty")

        article = self.repository.get_article(publication.article_id)
        if publication.status is PublicationStatus.FAILED:
            return self.publish_article(
                article.article_id,
                channel=publication.channel,
                idempotency_key=idempotency_key,
                publisher=publisher,
                attachment_ids=publication.attachment_ids,
            )
        if publication.status is not PublicationStatus.DELIVERY_UNKNOWN:
            raise InvalidStateTransitionError(
                "only failed or delivery-unknown publications can be retried"
            )
        if not confirm_delivery_unknown:
            raise PublicationValidationError(
                "delivery_unknown_confirmation_required",
                "manual confirmation is required before retrying an ambiguous delivery",
            )
        if (
            article.version != publication.article_version
            or article.status
            not in {
                ArticleStatus.FAILED,
                ArticleStatus.PUBLISHING,
                ArticleStatus.PUBLISHED,
            }
        ):
            raise InvalidStateTransitionError(
                "the ambiguous publication no longer matches an in-progress article"
            )

        content_sha256 = hashlib.sha256(
            article.markdown_content.encode("utf-8")
        ).hexdigest()
        approved_request = next(
            (
                request
                for request in self.repository.list_approval_requests(
                    article_id=article.article_id,
                    status=ApprovalStatus.APPROVED,
                )
                if request.article_version == article.version
                and request.channel is publication.channel
                and request.content_sha256 == content_sha256
            ),
            None,
        )
        if approved_request is None:
            raise InvalidStateTransitionError(
                "an approved publication request is required"
            )
        format_report = check_article_markdown(article.markdown_content)
        if not format_report.is_publishable:
            raise PublicationValidationError(
                format_report.error_code or "article_markdown_invalid",
                "article Markdown failed publication preflight",
            )
        if publication.channel is PublicationChannel.WECHAT_OFFICIAL_ACCOUNT:
            try:
                validate_wechat_title(article.title)
            except WeChatDraftValidationError as error:
                raise PublicationValidationError(
                    error.error_code,
                    str(error),
                ) from error
        elif publication.channel is PublicationChannel.XIAOHONGSHU:
            format_xiaohongshu_content(article)
        elif publication.channel is PublicationChannel.DOUYIN:
            format_douyin_content(article)

        # A different channel may have failed after leaving this article in
        # the shared FAILED lifecycle state.  Re-enter the in-progress state
        # before retrying this channel's delivery-unknown publication.
        if article.status is ArticleStatus.FAILED:
            article.start_publishing()
            self.repository.update_article(article)

        publication.retry_delivery_unknown()
        self.repository.update_publication(publication)
        self._record_event(
            event="publishing.publication.retry_requested",
            status="resumed",
            article=article,
            publication=publication,
            summary="用户确认外部平台未发布后重试",
        )

        try:
            result = self._publish_with_context(
                publisher,
                article,
                attachment_ids=publication.attachment_ids,
            )
        except PublicationDeliveryUnknownError as error:
            if error.external_id:
                publication.external_id = error.external_id
            publication.mark_delivery_unknown(error_code=error.error_code)
            unknown = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.delivery_unknown",
                status="delivery_unknown",
                article=article,
                publication=unknown,
                error_code=unknown.error_code,
                level=logging.WARNING,
            )
            return unknown
        except PublicationExecutionError as error:
            if article.status is ArticleStatus.PUBLISHING:
                article.mark_failed()
                self.repository.update_article(article)
            publication.mark_failed(error_code=error.error_code)
            failed = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.failed",
                status="failed",
                article=article,
                publication=failed,
                error_code=failed.error_code,
                level=logging.WARNING,
            )
            return failed
        except Exception as error:
            if article.status is ArticleStatus.PUBLISHING:
                article.mark_failed()
                self.repository.update_article(article)
            error_code = getattr(error, "error_code", "publisher_failed")
            publication.mark_failed(error_code=error_code)
            failed = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.failed",
                status="failed",
                article=article,
                publication=failed,
                error_code=error_code,
                level=logging.ERROR,
            )
            return failed

        return self._apply_publication_result(
            article,
            publication,
            result,
            summary="人工确认后重新发布",
        )

    def _resume_existing_publication(
        self,
        article: Article,
        publication: Publication,
        publisher: PublicationPublisher,
    ) -> Publication:
        """Continue one persisted publishing attempt after interruption."""

        if article.status in {
            ArticleStatus.APPROVED,
            ArticleStatus.FAILED,
        }:
            article.start_publishing()
            self.repository.update_article(article)
        elif article.status not in {
            ArticleStatus.PUBLISHING,
            ArticleStatus.PUBLISHED,
        }:
            raise InvalidStateTransitionError(
                "article is not in a resumable publishing state"
            )

        publication.attempt_count += 1
        publication.updated_at = article.updated_at
        self.repository.update_publication(publication)
        self._record_event(
            event="publishing.article.publish_requested",
            status="resumed",
            article=article,
            publication=publication,
            summary="恢复未完成的发布任务",
        )

        status_reader = getattr(publisher, "get_publication_status", None)
        if callable(status_reader) and publication.external_id:
            self._record_event(
                event="publishing.publication.status_check_requested",
                status="started",
                article=article,
                publication=publication,
                summary="查询外部平台发布状态",
            )
            try:
                result = status_reader(publication.external_id)
            except PublicationPollingError as error:
                publication.updated_at = article.updated_at
                pending = self.repository.update_publication(publication)
                self._record_event(
                    event="publishing.publication.status_check_deferred",
                    status="publishing",
                    article=article,
                    publication=pending,
                    error_code=error.error_code,
                    level=logging.WARNING,
                )
                return pending
            except PublicationExecutionError as error:
                if article.status is ArticleStatus.PUBLISHING:
                    article.mark_failed()
                    self.repository.update_article(article)
                publication.mark_failed(error_code=error.error_code)
                failed = self.repository.update_publication(publication)
                self._record_event(
                    event="publishing.publication.failed",
                    status="failed",
                    article=article,
                    publication=failed,
                    error_code=failed.error_code,
                    level=logging.WARNING,
                )
                return failed
            except Exception:
                if article.status is ArticleStatus.PUBLISHING:
                    article.mark_failed()
                    self.repository.update_article(article)
                publication.mark_failed(error_code="publisher_status_failed")
                failed = self.repository.update_publication(publication)
                self._record_event(
                    event="publishing.publication.failed",
                    status="failed",
                    article=article,
                    publication=failed,
                    error_code="publisher_status_failed",
                    level=logging.ERROR,
                )
                return failed

            return self._apply_publication_result(
                article,
                publication,
                result,
                summary="外部平台发布状态已更新",
            )

        try:
            result = self._publish_with_context(
                publisher,
                article,
                attachment_ids=publication.attachment_ids,
            )
        except PublicationDeliveryUnknownError as error:
            if error.external_id:
                publication.external_id = error.external_id
            publication.mark_delivery_unknown(error_code=error.error_code)
            unknown = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.delivery_unknown",
                status="delivery_unknown",
                article=article,
                publication=unknown,
                error_code=unknown.error_code,
                level=logging.WARNING,
            )
            return unknown
        except PublicationExecutionError as error:
            if article.status is ArticleStatus.PUBLISHING:
                article.mark_failed()
                self.repository.update_article(article)
            publication.mark_failed(error_code=error.error_code)
            failed = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.failed",
                status="failed",
                article=article,
                publication=failed,
                error_code=failed.error_code,
                level=logging.WARNING,
            )
            return failed
        except Exception as error:
            if article.status is ArticleStatus.PUBLISHING:
                article.mark_failed()
                self.repository.update_article(article)
            error_code = getattr(error, "error_code", "publisher_failed")
            publication.mark_failed(error_code=error_code)
            failed = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.failed",
                status="failed",
                article=article,
                publication=failed,
                error_code=error_code,
                level=logging.ERROR,
            )
            return failed

        if result.status is PublicationStatus.DRAFTED:
            publication.mark_drafted(external_id=result.external_id or "")
            if article.status is ArticleStatus.PUBLISHING:
                article.return_to_approved()
                self.repository.update_article(article)
            drafted = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.drafted",
                status="drafted",
                article=article,
                publication=drafted,
                summary="微信草稿已创建",
            )
            return drafted

        if result.status is PublicationStatus.PUBLISHING:
            publication.public_url = result.public_url
            publication.external_id = result.external_id
            publication.updated_at = article.updated_at
            pending = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.accepted",
                status="publishing",
                article=article,
                publication=pending,
                summary="鎭㈠鏈畬鎴愮殑鍙戝竷浠诲姟",
            )
            return pending
        if result.status is not PublicationStatus.PUBLISHED:
            raise PublicationExecutionError("publisher_invalid_result")

        if article.status is ArticleStatus.PUBLISHING:
            article.mark_published()
            self.repository.update_article(article)
        elif article.status is not ArticleStatus.PUBLISHED:
            raise PublicationExecutionError("article_not_publishable")
        publication.mark_published(
            public_url=result.public_url,
            external_id=result.external_id,
        )
        published = self.repository.update_publication(publication)
        self._record_event(
            event="publishing.publication.succeeded",
            status="completed",
            article=article,
            publication=published,
            summary="发布任务恢复完成",
        )
        return published

    def _apply_publication_result(
        self,
        article: Article,
        publication: Publication,
        result: PublicationResult,
        *,
        summary: str | None = None,
    ) -> Publication:
        """Persist synchronous success or asynchronous acceptance safely."""

        if result.status is PublicationStatus.DRAFTED:
            publication.mark_drafted(external_id=result.external_id or "")
            if article.status is ArticleStatus.PUBLISHING:
                article.return_to_approved()
                self.repository.update_article(article)
            drafted = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.drafted",
                status="drafted",
                article=article,
                publication=drafted,
                summary=summary or "微信草稿已创建",
            )
            return drafted

        if result.status is PublicationStatus.PUBLISHING:
            publication.public_url = result.public_url
            publication.external_id = result.external_id
            publication.updated_at = article.updated_at
            pending = self.repository.update_publication(publication)
            self._record_event(
                event="publishing.publication.accepted",
                status="publishing",
                article=article,
                publication=pending,
                summary=summary,
            )
            return pending

        if result.status is not PublicationStatus.PUBLISHED:
            raise PublicationExecutionError("publisher_invalid_result")

        if article.status is ArticleStatus.PUBLISHING:
            article.mark_published()
            self.repository.update_article(article)
        elif article.status is not ArticleStatus.PUBLISHED:
            raise PublicationExecutionError("article_not_publishable")
        publication.mark_published(
            public_url=result.public_url,
            external_id=result.external_id,
        )
        published = self.repository.update_publication(publication)
        self._record_event(
            event="publishing.publication.succeeded",
            status="completed",
            article=article,
            publication=published,
            summary=summary,
        )
        return published

    def list_publications(self, article_id: str) -> list[Publication]:
        """List all publication attempts for an Article."""

        return self.repository.list_publications(article_id)

    def list_recoverable_publications(
        self,
        *,
        channel: PublicationChannel,
    ) -> list[Publication]:
        """List in-flight tasks for one channel that can be status-polled."""

        return self.repository.list_incomplete_publications(
            channel=channel,
            status=PublicationStatus.PUBLISHING,
        )

    def list_recoverable_wechat_publications(self) -> list[Publication]:
        """Backward-compatible alias for older lifecycle callers."""

        return self.list_recoverable_publications(
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )

    @staticmethod
    def _publish_with_context(
        publisher: PublicationPublisher,
        article: Article,
        *,
        attachment_ids: tuple[str, ...],
    ) -> PublicationResult:
        """Pass media context to adapters while keeping old adapters compatible."""

        try:
            parameters = inspect.signature(publisher.publish).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_context = (
            "attachment_ids" in parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        if accepts_context:
            return publisher.publish(
                article,
                attachment_ids=attachment_ids,
            )
        return publisher.publish(article)

    @staticmethod
    def _new_publication_id() -> str:
        return uuid4().hex


__all__ = ["PublicationWorkflowMixin"]
