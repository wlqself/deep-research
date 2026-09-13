"""Logging, audit, and progress-event support for PublishingService."""

from __future__ import annotations

import logging

from ..log.log_audit import audit_event
from ..log.logging_utils import log_event
from .events import (
    PublishingEvent,
    PublishingEventStatus,
    PublishingEventType,
)
from .models import Article, Publication, utc_now


logger = logging.getLogger("deep_research.publishing")

_PROGRESS_EVENT_DEFINITIONS: dict[
    str,
    tuple[PublishingEventType, PublishingEventStatus],
] = {
    "publishing.article.imported": (
        PublishingEventType.ARTICLE_CREATED,
        PublishingEventStatus.SUCCEEDED,
    ),
    "publishing.article.edited": (
        PublishingEventType.ARTICLE_UPDATED,
        PublishingEventStatus.SUCCEEDED,
    ),
    "publishing.article.approved": (
        PublishingEventType.ARTICLE_APPROVED,
        PublishingEventStatus.SUCCEEDED,
    ),
    "publishing.approval.requested": (
        PublishingEventType.APPROVAL_REQUESTED,
        PublishingEventStatus.WAITING_APPROVAL,
    ),
    "publishing.article.publish_requested": (
        PublishingEventType.PUBLISHING_STARTED,
        PublishingEventStatus.IN_PROGRESS,
    ),
    "publishing.publication.succeeded": (
        PublishingEventType.PUBLISHED,
        PublishingEventStatus.SUCCEEDED,
    ),
    "publishing.publication.failed": (
        PublishingEventType.PUBLISHING_FAILED,
        PublishingEventStatus.FAILED,
    ),
}


class PublishingEventMixin:
    """Provide one shared event path for all publishing use cases."""

    def _record_event(
        self,
        *,
        event: str,
        status: str,
        article: Article | None = None,
        publication: Publication | None = None,
        thread_id: str | None = None,
        artifact_id: str | None = None,
        error_code: str | None = None,
        actor: str = "application",
        summary: str = "",
        level: int = logging.INFO,
    ) -> None:
        effective_thread_id = (
            article.source_thread_id if article is not None else thread_id
        )
        effective_artifact_id = (
            article.source_artifact_id if article is not None else artifact_id
        )
        effective_article_id = (
            article.article_id if article is not None else None
        )
        effective_publication_id = (
            publication.publication_id if publication is not None else None
        )
        effective_channel = (
            publication.channel if publication is not None else None
        )

        log_event(
            logger,
            level,
            event,
            thread_id=effective_thread_id,
            artifact_id=effective_artifact_id,
            status=status,
            error_code=error_code,
        )
        audit_event(
            event,
            status,
            thread_id=effective_thread_id,
            artifact_id=effective_artifact_id,
        )
        try:
            self.repository.record_audit_event(
                action=event,
                status=status,
                actor=actor,
                article_id=effective_article_id,
                publication_id=effective_publication_id,
                channel=effective_channel,
                error_code=error_code,
                occurred_at=utc_now(),
            )
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "publishing.audit.failed",
                thread_id=effective_thread_id,
                artifact_id=effective_artifact_id,
                status="failed",
                error_code="audit_persist_failed",
            )

        progress_definition = _PROGRESS_EVENT_DEFINITIONS.get(event)
        if progress_definition is None:
            return

        progress_event_type, progress_status = progress_definition
        try:
            self.event_sink(
                PublishingEvent.create(
                    event_type=progress_event_type,
                    status=progress_status,
                    article_id=effective_article_id,
                    article_version=(
                        article.version if article is not None else None
                    ),
                    publication_id=effective_publication_id,
                    channel=effective_channel,
                    error_code=(
                        error_code
                        if progress_status is PublishingEventStatus.FAILED
                        else None
                    ),
                    summary=summary,
                )
            )
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "publishing.progress_event.failed",
                thread_id=effective_thread_id,
                artifact_id=effective_artifact_id,
                status="failed",
                error_code="progress_event_failed",
            )


__all__ = ["PublishingEventMixin"]
