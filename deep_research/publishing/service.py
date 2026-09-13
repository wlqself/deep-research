"""Unified application service for the deterministic publishing workflow.

The implementation is split by use case into Article, approval, publication,
and event-support mixins.  This module remains the stable import boundary for
the rest of the application.
"""

from __future__ import annotations

from typing import Callable

from .artifacts import ArtifactService
from .events import PublishingEvent
from .service_approvals import ApprovalWorkflowMixin
from .service_articles import ArticleWorkflowMixin
from .service_events import PublishingEventMixin
from .service_publications import PublicationWorkflowMixin
from .service_support import (
    ArticleImportConflictError,
    InvalidSlugError,
    PublicationDeliveryUnknownError,
    PublicationExecutionError,
    PublicationPollingError,
    PublicationValidationError,
    PublicationPublisher,
    PublicationResult,
    PublisherConfigurationError,
    PublishingServiceError,
    excerpt_from_markdown,
    normalized_tags,
    slugify_title,
    title_from_filename,
    validate_slug,
)


class PublishingService(
    PublishingEventMixin,
    ArticleWorkflowMixin,
    ApprovalWorkflowMixin,
    PublicationWorkflowMixin,
):
    """Coordinate publishing operations without exposing persistence details."""

    def __init__(
        self,
        repository,
        artifact_service: ArtifactService,
        event_sink: Callable[[PublishingEvent], None] | None = None,
    ) -> None:
        self.repository = repository
        self.artifact_service = artifact_service
        self.event_sink = event_sink or (lambda event: None)


__all__ = [
    "ArticleImportConflictError",
    "InvalidSlugError",
    "PublicationExecutionError",
    "PublicationDeliveryUnknownError",
    "PublicationPollingError",
    "PublicationValidationError",
    "PublicationPublisher",
    "PublicationResult",
    "PublisherConfigurationError",
    "PublishingService",
    "PublishingServiceError",
    "excerpt_from_markdown",
    "normalized_tags",
    "slugify_title",
    "title_from_filename",
    "validate_slug",
]
