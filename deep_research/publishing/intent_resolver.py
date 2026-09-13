"""Deterministic resolution of conversational publishing intents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from .approvals import ApprovalRequest, ApprovalStatus
from .intents import (
    PublishingIntent,
    PublishingIntentAction,
    PublishingIntentError,
)
from .models import (
    Article,
    ArticleStatus,
    PublicationChannel,
    PublicationStatus,
)
from ..hitl.publication_state import resolve_publication_hitl_state


class PublishingIntentResolutionError(PublishingIntentError):
    """Raised when a mutating intent cannot be safely resolved."""


class PublishingIntentResolutionStatus(StrEnum):
    """Outcome of resolving an intent against one conversation thread."""

    READ_ONLY = "read_only"
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class PublishingApprovalTarget:
    """Safe target metadata selected from durable records."""

    approval_id: str | None
    article_id: str
    article_title: str
    article_slug: str
    article_version: int
    channel: PublicationChannel
    approval_status: ApprovalStatus | None
    publication_status: PublicationStatus | None = None
    publication_error_code: str | None = None
    publication_id: str | None = None
    attachment_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PublishingIntentResolution:
    """A side-effect-free resolution result for the application boundary."""

    intent: PublishingIntent
    status: PublishingIntentResolutionStatus
    target: PublishingApprovalTarget | None = None
    candidates: tuple[PublishingApprovalTarget, ...] = ()

    @property
    def is_resolved(self) -> bool:
        return self.status is PublishingIntentResolutionStatus.RESOLVED


class PublishingIntentResolver:
    """Resolve intents using only the current thread's persisted records."""

    def __init__(self, publishing_service, hitl_service=None) -> None:
        self.publishing_service = publishing_service
        self.hitl_service = hitl_service

    def resolve(
        self,
        intent: PublishingIntent,
        *,
        thread_id: str,
    ) -> PublishingIntentResolution:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise PublishingIntentResolutionError(
                "thread_id must not be empty"
            )
        if not isinstance(intent, PublishingIntent):
            raise PublishingIntentResolutionError(
                "intent must be a PublishingIntent"
            )

        if intent.action in {
            PublishingIntentAction.NONE,
            PublishingIntentAction.STATUS,
        }:
            return PublishingIntentResolution(
                intent=intent,
                status=PublishingIntentResolutionStatus.READ_ONLY,
                candidates=tuple(self._targets_for_thread(thread_id)),
            )

        if intent.action is PublishingIntentAction.REQUEST_PUBLICATION:
            candidates = self._article_targets_for_thread(
                thread_id,
                channel=intent.channel,
            )
        else:
            candidates = [
                target
                for target in self._targets_for_thread(thread_id)
                if self._action_can_use_target(intent.action, target)
            ]
        candidates = self._filter_by_target_hint(
            candidates,
            intent.target_hint,
        )
        candidates = self._filter_by_channel(
            candidates,
            intent.channel,
        )
        if len(candidates) == 1:
            return PublishingIntentResolution(
                intent=intent,
                status=PublishingIntentResolutionStatus.RESOLVED,
                target=candidates[0],
                candidates=tuple(candidates),
            )
        if not candidates:
            return PublishingIntentResolution(
                intent=intent,
                status=PublishingIntentResolutionStatus.NOT_FOUND,
            )
        return PublishingIntentResolution(
            intent=intent,
            status=PublishingIntentResolutionStatus.AMBIGUOUS,
            candidates=tuple(candidates),
        )

    def _targets_for_thread(
        self,
        thread_id: str,
    ) -> list[PublishingApprovalTarget]:
        approvals = self.publishing_service.list_publication_approvals_for_thread(
            thread_id,
            include_resolved=False,
        )
        targets: list[PublishingApprovalTarget] = []
        for approval in approvals:
            article = self.publishing_service.get_article(approval.article_id)
            if not self._matches_current_article(approval, article):
                continue

            publication_status = self._publication_status(
                article,
                approval.channel,
            )
            if publication_status is PublicationStatus.PUBLISHED:
                continue
            publication = self._publication_for_article(
                article,
                approval.channel,
            )
            targets.append(
                PublishingApprovalTarget(
                    approval_id=approval.approval_id,
                    article_id=approval.article_id,
                    article_title=article.title,
                    article_slug=article.slug,
                    article_version=approval.article_version,
                    channel=approval.channel,
                    approval_status=ApprovalStatus(
                        resolve_publication_hitl_state(
                            approval,
                            self.hitl_service,
                        ).approval_status
                    ),
                    publication_status=publication_status,
                    publication_error_code=(
                        publication.error_code
                        if publication is not None
                        else None
                    ),
                    publication_id=(
                        publication.publication_id
                        if publication is not None
                        else None
                    ),
                    attachment_ids=approval.attachment_ids,
                )
            )
        return targets

    def _article_targets_for_thread(
        self,
        thread_id: str,
        *,
        channel: PublicationChannel | None,
    ) -> list[PublishingApprovalTarget]:
        if channel is None:
            return []

        targets: list[PublishingApprovalTarget] = []
        for article in self.publishing_service.list_articles():
            if (
                article.source_thread_id != thread_id
                or not article.can_request_publication_approval()
            ):
                continue

            publication_status = self._publication_status(article, channel)
            if publication_status is PublicationStatus.PUBLISHED:
                continue
            publication = self._publication_for_article(article, channel)

            targets.append(
                PublishingApprovalTarget(
                    approval_id=None,
                    article_id=article.article_id,
                    article_title=article.title,
                    article_slug=article.slug,
                    article_version=article.version,
                    channel=channel,
                    approval_status=None,
                    publication_status=publication_status,
                    publication_error_code=None,
                    publication_id=(
                        publication.publication_id
                        if publication is not None
                        else None
                    ),
                )
            )
        return targets

    @staticmethod
    def _action_can_use_target(
        action: PublishingIntentAction,
        target: PublishingApprovalTarget,
    ) -> bool:
        if action in {
            PublishingIntentAction.APPROVE,
            PublishingIntentAction.REJECT,
        }:
            return target.approval_status is ApprovalStatus.PENDING
        if action is PublishingIntentAction.RESUME:
            return (
                target.approval_status is ApprovalStatus.APPROVED
                and target.publication_status
                in {
                    None,
                    PublicationStatus.PUBLISHING,
                    PublicationStatus.FAILED,
                    PublicationStatus.DELIVERY_UNKNOWN,
                }
            )
        return False

    @staticmethod
    def _filter_by_target_hint(
        candidates: list[PublishingApprovalTarget],
        target_hint: str | None,
    ) -> list[PublishingApprovalTarget]:
        if target_hint is None:
            return candidates

        normalized_hint = target_hint.strip().casefold()
        if not normalized_hint:
            return candidates
        return [
            candidate
            for candidate in candidates
            if normalized_hint in candidate.article_title.casefold()
            or normalized_hint in candidate.article_slug.casefold()
        ]

    @staticmethod
    def _filter_by_channel(
        candidates: list[PublishingApprovalTarget],
        channel: PublicationChannel | None,
    ) -> list[PublishingApprovalTarget]:
        if channel is None:
            return candidates
        return [candidate for candidate in candidates if candidate.channel is channel]

    def _publication_status(
        self,
        article: Article,
        channel: PublicationChannel,
    ) -> PublicationStatus | None:
        publication = self._publication_for_article(article, channel)
        return publication.status if publication is not None else None

    def _publication_for_article(
        self,
        article: Article,
        channel: PublicationChannel,
    ):
        return next(
            (
                publication
                for publication in self.publishing_service.list_publications(
                    article.article_id,
                )
                if publication.article_version == article.version
                and publication.channel is channel
            ),
            None,
        )

    @staticmethod
    def _matches_current_article(
        approval: ApprovalRequest,
        article: Article,
    ) -> bool:
        if article.version != approval.article_version:
            return False
        content_sha256 = hashlib.sha256(
            article.markdown_content.encode("utf-8")
        ).hexdigest()
        return content_sha256 == approval.content_sha256


__all__ = [
    "PublishingApprovalTarget",
    "PublishingIntentResolution",
    "PublishingIntentResolutionError",
    "PublishingIntentResolutionStatus",
    "PublishingIntentResolver",
]
