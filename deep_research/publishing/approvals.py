"""Domain model for durable human approval requests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from .models import (
    InvalidDomainValueError,
    InvalidStateTransitionError,
    PublicationChannel,
    PublishingDomainError,
    utc_now,
)


class ApprovalRequestError(PublishingDomainError):
    """Base error for deterministic approval-request operations."""


class ApprovalAction(StrEnum):
    """Actions that may require an explicit human decision."""

    PUBLISH = "publish"


class ApprovalStatus(StrEnum):
    """Lifecycle of a human approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidDomainValueError(f"{field_name} must be a string")

    normalized = value.strip()
    if not normalized:
        raise InvalidDomainValueError(f"{field_name} must not be empty")

    return normalized


def _validate_sha256(value: str, field_name: str) -> str:
    normalized = _require_text(value, field_name).lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise InvalidDomainValueError(
            f"{field_name} must be a lowercase hexadecimal SHA-256 digest"
        )

    return normalized


@dataclass
class ApprovalRequest:
    """A version-pinned request for a human decision before publication.

    The request stores only stable identifiers and a content digest. The
    Article remains the source of the content; the digest and version prevent
    approval from silently applying to a later edit.
    """

    approval_id: str
    action: ApprovalAction
    article_id: str
    article_version: int
    channel: PublicationChannel
    content_sha256: str
    # Immutable image selection captured with the approval.  Publication
    # execution must use this snapshot instead of a later mutable UI choice.
    attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    decided_at: datetime | None = None
    decision_actor: str | None = None
    decision_reason: str | None = None

    def __post_init__(self) -> None:
        self.approval_id = _require_text(
            self.approval_id,
            "approval_id",
        )
        self.article_id = _require_text(
            self.article_id,
            "article_id",
        )
        self.content_sha256 = _validate_sha256(
            self.content_sha256,
            "content_sha256",
        )
        self.attachment_ids = tuple(
            dict.fromkeys(
                attachment_id.strip()
                for attachment_id in self.attachment_ids
                if isinstance(attachment_id, str) and attachment_id.strip()
            )
        )

        if not isinstance(self.article_version, int) or self.article_version < 1:
            raise InvalidDomainValueError(
                "article_version must be a positive integer"
            )

        if not isinstance(self.action, ApprovalAction):
            self.action = ApprovalAction(self.action)

        if not isinstance(self.channel, PublicationChannel):
            self.channel = PublicationChannel(self.channel)

        if not isinstance(self.status, ApprovalStatus):
            self.status = ApprovalStatus(self.status)

        if self.decision_actor is not None:
            self.decision_actor = _require_text(
                self.decision_actor,
                "decision_actor",
            )

        if self.decision_reason is not None:
            self.decision_reason = _require_text(
                self.decision_reason,
                "decision_reason",
            )

    @property
    def is_pending(self) -> bool:
        return self.status is ApprovalStatus.PENDING

    def approve(
        self,
        *,
        decision_actor: str,
        decision_reason: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Approve the exact Article version and channel in this request."""

        self._require_pending()
        timestamp = now or utc_now()
        self.status = ApprovalStatus.APPROVED
        self.decided_at = timestamp
        self.decision_actor = _require_text(
            decision_actor,
            "decision_actor",
        )
        self.decision_reason = (
            _require_text(decision_reason, "decision_reason")
            if decision_reason is not None
            else None
        )
        self.updated_at = timestamp

    def reject(
        self,
        *,
        decision_actor: str,
        decision_reason: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Reject the request and require a new request for a later attempt."""

        self._require_pending()
        timestamp = now or utc_now()
        self.status = ApprovalStatus.REJECTED
        self.decided_at = timestamp
        self.decision_actor = _require_text(
            decision_actor,
            "decision_actor",
        )
        self.decision_reason = (
            _require_text(decision_reason, "decision_reason")
            if decision_reason is not None
            else None
        )
        self.updated_at = timestamp

    def expire(
        self,
        *,
        now: datetime | None = None,
    ) -> None:
        """Expire a request that is no longer eligible for a decision."""

        self._require_pending()
        timestamp = now or utc_now()
        self.status = ApprovalStatus.EXPIRED
        self.decided_at = timestamp
        self.updated_at = timestamp

    def _require_pending(self) -> None:
        if not self.is_pending:
            raise InvalidStateTransitionError(
                "only pending approval requests can be decided"
            )


__all__ = [
    "ApprovalAction",
    "ApprovalRequest",
    "ApprovalRequestError",
    "ApprovalStatus",
]
