"""Domain model for durable human-in-the-loop interactions."""
from __future__ import annotations
"""
HITLInteraction
HITLAction
HITLStatus
HITL 领域异常
状态转换方法：- pending -> approved
- pending -> rejected
- pending -> expired
- approved -> resuming
- failed -> resuming
- resuming -> resumed
- resuming -> failed
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum


class HITLDomainError(Exception):
    """Base exception for HITL domain failures."""


class InvalidHITLValueError(HITLDomainError):
    """Raised when an interaction contains an invalid domain value."""


class InvalidHITLTransitionError(HITLDomainError):
    """Raised when an interaction makes an illegal state transition."""


class HITLAction(StrEnum):
    """Human decisions supported by the first durable HITL model."""

    APPROVE = "approve"
    REJECT = "reject"
    RESUME = "resume"


class HITLStatus(StrEnum):
    """Lifecycle of a persisted interaction with a human."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    RESUMING = "resuming"
    RESUMED = "resumed"
    FAILED = "failed"
    STALE = "stale"
    EXPIRED = "expired"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for HITL records."""

    return datetime.now(timezone.utc)


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidHITLValueError(f"{field_name} must be a string")

    normalized = value.strip()
    if not normalized:
        raise InvalidHITLValueError(f"{field_name} must not be empty")

    return normalized


@dataclass
class HITLInteraction:
    """A durable pause point for one Agent run and one exact target.

    This model records the interaction boundary only. It does not execute an
    approval, resume an Agent run, or call a publishing connector.
    """

    interaction_id: str
    thread_id: str
    run_id: str
    action: HITLAction
    target_type: str
    target_id: str
    target_version: int
    status: HITLStatus = HITLStatus.PENDING
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    resolved_at: datetime | None = None
    decision_actor: str | None = None
    decision_reason: str | None = None
    error_code: str | None = None
    interrupt_id: str | None = None
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "interaction_id",
            "thread_id",
            "run_id",
            "target_type",
            "target_id",
        ):
            setattr(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )

        if not isinstance(self.target_version, int) or self.target_version < 1:
            raise InvalidHITLValueError(
                "target_version must be a positive integer"
            )

        if not isinstance(self.action, HITLAction):
            self.action = HITLAction(self.action)
        if not isinstance(self.status, HITLStatus):
            self.status = HITLStatus(self.status)

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
        if self.error_code is not None:
            self.error_code = _require_text(self.error_code, "error_code")
        if self.interrupt_id is not None:
            self.interrupt_id = _require_text(
                self.interrupt_id,
                "interrupt_id",
            )
        if self.checkpoint_id is not None:
            self.checkpoint_id = _require_text(
                self.checkpoint_id,
                "checkpoint_id",
            )

    @property
    def is_pending(self) -> bool:
        return self.status is HITLStatus.PENDING

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= utc_now()

    @property
    def can_resume(self) -> bool:
        return self.status in {
            HITLStatus.APPROVED,
            # A rejection/request-for-changes is still a decision on the
            # native graph interrupt. Resume lets the Agent acknowledge it.
            HITLStatus.REJECTED,
            HITLStatus.FAILED,
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            HITLStatus.RESUMED,
            HITLStatus.STALE,
            HITLStatus.EXPIRED,
        }

    def approve(
        self,
        *,
        decision_actor: str,
        decision_reason: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Record an explicit human approval without resuming execution."""

        self._require_status(HITLStatus.PENDING)
        timestamp = now or utc_now()
        self.status = HITLStatus.APPROVED
        self.resolved_at = timestamp
        self.updated_at = timestamp
        self.decision_actor = _require_text(
            decision_actor,
            "decision_actor",
        )
        self.decision_reason = (
            _require_text(decision_reason, "decision_reason")
            if decision_reason is not None
            else None
        )

    def reject(
        self,
        *,
        decision_actor: str,
        decision_reason: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Record an explicit human rejection."""

        self._require_status(HITLStatus.PENDING)
        timestamp = now or utc_now()
        self.status = HITLStatus.REJECTED
        self.resolved_at = timestamp
        self.updated_at = timestamp
        self.decision_actor = _require_text(
            decision_actor,
            "decision_actor",
        )
        self.decision_reason = (
            _require_text(decision_reason, "decision_reason")
            if decision_reason is not None
            else None
        )

    def start_resume(self, *, now: datetime | None = None) -> None:
        """Mark a resolved interaction as being resumed."""

        if not self.can_resume:
            raise InvalidHITLTransitionError(
                "only approved, rejected, or failed interactions can resume"
            )

        self.status = HITLStatus.RESUMING
        self.updated_at = now or utc_now()
        self.error_code = None

    def mark_resumed(self, *, now: datetime | None = None) -> None:
        """Mark the interrupted run as successfully resumed."""

        self._require_status(HITLStatus.RESUMING)
        timestamp = now or utc_now()
        self.status = HITLStatus.RESUMED
        self.updated_at = timestamp

    def mark_failed(
        self,
        *,
        error_code: str,
        now: datetime | None = None,
    ) -> None:
        """Persist a safe failure code while keeping retry possible."""

        self._require_status(HITLStatus.RESUMING)
        timestamp = now or utc_now()
        self.status = HITLStatus.FAILED
        self.updated_at = timestamp
        self.error_code = _require_text(error_code, "error_code")

    def mark_stale(
        self,
        *,
        error_code: str = "hitl_interrupt_missing",
        now: datetime | None = None,
    ) -> None:
        """Mark a resolved interaction whose graph interrupt no longer exists."""

        if self.status not in {
            HITLStatus.APPROVED,
            HITLStatus.REJECTED,
            HITLStatus.RESUMING,
            HITLStatus.FAILED,
        }:
            raise InvalidHITLTransitionError(
                "only resolved interactions can become stale"
            )
        timestamp = now or utc_now()
        self.status = HITLStatus.STALE
        self.updated_at = timestamp
        self.error_code = _require_text(error_code, "error_code")

    def expire(self, *, now: datetime | None = None) -> None:
        """Expire an unanswered interaction without executing its action."""

        self._require_status(HITLStatus.PENDING)
        timestamp = now or utc_now()
        self.status = HITLStatus.EXPIRED
        self.resolved_at = timestamp
        self.updated_at = timestamp

    def _require_status(self, expected: HITLStatus) -> None:
        if self.status is not expected:
            raise InvalidHITLTransitionError(
                f"interaction must be {expected.value}, "
                f"not {self.status.value}"
            )


__all__ = [
    "HITLAction",
    "HITLDomainError",
    "HITLInteraction",
    "HITLStatus",
    "InvalidHITLTransitionError",
    "InvalidHITLValueError",
    "utc_now",
]
