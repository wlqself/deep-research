"""Validated conversational intents for the publishing workflow.

This module deliberately contains no database access and no state-changing
operation.  An intent is only a candidate interpretation of a user message;
the application layer must resolve it against the current thread and durable
publishing records before doing anything with side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import PublicationChannel, PublishingDomainError


class PublishingIntentError(PublishingDomainError):
    """Raised when a conversational publishing intent is malformed."""


class PublishingIntentAction(StrEnum):
    """Non-executing actions understood by the publishing conversation layer."""

    NONE = "none"
    STATUS = "status"
    APPROVE = "approve"
    REJECT = "reject"
    RESUME = "resume"
    REQUEST_PUBLICATION = "request_publication"


@dataclass(frozen=True, slots=True)
class PublishingIntent:
    """An untrusted, side-effect-free interpretation of user language.

    ``target_hint`` is descriptive text such as an article title.  It is not
    an Article ID, Approval ID, or filesystem path and must never be used as a
    direct database lookup key.  The resolver must map it to a unique record
    in the current thread before executing any command.
    """

    action: PublishingIntentAction
    target_hint: str | None = None
    channel: PublicationChannel | None = None
    decision_reason: str | None = None

    def __post_init__(self) -> None:
        action = self.action
        if not isinstance(action, PublishingIntentAction):
            try:
                action = PublishingIntentAction(action)
            except (TypeError, ValueError) as error:
                raise PublishingIntentError(
                    "invalid publishing intent action"
                ) from error
            object.__setattr__(self, "action", action)

        if self.target_hint is not None:
            self._validate_optional_text(
                self.target_hint,
                "target_hint",
                max_length=240,
            )
        if self.channel is not None and not isinstance(
            self.channel,
            PublicationChannel,
        ):
            try:
                object.__setattr__(
                    self,
                    "channel",
                    PublicationChannel(self.channel),
                )
            except (TypeError, ValueError) as error:
                raise PublishingIntentError(
                    "invalid publishing intent channel"
                ) from error
        if self.decision_reason is not None:
            self._validate_optional_text(
                self.decision_reason,
                "decision_reason",
                max_length=500,
            )

        if (
            action is PublishingIntentAction.REQUEST_PUBLICATION
            and self.channel is None
        ):
            raise PublishingIntentError(
                "publication request requires a channel"
            )

        if action in {
            PublishingIntentAction.NONE,
            PublishingIntentAction.STATUS,
        } and self.decision_reason is not None:
            raise PublishingIntentError(
                "status intents cannot contain a decision reason"
            )

    @property
    def is_read_only(self) -> bool:
        """Whether resolving this intent can never change publishing state."""

        return self.action in {
            PublishingIntentAction.NONE,
            PublishingIntentAction.STATUS,
        }

    @property
    def requires_durable_command(self) -> bool:
        """Whether the intent needs application-layer command validation."""

        return not self.is_read_only

    @staticmethod
    def _validate_optional_text(
        value: str,
        field_name: str,
        *,
        max_length: int,
    ) -> None:
        if not isinstance(value, str):
            raise PublishingIntentError(
                f"{field_name} must be a string"
            )
        if not value.strip():
            raise PublishingIntentError(
                f"{field_name} must not be empty"
            )
        if len(value.strip()) > max_length:
            raise PublishingIntentError(
                f"{field_name} is too long"
            )


__all__ = [
    "PublishingIntent",
    "PublishingIntentAction",
    "PublishingIntentError",
]
