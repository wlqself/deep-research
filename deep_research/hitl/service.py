"""Application service for durable human-in-the-loop interactions."""

from __future__ import annotations

from typing import Iterable
from datetime import timedelta
from uuid import uuid4

from .models import (
    HITLAction,
    HITLInteraction,
    HITLStatus,
    utc_now,
)
from .repository import (
    ActiveThreadInteractionError,
    DuplicateInteractionError,
    HITLRepository,
    InteractionNotFoundError,
)


class HITLServiceError(Exception):
    """Base exception for HITL application-service failures."""


class HITLThreadBusyError(HITLServiceError):
    """A new HITL cannot be created before the current one is completed."""

    def __init__(self, interaction: HITLInteraction) -> None:
        self.interaction = interaction
        super().__init__(
            "the current HITL interaction must be completed before creating "
            f"another one: {interaction.interaction_id}"
        )


class HITLService:
    """Coordinate durable interaction state without executing Agent work."""

    def __init__(self, repository: HITLRepository, *, expiration_seconds: int = 86_400) -> None:
        self.repository = repository
        self.expiration_seconds = max(60, int(expiration_seconds))

    def create_or_get_interaction(
        self,
        *,
        thread_id: str,
        run_id: str,
        action: HITLAction,
        target_type: str,
        target_id: str,
        target_version: int,
        interaction_id: str | None = None,
    ) -> HITLInteraction:
        """Create one durable pause point or return its existing record.

        The request key is derived from the Agent run and exact target. This
        makes repeated stream delivery or client retries return the same
        interaction instead of creating another approval task.
        """

        if not isinstance(action, HITLAction):
            action = HITLAction(action)

        # A new Agent run may receive a new tool_call_id while it is still
        # talking about the same approval target.  Do not create another
        # visible approval task just because the run id changed.
        active_for_target = self.repository.list_for_target(
            target_type,
            target_id,
            target_version=target_version,
            statuses=(
                HITLStatus.PENDING,
                HITLStatus.APPROVED,
                HITLStatus.REJECTED,
                HITLStatus.RESUMING,
                HITLStatus.FAILED,
            ),
            limit=100,
        )
        existing = next(
            (
                item
                for item in active_for_target
                if item.thread_id == thread_id and item.action is action
            ),
            None,
        )
        if existing is not None:
            return existing

        existing = self.repository.find_by_request_key(
            run_id=run_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            target_version=target_version,
        )
        if existing is not None:
            return existing

        blocking = self._blocking_interaction(thread_id)
        if blocking is not None:
            raise HITLThreadBusyError(blocking)

        interaction = HITLInteraction(
            interaction_id=interaction_id or uuid4().hex,
            thread_id=thread_id,
            run_id=run_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            target_version=target_version,
            expires_at=utc_now() + timedelta(seconds=self.expiration_seconds),
        )
        try:
            return self.repository.create(interaction)
        except ActiveThreadInteractionError as error:
            # The repository check is repeated inside BEGIN IMMEDIATE so two
            # concurrent tool calls cannot both create a new HITL. Preserve
            # idempotency when the winner created the exact same target.
            blocking = error.interaction
            if (
                blocking.thread_id == thread_id
                and blocking.action is action
                and blocking.target_type == target_type
                and blocking.target_id == target_id
                and blocking.target_version == target_version
            ):
                return blocking
            raise HITLThreadBusyError(blocking) from error
        except DuplicateInteractionError:
            existing = self.repository.find_by_request_key(
                run_id=run_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                target_version=target_version,
            )
            if existing is not None:
                return existing
            raise HITLServiceError(
                "HITL interaction could not be created"
            )

    def _blocking_interaction(self, thread_id: str) -> HITLInteraction | None:
        """Return the newest unresolved interaction for a thread.

        Resolved records remain blocking until the graph resume has completed;
        otherwise a later tool could create a second interrupt while the first
        one is still waiting to be resumed.
        """

        statuses = tuple(
            status
            for status in HITLStatus
            if status
            not in {
                HITLStatus.RESUMED,
                HITLStatus.STALE,
                HITLStatus.EXPIRED,
            }
        )
        interactions = self.repository.list_for_thread(
            thread_id,
            statuses=statuses,
            limit=100,
        )
        for interaction in interactions:
            if interaction.status is HITLStatus.PENDING and interaction.is_expired:
                try:
                    self.expire(interaction.interaction_id)
                except Exception:
                    return interaction
                continue
            return interaction
        return None

    def get_interaction(self, interaction_id: str) -> HITLInteraction:
        return self.repository.get(interaction_id)

    def list_for_thread(
        self,
        thread_id: str,
        *,
        statuses: Iterable[HITLStatus] | None = None,
        limit: int = 100,
    ) -> list[HITLInteraction]:
        return self.repository.list_for_thread(
            thread_id,
            statuses=statuses,
            limit=limit,
        )

    def list_recoverable_for_thread(
        self,
        thread_id: str,
        *,
        limit: int = 100,
    ) -> list[HITLInteraction]:
        """Return durable interactions relevant to a later user turn."""

        interactions = self.list_for_thread(
            thread_id,
            statuses=(
                HITLStatus.PENDING,
                HITLStatus.APPROVED,
                HITLStatus.REJECTED,
                HITLStatus.RESUMING,
                HITLStatus.FAILED,
            ),
            limit=limit,
        )
        for interaction in interactions:
            if interaction.is_pending and interaction.is_expired:
                try:
                    self.expire(interaction.interaction_id)
                except Exception:
                    continue
        return [
            interaction
            for interaction in interactions
            if interaction.status is not HITLStatus.EXPIRED
        ]

    def list_for_target(
        self,
        target_type: str,
        target_id: str,
        *,
        target_version: int | None = None,
        statuses: Iterable[HITLStatus] | None = None,
        limit: int = 100,
    ) -> list[HITLInteraction]:
        """List interactions for a target addressed outside its thread."""

        return self.repository.list_for_target(
            target_type,
            target_id,
            target_version=target_version,
            statuses=statuses,
            limit=limit,
        )

    def approve(
        self,
        interaction_id: str,
        *,
        decision_actor: str,
        decision_reason: str | None = None,
    ) -> HITLInteraction:
        interaction = self._get(interaction_id)
        interaction.approve(
            decision_actor=decision_actor,
            decision_reason=decision_reason,
        )
        return self.repository.update(interaction)

    def reject(
        self,
        interaction_id: str,
        *,
        decision_actor: str,
        decision_reason: str | None = None,
    ) -> HITLInteraction:
        interaction = self._get(interaction_id)
        interaction.reject(
            decision_actor=decision_actor,
            decision_reason=decision_reason,
        )
        return self.repository.update(interaction)

    def start_resume(self, interaction_id: str) -> HITLInteraction:
        interaction, claimed = self.repository.claim_resume(interaction_id)
        if claimed or interaction.status is HITLStatus.RESUMING:
            return interaction
        # claim_resume returns the current record when another request won
        # the race or when the state is not resumable. Preserve the domain
        # transition error for the latter.
        interaction.start_resume()
        return self.repository.update(interaction)

    def claim_resume(
        self,
        interaction_id: str,
    ) -> tuple[HITLInteraction, bool]:
        """Return the interaction and whether this request won the claim."""

        return self.repository.claim_resume(interaction_id)

    def mark_resumed(self, interaction_id: str) -> HITLInteraction:
        interaction = self._get(interaction_id)
        interaction.mark_resumed()
        return self.repository.update(interaction)

    def mark_failed(
        self,
        interaction_id: str,
        *,
        error_code: str,
    ) -> HITLInteraction:
        interaction = self._get(interaction_id)
        interaction.mark_failed(error_code=error_code)
        return self.repository.update(interaction)

    def bind_runtime(
        self,
        interaction_id: str,
        *,
        interrupt_id: str,
        checkpoint_id: str | None = None,
    ) -> HITLInteraction:
        """Persist the exact LangGraph pause represented by an interaction."""

        interaction = self._get(interaction_id)
        interaction.interrupt_id = interrupt_id
        interaction.checkpoint_id = checkpoint_id
        return self.repository.update(interaction)

    def mark_stale(
        self,
        interaction_id: str,
        *,
        error_code: str = "hitl_interrupt_missing",
    ) -> HITLInteraction:
        interaction = self._get(interaction_id)
        interaction.mark_stale(error_code=error_code)
        return self.repository.update(interaction)

    def expire(self, interaction_id: str) -> HITLInteraction:
        interaction = self._get(interaction_id)
        interaction.expire()
        return self.repository.update(interaction)

    def _get(self, interaction_id: str) -> HITLInteraction:
        try:
            return self.repository.get(interaction_id)
        except InteractionNotFoundError as error:
            raise HITLServiceError("HITL interaction not found") from error


__all__ = [
    "HITLService",
    "HITLServiceError",
    "HITLThreadBusyError",
]
