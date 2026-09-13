"""Canonical publication-approval state projection.

The publishing database keeps the article/version/channel approval record,
while the HITL database owns the human decision and graph lifecycle.  This
module is the single place where those records are interpreted together.
New callers must use the HITL interaction when one exists; the publishing
approval is only a compatibility fallback for records created before HITL was
enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..publishing.approvals import ApprovalRequest, ApprovalStatus
from .models import HITLAction, HITLInteraction, HITLStatus


@dataclass(frozen=True, slots=True)
class PublicationHITLState:
    """The canonical workflow state for one publication approval target."""

    approval_status: str
    workflow_status: str
    interaction_id: str | None
    interaction_status: str | None
    state_source: str
    action: str | None
    requires_user_confirmation: bool
    can_resume: bool


def _latest_publication_interaction(
    hitl_service,
    approval: ApprovalRequest,
) -> HITLInteraction | None:
    if hitl_service is None:
        return None
    interactions = hitl_service.list_for_target(
        "publication_approval",
        approval.approval_id,
        target_version=approval.article_version,
        statuses=None,
    )
    return next(
        (
            interaction
            for interaction in interactions
            if interaction.action is HITLAction.APPROVE
        ),
        None,
    )


def resolve_publication_hitl_state(
    approval: ApprovalRequest,
    hitl_service=None,
) -> PublicationHITLState:
    """Resolve one publication target from the canonical HITL lifecycle.

    ``ApprovalRequest.status`` remains available as a durable compatibility
    projection, but once an interaction exists the HITL status determines the
    human-decision state.  This is important during the retry window where
    one SQLite database may have committed before the other one did.
    """

    interaction = _latest_publication_interaction(hitl_service, approval)
    if interaction is None:
        return PublicationHITLState(
            approval_status=approval.status.value,
            workflow_status=approval.status.value,
            interaction_id=None,
            interaction_status=None,
            state_source="publishing_approval_legacy",
            action=("approve" if approval.status is ApprovalStatus.PENDING else None),
            requires_user_confirmation=approval.status is ApprovalStatus.PENDING,
            can_resume=False,
        )

    status = interaction.status
    if status is HITLStatus.PENDING:
        decision_status = ApprovalStatus.PENDING.value
        action = "approve"
        requires_confirmation = True
        can_resume = False
    elif status is HITLStatus.REJECTED:
        decision_status = ApprovalStatus.REJECTED.value
        action = "resume"
        requires_confirmation = True
        can_resume = True
    elif status in {
        HITLStatus.APPROVED,
        HITLStatus.RESUMING,
        HITLStatus.RESUMED,
        HITLStatus.FAILED,
    }:
        # FAILED means the human decision is already recorded and only the
        # graph continuation failed.  Preserve the decision as approved even
        # if the publishing projection has not caught up yet.
        decision_status = (
            ApprovalStatus.REJECTED.value
            if approval.status is ApprovalStatus.REJECTED
            else ApprovalStatus.APPROVED.value
        )
        action = (
            "resume"
            if status
            in {
                HITLStatus.APPROVED,
                HITLStatus.RESUMING,
                HITLStatus.FAILED,
            }
            else None
        )
        requires_confirmation = status in {HITLStatus.APPROVED, HITLStatus.FAILED}
        can_resume = status in {
            HITLStatus.APPROVED,
            HITLStatus.FAILED,
        }
    else:
        # STALE/EXPIRED are terminal recovery states.  They must not be
        # presented as a live confirmation, but keep the legacy decision
        # visible to the publishing center for diagnostics.
        decision_status = approval.status.value
        action = None
        requires_confirmation = False
        can_resume = False

    return PublicationHITLState(
        approval_status=decision_status,
        workflow_status=status.value,
        interaction_id=interaction.interaction_id,
        interaction_status=status.value,
        state_source="hitl_interaction",
        action=action,
        requires_user_confirmation=requires_confirmation,
        can_resume=can_resume,
    )


__all__ = [
    "PublicationHITLState",
    "resolve_publication_hitl_state",
]
