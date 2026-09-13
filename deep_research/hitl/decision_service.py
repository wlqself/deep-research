"""Application-level bridges from HITL decisions to publishing workflow."""

from __future__ import annotations

from dataclasses import dataclass

from ..publishing.approvals import ApprovalRequest, ApprovalStatus
from ..publishing.service import PublishingService
from .models import HITLAction, HITLInteraction, HITLStatus
from .service import HITLService, HITLServiceError

"""
新增 HITL 与 PublishingService 的批准桥接
校验线程、目标类型、文章版本
先更新 Publishing approval，再更新 HITL 状态
支持重复请求和异常后的重试协调
"""
# 异常与结果类
class HITLDecisionError(HITLServiceError):
    """A safe, deterministic failure while applying a HITL decision."""

    def __init__(self, error_code: str, message: str = "") -> None:
        super().__init__(message or error_code)
        self.error_code = error_code

# 一次 HITL 决策及其对应的发布审批请求已同步完成
@dataclass(frozen=True, slots=True)
class HITLDecisionResult:
    """The synchronized result of one HITL decision and its target."""

    interaction: HITLInteraction
    publication_approval: ApprovalRequest

# hitl_service（管理 HITL 交互生命周期）和 publishing_service（管理文章/审批/发布）
class HITLDecisionService:
    """Apply durable HITL decisions to deterministic application services."""

    def __init__(
        self,
        hitl_service: HITLService,
        publishing_service: PublishingService,
    ) -> None:
        self.hitl_service = hitl_service
        self.publishing_service = publishing_service
    # 批准发布
    def approve_publication(
        self,
        interaction_id: str,
        *,
        thread_id: str,
        decision_actor: str,
        decision_reason: str | None = None,
    ) -> HITLDecisionResult:
        """Approve one exact publication approval and synchronize HITL state.

        Publishing and HITL use separate SQLite databases, so this method is
        deliberately retryable. It first makes the publishing decision, then
        records the matching HITL decision. If either side already completed,
        the next call reconciles the remaining side instead of duplicating the
        operation.
        """
        # 获取并校验 HITL 交互
        interaction = self._get_publication_interaction(
            interaction_id,
            thread_id=thread_id,
        )
        # 获取并校验目标审批请求
        approval = self._get_target_approval(interaction)

        if approval.status is ApprovalStatus.PENDING:
            try:
                approval = self.publishing_service.approve_publication_request(
                    approval.approval_id,
                    decision_actor=decision_actor,
                    decision_reason=decision_reason,
                )
            except Exception as error:
                approval = self._get_target_approval(interaction)
                # 如果抛异常，重新获取审批状态：如果已经变成 APPROVED（说明第一次其实成功了，只是网络/超时问题），就继续；否则抛异常；
                if approval.status is not ApprovalStatus.APPROVED:
                    raise HITLDecisionError(
                        "hitl_publication_decision_failed",
                        "publication approval could not be applied",
                    ) from error
        elif approval.status is not ApprovalStatus.APPROVED:
            raise HITLDecisionError(
                "hitl_publication_decision_conflict",
                "publication approval is no longer pending",
            )
        # HITL 侧。逻辑和上面完全对称：先尝试写，失败则重新读取确认状态，已成功就继续，否则抛异常；
        if interaction.status is HITLStatus.PENDING:
            try:
                interaction = self.hitl_service.approve(
                    interaction.interaction_id,
                    decision_actor=decision_actor,
                    decision_reason=decision_reason,
                )
            except Exception as error:
                latest = self.hitl_service.get_interaction(
                    interaction.interaction_id,
                )
                if latest.status is not HITLStatus.APPROVED:
                    raise HITLDecisionError(
                        "hitl_decision_record_failed",
                        "HITL approval could not be recorded",
                    ) from error
                interaction = latest
        elif interaction.status is not HITLStatus.APPROVED:
            raise HITLDecisionError(
                "hitl_decision_conflict",
                "HITL interaction is no longer pending",
            )

        return HITLDecisionResult(
            interaction=interaction,
            publication_approval=approval,
        )
    # 拒绝发布，期望 action 是 REJECT
    def reject_publication(
        self,
        interaction_id: str,
        *,
        thread_id: str,
        decision_actor: str,
        decision_reason: str | None = None,
    ) -> HITLDecisionResult:
        """Reject one exact publication approval and synchronize HITL state."""

        interaction = self._get_publication_interaction(
            interaction_id,
            thread_id=thread_id,
            expected_action={HITLAction.APPROVE, HITLAction.REJECT},
        )
        approval = self._get_target_approval(interaction)

        if approval.status is ApprovalStatus.PENDING:
            try:
                approval = self.publishing_service.reject_publication_request(
                    approval.approval_id,
                    decision_actor=decision_actor,
                    decision_reason=decision_reason,
                )
            except Exception as error:
                approval = self._get_target_approval(interaction)
                if approval.status is not ApprovalStatus.REJECTED:
                    raise HITLDecisionError(
                        "hitl_publication_decision_failed",
                        "publication rejection could not be applied",
                    ) from error
        elif approval.status is not ApprovalStatus.REJECTED:
            raise HITLDecisionError(
                "hitl_publication_decision_conflict",
                "publication approval is no longer pending",
            )

        if interaction.status is HITLStatus.PENDING:
            try:
                interaction = self.hitl_service.reject(
                    interaction.interaction_id,
                    decision_actor=decision_actor,
                    decision_reason=decision_reason,
                )
            except Exception as error:
                latest = self.hitl_service.get_interaction(
                    interaction.interaction_id,
                )
                if latest.status is not HITLStatus.REJECTED:
                    raise HITLDecisionError(
                        "hitl_decision_record_failed",
                        "HITL rejection could not be recorded",
                    ) from error
                interaction = latest
        elif interaction.status is not HITLStatus.REJECTED:
            raise HITLDecisionError(
                "hitl_decision_conflict",
                "HITL interaction is no longer pending",
            )

        return HITLDecisionResult(
            interaction=interaction,
            publication_approval=approval,
        )

    def _get_publication_interaction(
        self,
        interaction_id: str,
        *,
        thread_id: str,
        expected_action: HITLAction | set[HITLAction] = HITLAction.APPROVE,
    ) -> HITLInteraction:
        try:
            interaction = self.hitl_service.get_interaction(interaction_id)
        except HITLServiceError as error:
            raise HITLDecisionError(
                "hitl_interaction_not_found",
                "HITL interaction not found",
            ) from error

        if interaction.thread_id != thread_id:
            raise HITLDecisionError(
                "hitl_interaction_not_found",
                "HITL interaction not found",
            )
        expected_actions = (
            expected_action
            if isinstance(expected_action, set)
            else {expected_action}
        )
        if (
            interaction.action not in expected_actions
            or interaction.target_type != "publication_approval"
        ):
            raise HITLDecisionError(
                "hitl_decision_target_invalid",
                "HITL interaction target is not a publication approval",
            )
        return interaction

    def _get_target_approval(
        self,
        interaction: HITLInteraction,
    ) -> ApprovalRequest:
        try:
            approval = self.publishing_service.repository.get_approval_request(
                interaction.target_id,
            )
        except Exception as error:
            raise HITLDecisionError(
                "hitl_publication_target_not_found",
                "publication approval target not found",
            ) from error

        if approval.article_version != interaction.target_version:
            raise HITLDecisionError(
                "hitl_publication_target_stale",
                "publication approval target is stale",
            )
        return approval


__all__ = [
    "HITLDecisionError",
    "HITLDecisionResult",
    "HITLDecisionService",
]
