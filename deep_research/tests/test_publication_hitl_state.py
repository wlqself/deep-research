import tempfile
import unittest
from pathlib import Path

from deep_research.hitl.models import HITLAction, HITLStatus
from deep_research.hitl.publication_state import resolve_publication_hitl_state
from deep_research.hitl.repository import HITLRepository
from deep_research.hitl.service import HITLService
from deep_research.publishing.approvals import (
    ApprovalAction,
    ApprovalRequest,
    ApprovalStatus,
)
from deep_research.publishing.models import PublicationChannel


class PublicationHITLStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = HITLRepository(
            Path(self.temp_dir.name) / "hitl.sqlite"
        )
        self.repository.initialize()
        self.service = HITLService(self.repository)
        self.approval = ApprovalRequest(
            approval_id="approval-1",
            action=ApprovalAction.PUBLISH,
            article_id="article-1",
            article_version=1,
            channel=PublicationChannel.XIAOHONGSHU,
            content_sha256="a" * 64,
        )

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def create_interaction(self):
        return self.service.create_or_get_interaction(
            interaction_id="interaction-1",
            thread_id="thread-1",
            run_id="run-1",
            action=HITLAction.APPROVE,
            target_type="publication_approval",
            target_id=self.approval.approval_id,
            target_version=self.approval.article_version,
        )

    def test_legacy_approval_is_only_fallback_without_hitl_record(self):
        state = resolve_publication_hitl_state(self.approval, self.service)

        self.assertEqual(state.approval_status, ApprovalStatus.PENDING.value)
        self.assertEqual(state.state_source, "publishing_approval_legacy")
        self.assertIsNone(state.interaction_id)

    def test_hitl_pending_overrides_publishing_projection(self):
        self.approval.status = ApprovalStatus.APPROVED
        interaction = self.create_interaction()

        state = resolve_publication_hitl_state(self.approval, self.service)

        self.assertEqual(state.state_source, "hitl_interaction")
        self.assertEqual(state.interaction_id, interaction.interaction_id)
        self.assertEqual(state.interaction_status, HITLStatus.PENDING.value)
        self.assertEqual(state.workflow_status, HITLStatus.PENDING.value)
        self.assertEqual(state.approval_status, ApprovalStatus.PENDING.value)
        self.assertTrue(state.requires_user_confirmation)

    def test_failed_resume_state_is_recoverable_and_not_a_new_approval(self):
        interaction = self.create_interaction()
        self.service.approve(interaction.interaction_id, decision_actor="user")
        self.service.start_resume(interaction.interaction_id)
        self.service.mark_failed(
            interaction.interaction_id,
            error_code="hitl_graph_unavailable",
        )

        state = resolve_publication_hitl_state(self.approval, self.service)

        self.assertEqual(state.interaction_status, HITLStatus.FAILED.value)
        self.assertEqual(state.workflow_status, HITLStatus.FAILED.value)
        self.assertEqual(state.action, "resume")
        self.assertTrue(state.requires_user_confirmation)
        self.assertTrue(state.can_resume)


if __name__ == "__main__":
    unittest.main()
