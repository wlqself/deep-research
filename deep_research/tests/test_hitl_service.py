import tempfile
import unittest
from pathlib import Path

from deep_research.hitl.models import (
    HITLAction,
    HITLStatus,
    InvalidHITLValueError,
    InvalidHITLTransitionError,
)
from deep_research.hitl.repository import HITLRepository
from deep_research.hitl.service import (
    HITLService,
    HITLServiceError,
    HITLThreadBusyError,
)


class HITLServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = HITLRepository(
            Path(self.temp_dir.name) / "hitl.sqlite"
        )
        self.repository.initialize()
        self.service = HITLService(self.repository)

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def create_interaction(
        self,
        *,
        interaction_id=None,
        thread_id="thread-1",
        run_id="run-1",
        target_id="approval-1",
    ):
        return self.service.create_or_get_interaction(
            interaction_id=interaction_id,
            thread_id=thread_id,
            run_id=run_id,
            action=HITLAction.APPROVE,
            target_type="publication_approval",
            target_id=target_id,
            target_version=1,
        )

    def test_create_or_get_is_idempotent(self):
        first = self.create_interaction()
        replay = self.create_interaction(interaction_id="ignored-on-replay")

        self.assertEqual(first.interaction_id, replay.interaction_id)
        self.assertEqual(len(self.repository.list_for_run("run-1")), 1)

    def test_same_target_is_reused_when_agent_run_id_changes(self):
        first = self.create_interaction(run_id="run-1")
        replay = self.create_interaction(
            run_id="run-2",
            interaction_id="ignored-for-same-target",
        )

        self.assertEqual(first.interaction_id, replay.interaction_id)
        self.assertEqual(
            len(
                self.repository.list_for_target(
                    "publication_approval",
                    "approval-1",
                    target_version=1,
                    statuses=(HITLStatus.PENDING,),
                )
            ),
            1,
        )

    def test_new_interaction_is_blocked_until_previous_one_is_resumed(self):
        first = self.create_interaction()

        with self.assertRaises(HITLThreadBusyError) as pending_error:
            self.create_interaction(run_id="run-2", target_id="approval-2")
        self.assertEqual(pending_error.exception.interaction.interaction_id, first.interaction_id)

        self.service.approve(first.interaction_id, decision_actor="user")
        with self.assertRaises(HITLThreadBusyError):
            self.create_interaction(run_id="run-2", target_id="approval-2")

        self.service.start_resume(first.interaction_id)
        with self.assertRaises(HITLThreadBusyError):
            self.create_interaction(run_id="run-2", target_id="approval-2")

        self.service.mark_resumed(first.interaction_id)
        second = self.create_interaction(run_id="run-2", target_id="approval-2")
        self.assertEqual(second.target_id, "approval-2")

    def test_recoverable_query_is_limited_to_thread(self):
        pending = self.create_interaction()
        other = self.create_interaction(
            interaction_id="interaction-2",
            thread_id="thread-2",
            run_id="run-2",
            target_id="approval-2",
        )
        other.approve(decision_actor="user")
        self.repository.update(other)

        records = self.service.list_recoverable_for_thread("thread-1")

        self.assertEqual(
            [record.interaction_id for record in records],
            [pending.interaction_id],
        )

    def test_approve_then_resume(self):
        interaction = self.create_interaction()

        approved = self.service.approve(
            interaction.interaction_id,
            decision_actor="user",
            decision_reason="reviewed",
        )
        self.assertEqual(approved.status, HITLStatus.APPROVED)

        resuming = self.service.start_resume(interaction.interaction_id)
        self.assertEqual(resuming.status, HITLStatus.RESUMING)

        resumed = self.service.mark_resumed(interaction.interaction_id)
        self.assertEqual(resumed.status, HITLStatus.RESUMED)

    def test_reject_requires_reason_and_is_terminal(self):
        interaction = self.create_interaction()

        with self.assertRaises(InvalidHITLValueError):
            self.service.reject(
                interaction.interaction_id,
                decision_actor="user",
                decision_reason="",
            )

        rejected = self.service.reject(
            interaction.interaction_id,
            decision_actor="user",
            decision_reason="内容需要修改",
        )
        self.assertEqual(rejected.status, HITLStatus.REJECTED)

    def test_failed_interaction_can_resume_again(self):
        interaction = self.create_interaction()
        self.service.approve(interaction.interaction_id, decision_actor="user")
        self.service.start_resume(interaction.interaction_id)
        failed = self.service.mark_failed(
            interaction.interaction_id,
            error_code="resume_failed",
        )
        self.assertEqual(failed.status, HITLStatus.FAILED)

        retried = self.service.start_resume(interaction.interaction_id)
        self.assertEqual(retried.status, HITLStatus.RESUMING)

    def test_illegal_resume_is_rejected(self):
        interaction = self.create_interaction()

        with self.assertRaises(InvalidHITLTransitionError):
            self.service.start_resume(interaction.interaction_id)

    def test_missing_interaction_is_safe_service_error(self):
        with self.assertRaises(HITLServiceError):
            self.service.approve("missing", decision_actor="user")

    def test_expire_removes_interaction_from_recoverable_results(self):
        interaction = self.create_interaction()
        self.service.expire(interaction.interaction_id)

        self.assertEqual(
            self.service.list_recoverable_for_thread("thread-1"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
