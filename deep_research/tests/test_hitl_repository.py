import tempfile
import unittest
from pathlib import Path

from deep_research.hitl.models import (
    HITLAction,
    HITLInteraction,
    HITLStatus,
    InvalidHITLTransitionError,
)
from deep_research.hitl.repository import (
    DuplicateInteractionError,
    HITLRepository,
    InteractionNotFoundError,
    RepositoryClosedError,
)


def build_interaction(
    *,
    interaction_id="interaction-1",
    thread_id="thread-1",
    run_id="run-1",
    target_id="approval-1",
):
    return HITLInteraction(
        interaction_id=interaction_id,
        thread_id=thread_id,
        run_id=run_id,
        action=HITLAction.APPROVE,
        target_type="publication_approval",
        target_id=target_id,
        target_version=1,
    )


class HITLModelTests(unittest.TestCase):
    def test_approval_and_resume_are_separate_transitions(self):
        interaction = build_interaction()

        interaction.approve(decision_actor="user")
        self.assertEqual(interaction.status, HITLStatus.APPROVED)
        self.assertTrue(interaction.can_resume)

        interaction.start_resume()
        self.assertEqual(interaction.status, HITLStatus.RESUMING)

        interaction.mark_resumed()
        self.assertEqual(interaction.status, HITLStatus.RESUMED)
        self.assertTrue(interaction.is_terminal)

    def test_failed_resume_can_be_retried(self):
        interaction = build_interaction()
        interaction.approve(decision_actor="user")
        interaction.start_resume()
        interaction.mark_failed(error_code="resume_failed")

        self.assertEqual(interaction.status, HITLStatus.FAILED)
        self.assertTrue(interaction.can_resume)

        interaction.start_resume()
        self.assertEqual(interaction.status, HITLStatus.RESUMING)

    def test_illegal_transition_is_rejected(self):
        interaction = build_interaction()

        with self.assertRaises(InvalidHITLTransitionError):
            interaction.mark_resumed()


class HITLRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "hitl.sqlite"
        self.repository = HITLRepository(self.db_path)
        self.repository.initialize()

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def test_database_can_reopen_and_restore_state(self):
        interaction = build_interaction()
        self.repository.create(interaction)

        interaction.approve(decision_actor="user")
        self.repository.update(interaction)
        self.repository.close()

        reopened = HITLRepository(self.db_path)
        reopened.initialize()
        restored = reopened.get(interaction.interaction_id)

        self.assertEqual(restored.status, HITLStatus.APPROVED)
        self.assertEqual(restored.thread_id, "thread-1")
        reopened.close()

    def test_request_key_is_unique_and_findable(self):
        interaction = build_interaction()
        self.repository.create(interaction)

        with self.assertRaises(DuplicateInteractionError):
            self.repository.create(
                build_interaction(interaction_id="interaction-2")
            )

        found = self.repository.find_by_request_key(
            run_id="run-1",
            action=HITLAction.APPROVE,
            target_type="publication_approval",
            target_id="approval-1",
            target_version=1,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.interaction_id, "interaction-1")

    def test_thread_query_does_not_cross_thread_boundary(self):
        self.repository.create(build_interaction())
        self.repository.create(
            build_interaction(
                interaction_id="interaction-2",
                thread_id="thread-2",
                run_id="run-2",
                target_id="approval-2",
            )
        )

        records = self.repository.list_for_thread("thread-1")

        self.assertEqual([record.interaction_id for record in records], [
            "interaction-1",
        ])

    def test_target_query_finds_interactions_across_threads(self):
        self.repository.create(build_interaction())
        self.repository.create(
            build_interaction(
                interaction_id="interaction-2",
                thread_id="thread-2",
                run_id="run-2",
            )
        )

        records = self.repository.list_for_target(
            "publication_approval",
            "approval-1",
            statuses=(HITLStatus.PENDING,),
        )

        self.assertEqual(
            sorted(record.interaction_id for record in records),
            ["interaction-1", "interaction-2"],
        )

    def test_closed_repository_rejects_reads(self):
        self.repository.close()

        with self.assertRaises(RepositoryClosedError):
            self.repository.list_for_thread("thread-1")

    def test_missing_interaction_is_reported(self):
        with self.assertRaises(InteractionNotFoundError):
            self.repository.get("missing")

    def test_resume_claim_has_one_winner(self):
        interaction = build_interaction()
        self.repository.create(interaction)
        interaction.approve(decision_actor="user")
        self.repository.update(interaction)

        first, claimed = self.repository.claim_resume(
            interaction.interaction_id,
        )
        second, claimed_again = self.repository.claim_resume(
            interaction.interaction_id,
        )

        self.assertTrue(claimed)
        self.assertFalse(claimed_again)
        self.assertEqual(first.status, HITLStatus.RESUMING)
        self.assertEqual(second.status, HITLStatus.RESUMING)

    def test_runtime_binding_is_persisted(self):
        interaction = build_interaction()
        self.repository.create(interaction)

        interaction.interrupt_id = "interrupt-1"
        interaction.checkpoint_id = "checkpoint-1"
        self.repository.update(interaction)

        restored = self.repository.get(interaction.interaction_id)
        self.assertEqual(restored.interrupt_id, "interrupt-1")
        self.assertEqual(restored.checkpoint_id, "checkpoint-1")

    def test_old_schema_is_migrated_to_stale_and_runtime_columns(self):
        self.repository.close()
        legacy_db_path = Path(self.temp_dir.name) / "legacy.sqlite"
        connection = __import__("sqlite3").connect(legacy_db_path)
        connection.executescript(
            """
            CREATE TABLE hitl_interactions (
                interaction_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                action TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                target_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                decision_actor TEXT,
                decision_reason TEXT,
                error_code TEXT,
                UNIQUE (run_id, action, target_type, target_id, target_version)
            );
            INSERT INTO hitl_interactions VALUES (
                'legacy-1', 'thread-1', 'run-1', 'approve',
                'publication_approval', 'approval-1', 1, 'approved',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00', NULL, 'user', NULL, NULL
            );
            """
        )
        connection.commit()
        connection.close()

        self.repository = __import__(
            "deep_research.hitl.repository",
            fromlist=["HITLRepository"],
        ).HITLRepository(legacy_db_path)
        self.repository.initialize()
        restored = self.repository.get("legacy-1")

        self.assertEqual(restored.status, HITLStatus.APPROVED)
        self.assertIsNone(restored.interrupt_id)
        self.assertIsNone(restored.checkpoint_id)


if __name__ == "__main__":
    unittest.main()
