import unittest
import unittest

from datetime import datetime, timezone

from pydantic import ValidationError

from deep_research.memory.type import (
    MemoryCandidate,
    MemoryEntry,
    MemorySuppression,
    SummaryArchive,
)


def entry_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "memory-1",
        "memory_key": "user.answer_style.conciseness",
        "scope": "global",
        "kind": "user",
        "title": "Answer style",
        "summary": "The user prefers concise answers.",
        "content": "Prefer concise answers.",
        "keywords": ["style"],
        "source_type": "explicit_user",
        "source_thread_id": "thread-1",
        "created_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T00:00:00Z",
        "status": "active",
    }
    data.update(overrides)
    return data


def candidate_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "kind": "user",
        "memory_key": "user.answer_style.conciseness",
        "scope": "global",
        "title": "Answer style",
        "summary": "The user prefers concise answers.",
        "content": "Prefer concise answers.",
        "keywords": ["style"],
        "source_type": "explicit_user",
        "source_thread_id": "thread-1",
    }
    data.update(overrides)
    return data


def suppression_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "suppression_id": "suppression-1",
        "kind": "user",
        "memory_key": "user.phone",
        "scope": "global",
        "forgotten_memory_id": "memory-1",
        "source_thread_id": "thread-1",
        "created_at": datetime.now(timezone.utc),
        "reason": "User requested forget.",
    }
    data.update(overrides)
    return data


class MemoryTypeTests(unittest.TestCase):
    def test_entry_accepts_user_project_and_basic_fields(self):
        for kind in ("user", "project"):
            entry = MemoryEntry.model_validate(entry_data(kind=kind))
            self.assertEqual(entry.kind, kind)

    def test_memory_key_and_scope_are_validated(self):
        entry = MemoryEntry.model_validate(
            entry_data(
                memory_key="project.research.boundary",
                scope="project",
                kind="project",
            )
        )
        self.assertEqual(entry.memory_key, "project.research.boundary")
        self.assertEqual(entry.scope, "project")

        with self.assertRaises(ValidationError):
            MemoryEntry.model_validate(
                entry_data(memory_key="User answer style")
            )

        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(
                candidate_data(scope="thread")
            )

    def test_reference_requires_http_url_and_verified_at(self):
        entry = MemoryEntry.model_validate(
            entry_data(
                kind="reference",
                url="https://example.com/reference",
                verified_at="2026-08-28T00:00:00Z",
            )
        )
        self.assertEqual(str(entry.url), "https://example.com/reference")

        with self.assertRaises(ValidationError):
            MemoryEntry.model_validate(
                entry_data(
                    kind="reference",
                    verified_at="2026-08-28T00:00:00Z",
                )
            )

        with self.assertRaises(ValidationError):
            MemoryEntry.model_validate(
                entry_data(
                    kind="reference",
                    url="ftp://example.com/reference",
                    verified_at="2026-08-28T00:00:00Z",
                )
            )

    def test_feedback_requires_three_structured_fields(self):
        entry = MemoryEntry.model_validate(
            entry_data(
                kind="feedback",
                incorrect="Using a broad answer is ineffective.",
                correct="Answer only the requested scope.",
                applies_when="When the user asks for a focused code review.",
            )
        )
        self.assertEqual(entry.kind, "feedback")

        with self.assertRaises(ValidationError):
            MemoryEntry.model_validate(
                entry_data(
                    kind="feedback",
                    incorrect="An incorrect approach.",
                    correct="",
                    applies_when="A specific condition.",
                )
            )

    def test_ignore_is_candidate_only(self):
        candidate = MemoryCandidate.model_validate(
            candidate_data(kind="ignore")
        )
        self.assertEqual(candidate.kind, "ignore")

        with self.assertRaises(ValidationError):
            MemoryEntry.model_validate(entry_data(kind="ignore"))

    def test_candidate_validates_reference_and_feedback(self):
        MemoryCandidate.model_validate(
            candidate_data(
                kind="reference",
                url="http://example.com/reference",
                verified_at="2026-08-28T00:00:00Z",
            )
        )
        MemoryCandidate.model_validate(
            candidate_data(
                kind="feedback",
                incorrect="Incorrect approach.",
                correct="Correct approach.",
                applies_when="When applicable.",
            )
        )

    def test_models_reject_unknown_fields(self):
        with self.assertRaises(ValidationError):
            MemoryEntry.model_validate(entry_data(secret="must reject"))

        with self.assertRaises(ValidationError):
            MemoryCandidate.model_validate(candidate_data(secret="must reject"))

        with self.assertRaises(ValidationError):
            MemorySuppression.model_validate(
                suppression_data(user_id="must reject")
            )

        with self.assertRaises(ValidationError):
            MemorySuppression.model_validate(
                suppression_data(namespace="must reject")
            )

    def test_memory_suppression_accepts_valid_data(self):
        suppression = MemorySuppression.model_validate(
            suppression_data()
        )

        self.assertEqual(suppression.kind, "user")
        self.assertEqual(suppression.memory_key, "user.phone")
        self.assertEqual(suppression.scope, "global")
        self.assertEqual(
            suppression.forgotten_memory_id,
            "memory-1",
        )

    def test_memory_suppression_requires_non_empty_identity_fields(self):
        for field in (
            "suppression_id",
            "memory_key",
            "forgotten_memory_id",
            "reason",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    MemorySuppression.model_validate(
                        suppression_data(**{field: ""})
                    )

    def test_memory_suppression_validates_kind_scope_and_key(self):
        for kind in ("user", "reference", "project", "feedback"):
            with self.subTest(kind=kind):
                suppression = MemorySuppression.model_validate(
                    suppression_data(kind=kind)
                )
                self.assertEqual(suppression.kind, kind)

        for scope in ("global", "project"):
            with self.subTest(scope=scope):
                suppression = MemorySuppression.model_validate(
                    suppression_data(scope=scope)
                )
                self.assertEqual(suppression.scope, scope)

        for memory_key in ("User.Phone", "user phone", ".user.phone"):
            with self.subTest(memory_key=memory_key):
                with self.assertRaises(ValidationError):
                    MemorySuppression.model_validate(
                        suppression_data(memory_key=memory_key)
                    )

        with self.assertRaises(ValidationError):
            MemorySuppression.model_validate(
                suppression_data(kind="ignore")
            )

        with self.assertRaises(ValidationError):
            MemorySuppression.model_validate(
                suppression_data(scope="thread")
            )

    def test_summary_archive_has_deduplication_fields(self):
        archive = SummaryArchive.model_validate(
            {
                "id": "archive-1",
                "thread_id": "thread-1",
                "summary_hash": "abc123",
                "original_summary": "Original summary.",
                "retrieval_summary": "Retrieval summary.",
                "topics": ["memory"],
                "version": 1,
                "archived_at": datetime.now(timezone.utc),
            }
        )
        self.assertEqual(archive.thread_id, "thread-1")
        self.assertEqual(archive.summary_hash, "abc123")

        with self.assertRaises(ValidationError):
            SummaryArchive.model_validate(
                {
                    "id": "archive-2",
                    "thread_id": "thread-1",
                    "summary_hash": "abc123",
                    "original_summary": "Original summary.",
                    "retrieval_summary": "Retrieval summary.",
                    "topics": [],
                    "version": 0,
                    "archived_at": datetime.now(timezone.utc),
                }
            )


if __name__ == "__main__":
    unittest.main()
