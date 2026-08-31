import unittest

from deep_research.memory.decision import decide_candidate
from deep_research.memory.type import MemoryCandidate, MemoryEntry


def make_candidate(
    *,
    kind: str = "user",
    memory_key: str = "user.answer_style.conciseness",
    title: str = "Answer style",
    summary: str = "The user prefers concise answers.",
    content: str = "Prefer concise answers.",
    keywords: list[str] | None = None,
) -> MemoryCandidate:
    return MemoryCandidate.model_validate(
        {
            "kind": kind,
            "memory_key": memory_key,
            "scope": "global",
            "title": title,
            "summary": summary,
            "content": content,
            "keywords": keywords or ["concise", "answer"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
        }
    )


def make_entry(
    *,
    entry_id: str = "memory-1",
    kind: str = "user",
    memory_key: str = "user.answer_style.conciseness",
    title: str = "Answer style",
    summary: str = "The user prefers concise answers.",
    content: str = "Prefer concise answers.",
    keywords: list[str] | None = None,
    status: str = "active",
) -> MemoryEntry:
    return MemoryEntry.model_validate(
        {
            "id": entry_id,
            "kind": kind,
            "memory_key": memory_key,
            "scope": "global",
            "title": title,
            "summary": summary,
            "content": content,
            "keywords": keywords or ["concise", "answer"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
            "created_at": "2026-08-28T00:00:00Z",
            "updated_at": "2026-08-28T00:00:00Z",
            "status": status,
        }
    )


class MemoryDecisionTests(unittest.TestCase):
    def test_ignore_returns_ignore_without_existing_id(self):
        decision = decide_candidate(
            make_candidate(kind="ignore"),
            [],
            intent="automatic",
        )

        self.assertEqual(decision.action, "ignore")
        self.assertIsNone(decision.existing_id)

    def test_new_key_returns_create(self):
        decision = decide_candidate(
            make_candidate(),
            [],
            intent="automatic",
        )

        self.assertEqual(decision.action, "create")
        self.assertIsNone(decision.existing_id)

    def test_same_content_returns_no_op(self):
        existing = make_entry()
        candidate = make_candidate(
            title="  Answer style  ",
            summary="The user prefers concise answers.",
            keywords=["ANSWER", "concise"],
        )

        decision = decide_candidate(
            candidate,
            [existing],
            intent="explicit_remember",
        )

        self.assertEqual(decision.action, "no-op")
        self.assertEqual(decision.existing_id, "memory-1")
        self.assertEqual(decision.reason, "same_content")

    def test_explicit_update_returns_update_for_same_key(self):
        existing = make_entry()
        candidate = make_candidate(
            summary="The user prefers detailed answers.",
            content="Prefer detailed answers.",
            keywords=["detailed", "answer"],
        )

        decision = decide_candidate(
            candidate,
            [existing],
            intent="explicit_update",
        )

        self.assertEqual(decision.action, "update")
        self.assertEqual(decision.existing_id, "memory-1")

    def test_automatic_conflict_does_not_overwrite_existing(self):
        existing = make_entry()
        candidate = make_candidate(
            summary="The user prefers detailed answers.",
            content="Prefer detailed answers.",
            keywords=["detailed", "answer"],
        )

        decision = decide_candidate(
            candidate,
            [existing],
            intent="automatic",
        )

        self.assertEqual(decision.action, "no-op")
        self.assertEqual(decision.existing_id, "memory-1")
        self.assertEqual(
            decision.reason,
            "conflict_without_explicit_update",
        )

    def test_different_kind_or_key_is_not_a_matching_entry(self):
        existing_entries = [
            make_entry(
                entry_id="code-style",
                memory_key="user.code_style.conciseness",
            ),
            make_entry(
                entry_id="project-style",
                kind="project",
                memory_key="project.answer_style.conciseness",
            ),
        ]

        decision = decide_candidate(
            make_candidate(),
            existing_entries,
            intent="automatic",
        )

        self.assertEqual(decision.action, "create")

    def test_superseded_entry_does_not_match(self):
        decision = decide_candidate(
            make_candidate(),
            [make_entry(status="superseded")],
            intent="automatic",
        )

        self.assertEqual(decision.action, "create")

    def test_multiple_active_same_key_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "multiple active entries share the same memory_key",
        ):
            decide_candidate(
                make_candidate(),
                [
                    make_entry(entry_id="memory-1"),
                    make_entry(entry_id="memory-2"),
                ],
                intent="automatic",
            )


if __name__ == "__main__":
    unittest.main()
