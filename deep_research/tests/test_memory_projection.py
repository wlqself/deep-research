import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from deep_research.config import settings
from deep_research.memory import projection
from deep_research.memory.projection import render_entry_markdown
from deep_research.memory.type import MemoryEntry


def make_entry(**overrides: object) -> MemoryEntry:
    data: dict[str, object] = {
        "id": "memory-1",
        "memory_key": "user.answer_style.conciseness",
        "scope": "global",
        "kind": "user",
        "title": "Answer style",
        "summary": "The user prefers concise answers.",
        "content": "Prefer concise answers.",
        "keywords": ["style", "answer"],
        "source_type": "explicit_user",
        "source_thread_id": "thread-1",
        "created_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T01:00:00Z",
        "status": "active",
    }
    data.update(overrides)
    return MemoryEntry.model_validate(data)


class MemoryProjectionTests(unittest.TestCase):
    def test_projection_directories_are_created_under_configured_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(settings, "memory_projection_path", temp_dir):
                projection.ensure_projection_directories()

            root = Path(temp_dir)
            self.assertTrue((root / "memories" / "reference").is_dir())
            self.assertTrue((root / "memories" / "project").is_dir())
            self.assertTrue((root / "memories" / "feedback").is_dir())
            self.assertTrue(
                (root / "memory-internal" / "summary-archive").is_dir()
            )

    def test_atomic_writer_replaces_complete_file_without_temp_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "USER.md"

            projection.write_text_atomic(target, "first\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "first\n")

            projection.write_text_atomic(target, "second\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "second\n")
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_atomic_writer_cleans_temp_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "USER.md"

            with patch.object(
                projection.os,
                "replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaises(OSError):
                    projection.write_text_atomic(target, "content\n")

            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_index_contains_all_fields_and_assigns_paged_entries(self):
        entries = [
            make_entry(
                id="project-1",
                kind="project",
                memory_key="project.one",
                updated_at="2026-08-28T01:00:00Z",
            ),
            make_entry(
                id="project-2",
                kind="project",
                memory_key="project.two",
                updated_at="2026-08-28T02:00:00Z",
            ),
            make_entry(
                id="project-3",
                kind="project",
                memory_key="project.three",
                updated_at="2026-08-28T03:00:00Z",
            ),
        ]

        rendered = projection.render_index_markdown(
            entries,
            page_size=2,
        )

        self.assertIn(
            "| ID | Page | Title | Summary | Keywords | Updated | Status |",
            rendered,
        )
        self.assertEqual(rendered.count("page-001.md"), 2)
        self.assertEqual(rendered.count("page-002.md"), 1)
        self.assertLess(
            rendered.index("project-3"),
            rendered.index("project-1"),
        )

    def test_index_escapes_table_cells(self):
        rendered = projection.render_index_markdown(
            [
                make_entry(
                    title="Input | output",
                    summary="A | B",
                    keywords=["x|y"],
                )
            ],
            page_size=10,
        )

        self.assertIn("Input \\| output", rendered)
        self.assertIn("A \\| B", rendered)
        self.assertIn("x\\|y", rendered)

    def test_user_projection_keeps_only_active_user_entries(self):
        rendered = projection.render_user_markdown(
            [
                make_entry(
                    id="user-old",
                    title="Old user preference",
                    memory_key="user.old.preference",
                    updated_at="2026-08-28T01:00:00Z",
                ),
                make_entry(
                    id="user-new",
                    title="New user preference",
                    memory_key="user.new.preference",
                    updated_at="2026-08-28T03:00:00Z",
                ),
                make_entry(
                    id="user-superseded",
                    title="Superseded user preference",
                    memory_key="user.superseded.preference",
                    status="superseded",
                    updated_at="2026-08-28T04:00:00Z",
                ),
                make_entry(
                    id="project-entry",
                    title="Project preference",
                    kind="project",
                    memory_key="project.entry",
                    scope="project",
                    updated_at="2026-08-28T05:00:00Z",
                ),
            ],
            max_entries=10,
            max_chars=3000,
        )

        self.assertIn("user.new.preference", rendered)
        self.assertIn("user.old.preference", rendered)
        self.assertNotIn("user.superseded.preference", rendered)
        self.assertNotIn("project.entry", rendered)

    def test_user_projection_respects_entry_and_character_limits(self):
        entries = [
            make_entry(
                id=f"user-{index}",
                title=f"User preference {index}",
                memory_key=f"user.preference.{index}",
                updated_at=f"2026-08-28T0{index}:00:00Z",
            )
            for index in range(1, 4)
        ]

        rendered = projection.render_user_markdown(
            entries,
            max_entries=2,
            max_chars=3000,
        )

        self.assertIn("user.preference.3", rendered)
        self.assertIn("user.preference.2", rendered)
        self.assertNotIn("user.preference.1", rendered)

        bounded = projection.render_user_markdown(
            [
                make_entry(summary="very long summary " * 100),
            ],
            max_entries=10,
            max_chars=120,
        )
        self.assertLessEqual(len(bounded), 120)

    def test_user_projection_contains_core_sections_and_metadata(self):
        rendered = render_entry_markdown(
            make_entry(
                title="Answer\nstyle",
                keywords=["concise\nanswer", "style"],
            )
        )

        self.assertIn("# Answer style\n", rendered)
        self.assertIn("## Summary\nThe user prefers concise answers.", rendered)
        self.assertIn("## Content\nPrefer concise answers.", rendered)
        self.assertIn("## When to read\n生成回答或调整交互方式时读取。", rendered)
        self.assertIn("- ID: memory-1", rendered)
        self.assertIn("- Memory key: user.answer_style.conciseness", rendered)
        self.assertIn("- Scope: global", rendered)
        self.assertIn("- Status: active", rendered)
        self.assertIn("- Updated: 2026-08-28T01:00:00+00:00", rendered)
        self.assertIn("- Keywords: concise answer, style", rendered)
        self.assertTrue(rendered.endswith("\n"))

    def test_reference_projection_contains_url_and_verification_time(self):
        rendered = render_entry_markdown(
            make_entry(
                kind="reference",
                memory_key="reference.langgraph.store",
                url="https://docs.langchain.com/oss/python/langgraph",
                verified_at="2026-08-28T02:00:00Z",
            )
        )

        self.assertIn(
            "- URL: https://docs.langchain.com/oss/python/langgraph",
            rendered,
        )
        self.assertIn(
            "- Verified at: 2026-08-28T02:00:00+00:00",
            rendered,
        )

    def test_feedback_projection_contains_structured_feedback(self):
        rendered = render_entry_markdown(
            make_entry(
                kind="feedback",
                memory_key="feedback.researcher.boundary",
                incorrect="Researcher used Main-only memory.",
                correct="Keep memory access in Main.",
                applies_when="When designing agent boundaries.",
            )
        )

        self.assertIn("## Feedback", rendered)
        self.assertIn(
            "- Incorrect: Researcher used Main-only memory.",
            rendered,
        )
        self.assertIn(
            "- Correct: Keep memory access in Main.",
            rendered,
        )
        self.assertIn(
            "- Applies when: When designing agent boundaries.",
            rendered,
        )

    def test_non_reference_url_does_not_require_or_render_verified_at(self):
        rendered = render_entry_markdown(
            make_entry(
                url="https://example.com/context",
            )
        )

        self.assertNotIn("- URL:", rendered)
        self.assertNotIn("- Verified at:", rendered)


if __name__ == "__main__":
    unittest.main()
