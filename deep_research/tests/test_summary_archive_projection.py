import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from deep_research.memory.projection import (
    render_summary_archive_markdown,
    summary_archive_filename,
    summary_archive_path,
)
from deep_research.memory.type import SummaryArchive


def make_archive(
    *,
    archive_id: str = "archive-1",
    thread_id: str = "thread-1",
    summary_hash: str = "hash-1/unsafe",
) -> SummaryArchive:
    return SummaryArchive.model_validate(
        {
            "id": archive_id,
            "thread_id": thread_id,
            "summary_hash": summary_hash,
            "original_summary": "Original conversation summary.",
            "retrieval_summary": "The project uses Main-only memory.",
            "topics": ["memory", "architecture"],
            "version": 2,
            "archived_at": datetime.now(timezone.utc),
        }
    )


class SummaryArchiveProjectionTests(unittest.TestCase):
    def test_filename_is_safe_and_path_is_internal(self):
        archive = make_archive()

        filename = summary_archive_filename(archive)

        self.assertTrue(filename.startswith("summary-"))
        self.assertTrue(filename.endswith(".md"))
        self.assertNotIn("/", filename)
        self.assertNotIn("\\", filename)

        with patch(
            "deep_research.memory.projection.projection_root",
            return_value=Path("projection-root"),
        ):
            path = summary_archive_path(archive)

        self.assertEqual(
            path.parent,
            Path("projection-root")
            / "memory-internal"
            / "summary-archive",
        )

    def test_render_contains_retrieval_original_topics_and_metadata(self):
        rendered = render_summary_archive_markdown(make_archive())

        self.assertIn("## Retrieval Summary", rendered)
        self.assertIn("The project uses Main-only memory.", rendered)
        self.assertIn("## Original Summary", rendered)
        self.assertIn("Original conversation summary.", rendered)
        self.assertIn("## Topics", rendered)
        self.assertIn("memory, architecture", rendered)
        self.assertIn("- Archive ID: archive-1", rendered)
        self.assertIn("- Thread ID: thread-1", rendered)
        self.assertIn("- Summary hash: hash-1/unsafe", rendered)
        self.assertIn("- Version: 2", rendered)

    def test_same_summary_hash_has_stable_projection_filename(self):
        first = summary_archive_filename(
            make_archive(archive_id="archive-1")
        )
        second = summary_archive_filename(
            make_archive(archive_id="archive-2")
        )

        self.assertEqual(first, second)

    def test_same_hash_in_different_threads_has_different_filename(self):
        first = summary_archive_filename(
            make_archive(thread_id="thread-1")
        )
        second = summary_archive_filename(
            make_archive(thread_id="thread-2")
        )

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
