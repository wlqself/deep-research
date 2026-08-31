import unittest
from typing import get_type_hints

from deep_research.state.research import ResearchState


class MemoryReviewStateTests(unittest.TestCase):
    def test_review_cursor_fields_are_declared_in_research_state(self):
        annotations = get_type_hints(
            ResearchState,
            include_extras=True,
        )

        self.assertIn("memory_review_turn_count", annotations)
        self.assertIn("last_reviewed_message_id", annotations)
        self.assertIn("last_archived_summary_hash", annotations)
        self.assertIn("memory_review_backlog_pending", annotations)


if __name__ == "__main__":
    unittest.main()
