import unittest

from deep_research.rag.retrieval_text import (
    build_metadata_weighted_embedding_text,
    restore_original_content,
)


class RetrievalTextTests(unittest.TestCase):
    def test_metadata_is_repeated_for_embedding(self):
        content = "answer content"
        embedded = build_metadata_weighted_embedding_text(
            content,
            {
                "title": "CVIU paper",
                "filename": "CVIU.pdf",
                "section_title": "Conclusion",
            },
            weight=2,
        )
        self.assertGreaterEqual(embedded.count("CVIU.pdf"), 2)
        self.assertEqual(restore_original_content(embedded), content)

    def test_empty_metadata_keeps_content_unchanged(self):
        content = "plain content"
        self.assertEqual(
            build_metadata_weighted_embedding_text(content, {}, weight=2),
            content,
        )
        self.assertEqual(restore_original_content(content), content)


if __name__ == "__main__":
    unittest.main()
