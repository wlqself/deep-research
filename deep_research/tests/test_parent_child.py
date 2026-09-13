import unittest

from deep_research.evaluation.parent_child import (
    build_metadata_weighted_text,
    build_parent_child_view,
    build_reranker_text,
    collapse_child_rankings,
)


class ParentChildTests(unittest.TestCase):
    def make_rows(self):
        return [
            {
                "_id": "c2",
                "title": "Doc",
                "text": "second",
                "metadata": {
                    "document_id": "doc-1",
                    "filename": "cviu.pdf",
                    "chunk_index": 1,
                },
            },
            {
                "_id": "c1",
                "title": "Doc",
                "text": "first",
                "metadata": {
                    "document_id": "doc-1",
                    "filename": "cviu.pdf",
                    "chunk_index": 0,
                },
            },
        ]

    def test_parent_windows_are_stable_and_children_keep_ids(self):
        children, parents, mapping = build_parent_child_view(
            self.make_rows(),
            parent_size=2,
        )
        self.assertEqual([row["_id"] for row in children], ["c1", "c2"])
        self.assertEqual(list(parents), ["doc-1:parent:0"])
        self.assertEqual(parents["doc-1:parent:0"]["text"], "first\n\nsecond")
        self.assertEqual(mapping["c1"], mapping["c2"])

    def test_collapse_deduplicates_parent_hits(self):
        _children, _parents, mapping = build_parent_child_view(
            self.make_rows(),
            parent_size=1,
        )
        self.assertEqual(
            collapse_child_rankings(["c1", "c2"], mapping, top_k=2),
            ["doc-1:parent:0", "doc-1:parent:1"],
        )

    def test_metadata_and_parent_context_are_prefixed(self):
        text = build_metadata_weighted_text(
            self.make_rows()[0],
            metadata_weight=2,
            parent_text="parent context",
        )
        self.assertGreaterEqual(text.count("cviu.pdf"), 2)
        self.assertIn("parent context", text)
        self.assertTrue(text.endswith("second"))

    def test_reranker_text_labels_parent_metadata_without_repeating_it(self):
        _children, parents, _mapping = build_parent_child_view(
            self.make_rows(),
            parent_size=2,
        )
        text = build_reranker_text(parents["doc-1:parent:0"])

        self.assertIn("[文档标题] Doc", text)
        self.assertIn("[文件名] cviu.pdf", text)
        self.assertIn("[正文]", text)
        self.assertEqual(text.count("cviu.pdf"), 1)


if __name__ == "__main__":
    unittest.main()
