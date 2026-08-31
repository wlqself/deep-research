import unittest

from deep_research.citations.formatting import (
    append_verified_sources,
    invalid_source_ids,
)


SOURCE = {
    "source_id": "S1",
    "title": "真实来源",
    "url": "https://example.com/source",
    "snippet": "source",
}


class CitationValidationTests(unittest.TestCase):
    def test_invalid_source_id_is_detected(self):
        invalid = invalid_source_ids(
            "结论 [S1]，错误引用 [S99]",
            {"S1": SOURCE},
        )

        self.assertEqual(
            invalid,
            ["S99"],
        )

    def test_only_thread_sources_are_appended(self):
        answer = append_verified_sources(
            "结论 [S1]",
            {"S1": SOURCE},
        )

        self.assertIn(
            "[S1] 真实来源 - https://example.com/source",
            answer,
        )

        self.assertNotIn(
            "S99",
            answer,
        )

    def test_sources_heading_is_not_appended_twice(self):
        sources = {"S1": SOURCE}

        answer = append_verified_sources(
            "结论 [S1]",
            sources,
        )

        repeated = append_verified_sources(
            answer,
            sources,
        )

        self.assertEqual(
            repeated,
            answer,
        )


if __name__ == "__main__":
    unittest.main()