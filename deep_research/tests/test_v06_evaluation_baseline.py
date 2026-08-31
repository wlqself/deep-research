import json
import unittest
from pathlib import Path


FIXTURE_PATH = (
    Path(__file__).with_name("v06_single_agent_eval_cases.json")
)

REQUIRED_METRICS = {
    "search_call_count",
    "page_read_count",
    "finding_count",
    "supported_finding_count",
    "conflicted_finding_count",
    "insufficient_finding_count",
    "unique_source_count",
    "invalid_source_reference_count",
    "summary_count",
    "elapsed_time",
}


class V06EvaluationBaselineTests(unittest.TestCase):
    def test_fixture_defines_stable_cases_and_metrics(self):
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], "v0.6")
        self.assertEqual(
            set(payload["required_metrics"]),
            REQUIRED_METRICS,
        )

        cases = payload["cases"]
        self.assertGreaterEqual(len(cases), 5)
        self.assertLessEqual(len(cases), 10)

        case_ids = [case["case_id"] for case in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))

        for case in cases:
            self.assertTrue(case["question"].strip())
            self.assertGreaterEqual(len(case["required_topics"]), 3)
            self.assertGreaterEqual(case["minimum_sources"], 2)
            self.assertGreaterEqual(case["minimum_page_reads"], 1)


if __name__ == "__main__":
    unittest.main()
