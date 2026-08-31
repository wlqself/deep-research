import unittest

from langchain.tools import ToolRuntime

from deep_research.context import ResearchContext
from deep_research.tools.list_findings import (
    list_research_findings,
)


def make_runtime(
    state: dict[str, object],
) -> ToolRuntime:
    return ToolRuntime(
        state=state,
        context=ResearchContext(
            max_search_calls=4,
            max_page_reads=6,
            max_page_chars=12000,
        ),
        config={},
        stream_writer=lambda _: None,
        tool_call_id="list-call",
        store=None,
    )


def finding(
    finding_id: str,
    claim: str,
    created_at: str,
    status: str = "supported",
) -> dict[str, object]:
    return {
        "finding_id": finding_id,
        "claim": claim,
        "evidence_summary": "Evidence",
        "source_ids": ["S1"],
        "status": status,
        "uncertainty": "",
        "conflicts": "",
        "created_at": created_at,
        "updated_at": created_at,
    }


class ListFindingTests(unittest.TestCase):
    def test_returns_empty_list_without_findings(self):
        result = list_research_findings.func(
            make_runtime({}),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["findings"],
            [],
        )

    def test_returns_stable_order(self):
        result = list_research_findings.func(
            make_runtime(
                {
                    "findings": {
                        "F2": finding(
                            "F2",
                            "Second",
                            "2026-08-21T12:00:00+00:00",
                        ),
                        "F1": finding(
                            "F1",
                            "First",
                            "2026-08-21T11:00:00+00:00",
                        ),
                    }
                }
            ),
        )

        self.assertEqual(
            [
                item["finding_id"]
                for item in result["findings"]
            ],
            ["F1", "F2"],
        )

    def test_filters_by_status(self):
        result = list_research_findings.func(
            make_runtime(
                {
                    "findings": {
                        "F1": finding(
                            "F1",
                            "Supported",
                            "2026-08-21T11:00:00+00:00",
                            "supported",
                        ),
                        "F2": finding(
                            "F2",
                            "Conflict",
                            "2026-08-21T12:00:00+00:00",
                            "conflicted",
                        ),
                    }
                }
            ),
            status="conflicted",
        )

        self.assertEqual(
            [
                item["finding_id"]
                for item in result["findings"]
            ],
            ["F2"],
        )

    def test_rejects_invalid_status(self):
        result = list_research_findings.func(
            make_runtime({}),
            status="unknown",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "invalid_status",
        )
        self.assertEqual(result["findings"], [])


if __name__ == "__main__":
    unittest.main()
