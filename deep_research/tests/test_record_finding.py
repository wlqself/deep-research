import unittest
import json

from langgraph.types import Command

from deep_research.tools.record_finding import (
    MAX_EVIDENCE_SUMMARY_CHARS,
    record_research_finding,
)
from langchain.tools import ToolRuntime

from deep_research.context import ResearchContext
from deep_research.tools.record_finding import (
    record_research_finding,
)


def make_runtime(
    state: dict[str, object] | None = None,
    tool_call_id: str | None = "test-call",
) -> ToolRuntime:
    return ToolRuntime(
        state=state or {},
        context=ResearchContext(
            max_search_calls=4,
            max_page_reads=6,
            max_page_chars=12000,
        ),
        config={},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=None,
    )


class RecordFindingTests(unittest.TestCase):

    def known_runtime(self):
        return make_runtime(
            {
                "sources": {
                    "S1": {
                        "source_id": "S1",
                        "title": "Known source",
                        "url": "https://example.com",
                        "snippet": "Known snippet",
                    }
                }
            }
        )
    
    def test_rejects_empty_claim(self):
        result = record_research_finding.func(
            "   ",
            "some evidence",
            ["S1"],
            "supported",
            "",
            "",
            make_runtime(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "invalid_claim",
        )

    def test_rejects_empty_evidence_summary(self):
        result = record_research_finding.func(
            "A claim",
            "   ",
            ["S1"],
            "supported",
            "",
            "",
            make_runtime(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "invalid_evidence_summary",
        )

    def test_rejects_non_string_source_ids(self):
        result = record_research_finding.func(
            "A claim",
            "Some evidence",
            ["S1", 123],
            "supported",
            "",
            "",
            make_runtime(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "invalid_source_ids",
        )

    def test_rejects_source_ids_that_are_not_a_list(self):
        result = record_research_finding.func(
            "A claim",
            "Some evidence",
            "S1",
            "supported",
            "",
            "",
            make_runtime(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "invalid_source_ids",
        )

    def test_rejects_unknown_source_id(self):
        result = record_research_finding.func(
            "A claim",
            "Some evidence",
            ["S99"],
            "supported",
            "",
            "",
            make_runtime(
                {
                    "sources": {
                        "S1": {
                            "source_id": "S1",
                            "title": "Known source",
                            "url": "https://example.com",
                            "snippet": "Known snippet",
                        }
                    }
                }
            ),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "unknown_source_id",
        )
        self.assertIn(
            "S99",
            result["message"],
        )
        self.assertNotIn(
            "Known source",
            result["message"],
        )

    def test_rejects_duplicate_source_ids(self):
        result = record_research_finding.func(
            "A claim",
            "Some evidence",
            ["S1", "S1"],
            "supported",
            "",
            "",
            make_runtime(
                {
                    "sources": {
                        "S1": {
                            "source_id": "S1",
                            "title": "Known source",
                            "url": "https://example.com",
                            "snippet": "Known snippet",
                        }
                    }
                }
            ),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "duplicate_source_id",
        )

    def test_rejects_empty_source_ids(self):
        result = record_research_finding.func(
            "A claim",
            "Some evidence",
            [],
            "supported",
            "",
            "",
            self.known_runtime(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "empty_source_ids",
        )

    def test_rejects_invalid_status(self):
        result = record_research_finding.func(
            "A claim",
            "Some evidence",
            ["S1"],
            "unknown",
            "",
            "",
            self.known_runtime(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "invalid_status",
        )

    def test_rejects_conflicted_without_conflicts(self):
        result = record_research_finding.func(
            "A claim",
            "Some evidence",
            ["S1"],
            "conflicted",
            "",
            "",
            self.known_runtime(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "missing_conflicts",
        )

    def test_rejects_long_evidence_summary(self):
        result = record_research_finding.func(
            "A claim",
            "x" * (MAX_EVIDENCE_SUMMARY_CHARS + 1),
            ["S1"],
            "supported",
            "",
            "",
            self.known_runtime(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "evidence_summary_too_long",
        )

    def test_rejects_missing_tool_call_id(self):
        runtime = make_runtime(
            state=self.known_runtime().state,
            tool_call_id=None,
        )

        result = record_research_finding.func(
            "A claim",
            "Some evidence",
            ["S1"],
            "supported",
            "",
            "",
            runtime,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"],
            "missing_tool_call_id",
        )

    def test_records_finding_with_command(self):
        runtime = self.known_runtime()

        result = record_research_finding.func(
            "A supported claim",
            "Evidence from the source.",
            ["S1"],
            "supported",
            "",
            "",
            runtime,
        )

        self.assertIsInstance(result, Command)

        self.assertEqual(
            runtime.state["sources"]["S1"]["source_id"],
            "S1",
        )

        findings = result.update["findings"]

        self.assertEqual(
            len(findings),
            1,
        )

        finding_id = next(iter(findings))
        saved = findings[finding_id]

        self.assertEqual(
            saved["finding_id"],
            finding_id,
        )
        self.assertEqual(
            saved["claim"],
            "A supported claim",
        )
        self.assertEqual(
            saved["source_ids"],
            ["S1"],
        )
        self.assertEqual(
            saved["status"],
            "supported",
        )
        self.assertEqual(
            saved["uncertainty"],
            "",
        )
        self.assertTrue(
            saved["created_at"].endswith("+00:00")
        )
        self.assertEqual(
            saved["created_at"],
            saved["updated_at"],
        )

        message = result.update["messages"][0]

        self.assertEqual(
            message.tool_call_id,
            "test-call",
        )

        payload = json.loads(message.content)

        self.assertTrue(payload["ok"])
        self.assertEqual(
            payload["finding_id"],
            finding_id,
        )
        
if __name__ == "__main__":
    unittest.main()
