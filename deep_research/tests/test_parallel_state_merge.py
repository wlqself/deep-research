import unittest

from deep_research.state.research import (
    TaskRecord,
    max_source_number,
    merge_findings,
    merge_sources,
    merge_tasks,
)


def source(source_id: str) -> dict[str, str]:
    return {
        "source_id": source_id,
        "title": f"Source {source_id}",
        "url": f"https://example.com/{source_id}",
        "snippet": f"Snippet {source_id}",
    }


def finding(finding_id: str, source_id: str) -> dict[str, object]:
    return {
        "finding_id": finding_id,
        "claim": f"Claim {finding_id}",
        "evidence_summary": "Test evidence",
        "source_ids": [source_id],
        "status": "supported",
        "uncertainty": "",
        "conflicts": "",
        "created_at": "2026-08-23T12:00:00Z",
        "updated_at": "2026-08-23T12:00:00Z",
    }


def task(task_id: str, status: str, source_ids: list[str]) -> TaskRecord:
    return {
        "task_id": task_id,
        "agent_name": "researcher",
        "description": f"Research {task_id}",
        "status": status,
        "created_at": "2026-08-23T12:00:00Z",
        "started_at": "2026-08-23T12:00:01Z",
        "completed_at": "2026-08-23T12:00:02Z",
        "updated_at": "2026-08-23T12:00:02Z",
        "finding_ids": [],
        "source_ids": source_ids,
        "error": "task_execution_failed" if status == "failed" else "",
    }


class ParallelStateMergeTests(unittest.TestCase):
    def test_two_independent_task_updates_merge_without_loss(self):
        task_a = {
            "sources": {"S1": source("S1"), "S3": source("S3")},
            "findings": {"FA": finding("FA", "S1")},
            "tasks": {"task-A": task("task-A", "completed", ["S1", "S3"])},
            "next_source_number": 4,
        }
        task_b = {
            "sources": {"S2": source("S2"), "S4": source("S4")},
            "findings": {"FB": finding("FB", "S2")},
            "tasks": {"task-B": task("task-B", "completed", ["S2", "S4"])},
            "next_source_number": 5,
        }

        sources = merge_sources(task_a["sources"], task_b["sources"])
        findings = merge_findings(task_a["findings"], task_b["findings"])
        tasks = merge_tasks(task_a["tasks"], task_b["tasks"])
        next_number = max_source_number(
            task_a["next_source_number"],
            task_b["next_source_number"],
        )

        self.assertEqual(set(sources), {"S1", "S2", "S3", "S4"})
        self.assertEqual(set(findings), {"FA", "FB"})
        self.assertEqual(set(tasks), {"task-A", "task-B"})
        self.assertEqual(next_number, 5)

    def test_failed_task_does_not_remove_other_task_results(self):
        successful_sources = {"S2": source("S2"), "S4": source("S4")}
        successful_findings = {"FB": finding("FB", "S2")}
        merged_sources = merge_sources(
            successful_sources,
            {},
        )
        merged_findings = merge_findings(
            successful_findings,
            {},
        )
        merged_tasks = merge_tasks(
            {"task-B": task("task-B", "completed", ["S2", "S4"])},
            {"task-A": task("task-A", "failed", [])},
        )

        self.assertEqual(set(merged_sources), {"S2", "S4"})
        self.assertEqual(set(merged_findings), {"FB"})
        self.assertEqual(merged_tasks["task-A"]["status"], "failed")
        self.assertEqual(merged_tasks["task-B"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
