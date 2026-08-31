import unittest

from deep_research.handlers.threads import _serialize_tasks


def task(status: str, completed_at: str) -> dict[str, object]:
    return {
        "agent_name": "researcher",
        "description": "Research task",
        "status": status,
        "completed_at": completed_at,
        "error": "task_execution_failed" if status == "failed" else "",
    }


class ThreadSnapshotTaskTests(unittest.TestCase):
    def test_serializes_only_terminal_tasks_in_completion_order(self):
        tasks = _serialize_tasks(
            {
                "tasks": {
                    "task-running": task("running", ""),
                    "task-completed": task(
                        "completed",
                        "2026-08-23T12:02:00Z",
                    ),
                    "task-failed": task(
                        "failed",
                        "2026-08-23T12:01:00Z",
                    ),
                    "task-cancelled": task(
                        "cancelled",
                        "2026-08-23T12:03:00Z",
                    ),
                }
            }
        )

        self.assertEqual(
            [task["task_id"] for task in tasks],
            ["task-failed", "task-completed", "task-cancelled"],
        )
        self.assertEqual(
            [task["status"] for task in tasks],
            ["failed", "completed", "cancelled"],
        )

    def test_rejects_invalid_task_values(self):
        tasks = _serialize_tasks(
            {
                "tasks": {
                    "task-invalid-status": task("running", ""),
                    "task-invalid-value": "not-a-record",
                    123: task("completed", "2026-08-23T12:00:00Z"),
                }
            }
        )

        self.assertEqual(tasks, [])


if __name__ == "__main__":
    unittest.main()
