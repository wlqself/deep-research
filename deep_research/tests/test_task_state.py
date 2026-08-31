import copy
import tempfile
import unittest
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deep_research.state import ResearchState
from deep_research.state.research import TaskRecord, merge_tasks


def task_record(
    task_id: str,
    *,
    status: str = "pending",
    updated_at: str = "2026-08-22T12:00:00.000Z",
    finding_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
) -> TaskRecord:
    return {
        "task_id": task_id,
        "agent_name": "researcher",
        "description": f"Research {task_id}",
        "status": status,
        "created_at": "2026-08-22T12:00:00.000Z",
        "started_at": (
            "2026-08-22T12:01:00.000Z"
            if status != "pending"
            else ""
        ),
        "completed_at": (
            "2026-08-22T12:02:00.000Z"
            if status in {"completed", "failed", "cancelled"}
            else ""
        ),
        "updated_at": updated_at,
        "finding_ids": finding_ids or [],
        "source_ids": source_ids or [],
        "error": (
            "researcher_failed"
            if status == "failed"
            else ""
        ),
    }


class TaskStateTests(unittest.TestCase):
    def test_merges_independent_tasks(self):
        merged = merge_tasks(
            {"task-A": task_record("task-A")},
            {"task-B": task_record("task-B")},
        )

        self.assertEqual(set(merged), {"task-A", "task-B"})

    def test_updates_one_task_through_lifecycle(self):
        pending = task_record(
            "task-A",
            status="pending",
            updated_at="2026-08-22T12:00:00.000Z",
        )
        running = task_record(
            "task-A",
            status="running",
            updated_at="2026-08-22T12:01:00.000Z",
        )
        completed = task_record(
            "task-A",
            status="completed",
            updated_at="2026-08-22T12:02:00.000Z",
            finding_ids=["F1"],
            source_ids=["S1"],
        )

        merged = merge_tasks({"task-A": pending}, {"task-A": running})
        merged = merge_tasks(merged, {"task-A": completed})

        self.assertEqual(merged["task-A"]["status"], "completed")
        self.assertEqual(merged["task-A"]["finding_ids"], ["F1"])
        self.assertEqual(merged["task-A"]["source_ids"], ["S1"])

    def test_terminal_status_cannot_regress_to_running(self):
        for status in ("completed", "failed"):
            with self.subTest(status=status):
                terminal = task_record(
                    "task-A",
                    status=status,
                    updated_at="2026-08-22T12:02:00.000Z",
                )
                stale_running = task_record(
                    "task-A",
                    status="running",
                    updated_at="2026-08-22T12:03:00.000Z",
                )

                merged = merge_tasks(
                    {"task-A": terminal},
                    {"task-A": stale_running},
                )

                self.assertEqual(merged["task-A"]["status"], status)
                self.assertEqual(
                    merged["task-A"]["updated_at"],
                    "2026-08-22T12:02:00.000Z",
                )

    def test_newer_terminal_record_wins(self):
        completed = task_record(
            "task-A",
            status="completed",
            updated_at="2026-08-22T12:02:00.000Z",
        )
        failed = task_record(
            "task-A",
            status="failed",
            updated_at="2026-08-22T12:03:00.000Z",
        )

        merged = merge_tasks(
            {"task-A": completed},
            {"task-A": failed},
        )

        self.assertEqual(merged["task-A"]["status"], "failed")
        self.assertEqual(
            merged["task-A"]["updated_at"],
            "2026-08-22T12:03:00.000Z",
        )

    def test_reducer_does_not_mutate_inputs_or_id_lists(self):
        current = {
            "task-A": task_record(
                "task-A",
                finding_ids=["F1"],
                source_ids=["S1"],
            )
        }
        update = {
            "task-B": task_record(
                "task-B",
                finding_ids=["F2"],
                source_ids=["S2"],
            )
        }
        current_before = copy.deepcopy(current)
        update_before = copy.deepcopy(update)

        merged = merge_tasks(current, update)

        self.assertEqual(current, current_before)
        self.assertEqual(update, update_before)
        self.assertIsNot(merged, current)
        self.assertIsNot(
            merged["task-A"]["finding_ids"],
            current["task-A"]["finding_ids"],
        )
        self.assertIsNot(
            merged["task-B"]["source_ids"],
            update["task-B"]["source_ids"],
        )


class TaskStatePersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_task_survives_sqlite_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(Path(temp_dir) / "tasks.sqlite")
            config = {
                "configurable": {
                    "thread_id": "task-restart-thread",
                }
            }

            async with AsyncSqliteSaver.from_conn_string(
                database_path
            ) as first_saver:
                first_graph = create_agent(
                    model=GenericFakeChatModel(messages=iter([])),
                    tools=[],
                    state_schema=ResearchState,
                    checkpointer=first_saver,
                )
                await first_graph.aupdate_state(
                    config,
                    {
                        "tasks": {
                            "task-run-1": task_record(
                                "task-run-1",
                                status="completed",
                                updated_at="2026-08-23T12:02:00.000Z",
                                finding_ids=["F1"],
                                source_ids=["S1"],
                            )
                        }
                    },
                )

            async with AsyncSqliteSaver.from_conn_string(
                database_path
            ) as second_saver:
                second_graph = create_agent(
                    model=GenericFakeChatModel(messages=iter([])),
                    tools=[],
                    state_schema=ResearchState,
                    checkpointer=second_saver,
                )
                state = await second_graph.aget_state(config)

        task = state.values["tasks"]["task-run-1"]
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["finding_ids"], ["F1"])
        self.assertEqual(task["source_ids"], ["S1"])

    async def test_cancelled_task_and_committed_research_survive_sqlite_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(Path(temp_dir) / "cancelled-tasks.sqlite")
            config = {
                "configurable": {
                    "thread_id": "cancelled-task-restart-thread",
                }
            }
            source = {
                "source_id": "S1",
                "title": "Committed source",
                "url": "https://example.com/committed",
                "snippet": "committed before cancellation",
            }
            finding = {
                "finding_id": "F1",
                "claim": "Committed claim",
                "evidence_summary": "committed before cancellation",
                "source_ids": ["S1"],
                "status": "supported",
                "uncertainty": "",
                "conflicts": "",
                "created_at": "2026-08-23T12:00:00Z",
                "updated_at": "2026-08-23T12:01:00Z",
            }
            cancelled_task = task_record(
                "task-run-cancelled",
                status="cancelled",
                updated_at="2026-08-23T12:02:00.000Z",
                finding_ids=["F1"],
                source_ids=["S1"],
            )
            cancelled_task["error"] = "user_cancelled"

            async with AsyncSqliteSaver.from_conn_string(
                database_path
            ) as first_saver:
                first_graph = create_agent(
                    model=GenericFakeChatModel(messages=iter([])),
                    tools=[],
                    state_schema=ResearchState,
                    checkpointer=first_saver,
                )
                await first_graph.aupdate_state(
                    config,
                    {
                        "sources": {"S1": source},
                        "next_source_number": 2,
                        "findings": {"F1": finding},
                        "tasks": {
                            "task-run-cancelled": cancelled_task,
                        },
                    },
                )

            async with AsyncSqliteSaver.from_conn_string(
                database_path
            ) as second_saver:
                second_graph = create_agent(
                    model=GenericFakeChatModel(messages=iter([])),
                    tools=[],
                    state_schema=ResearchState,
                    checkpointer=second_saver,
                )
                state = await second_graph.aget_state(config)

        self.assertEqual(
            state.values["tasks"]["task-run-cancelled"]["status"],
            "cancelled",
        )
        self.assertEqual(
            state.values["tasks"]["task-run-cancelled"]["error"],
            "user_cancelled",
        )
        self.assertEqual(state.values["sources"]["S1"], source)
        self.assertEqual(state.values["findings"]["F1"], finding)
        self.assertEqual(state.values["next_source_number"], 2)


if __name__ == "__main__":
    unittest.main()
