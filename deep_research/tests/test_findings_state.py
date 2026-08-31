import tempfile
import unittest
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deep_research.state import ResearchState
from deep_research.state.research import merge_findings


def finding(
    finding_id: str,
    claim: str,
) -> dict[str, object]:
    return {
        "finding_id": finding_id,
        "claim": claim,
        "evidence_summary": "Test evidence",
        "source_ids": ["S1"],
        "status": "supported",
        "uncertainty": "",
        "conflicts": "",
        "created_at": "2026-08-21T12:00:00+00:00",
        "updated_at": "2026-08-21T12:00:00+00:00",
    }


class FindingsStateTests(unittest.IsolatedAsyncioTestCase):
    def test_reducer_merges_new_finding(self):
        current = {"F1": finding("F1", "First claim")}
        update = {"F2": finding("F2", "Second claim")}

        merged = merge_findings(current, update)

        self.assertEqual(set(merged), {"F1", "F2"})
        self.assertEqual(merged["F2"]["claim"], "Second claim")

    def test_reducer_updates_same_finding_id(self):
        current = {"F1": finding("F1", "Old claim")}
        update = {"F1": finding("F1", "Updated claim")}

        merged = merge_findings(current, update)

        self.assertEqual(merged["F1"]["claim"], "Updated claim")

    def test_reducer_does_not_mutate_inputs(self):
        current = {"F1": finding("F1", "First claim")}
        update = {"F2": finding("F2", "Second claim")}
        current_before = dict(current)
        update_before = dict(update)

        merged = merge_findings(current, update)

        self.assertEqual(current, current_before)
        self.assertEqual(update, update_before)
        self.assertIsNot(merged, current)
        self.assertIsNot(merged, update)

    def test_reducer_treats_none_as_empty_dict(self):
        current = {"F1": finding("F1", "First claim")}
        update = {"F2": finding("F2", "Second claim")}

        self.assertEqual(merge_findings(None, None), {})
        self.assertEqual(merge_findings(None, update), update)
        self.assertEqual(merge_findings(current, None), current)

    def test_reducer_merges_multiple_findings(self):
        current = {
            "F1": finding("F1", "Original claim"),
            "F2": finding("F2", "Second claim"),
        }
        update = {
            "F1": finding("F1", "Updated claim"),
            "F3": finding("F3", "Third claim"),
            "F4": finding("F4", "Fourth claim"),
        }

        merged = merge_findings(current, update)

        self.assertEqual(set(merged), {"F1", "F2", "F3", "F4"})
        self.assertEqual(merged["F1"]["claim"], "Updated claim")
        self.assertEqual(merged["F2"]["claim"], "Second claim")
        self.assertEqual(merged["F3"]["claim"], "Third claim")
        self.assertEqual(merged["F4"]["claim"], "Fourth claim")

    async def test_findings_are_persisted_and_isolated_by_thread(self):
        graph = create_agent(
            model=GenericFakeChatModel(messages=iter([])),
            tools=[],
            state_schema=ResearchState,
            checkpointer=InMemorySaver(),
        )
        thread_a = {"configurable": {"thread_id": "findings-thread-a"}}
        thread_b = {"configurable": {"thread_id": "findings-thread-b"}}

        await graph.aupdate_state(
            thread_a,
            {"findings": {"F1": finding("F1", "Thread A claim")}},
        )

        state_a = await graph.aget_state(thread_a)
        state_b = await graph.aget_state(thread_b)

        self.assertEqual(
            state_a.values["findings"]["F1"]["claim"],
            "Thread A claim",
        )
        self.assertNotIn("findings", state_b.values)

    async def test_findings_survive_sqlite_saver_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(Path(temp_dir) / "findings.sqlite")
            config = {
                "configurable": {
                    "thread_id": "findings-restart-thread",
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
                        "findings": {
                            "F1": finding("F1", "Persisted claim")
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

        self.assertEqual(
            state.values["findings"]["F1"]["claim"],
            "Persisted claim",
        )


if __name__ == "__main__":
    unittest.main()
