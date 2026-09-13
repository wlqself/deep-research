import tempfile
import unittest
from pathlib import Path
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from deep_research.agent.hitl_resume import resume_agent_after_hitl
from deep_research.hitl.models import HITLAction, HITLStatus
from deep_research.hitl.repository import HITLRepository
from deep_research.hitl.service import HITLService


class _ApprovalState(TypedDict, total=False):
    decision: object


def _build_graph(checkpointer):
    def approval_node(state: _ApprovalState):
        return {
            "decision": interrupt(
                {
                    "kind": "publication_approval",
                    "interaction_id": "restart-interaction",
                }
            )
        }

    builder = StateGraph(_ApprovalState)
    builder.add_node("approval", approval_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=checkpointer)


class HITLRestartTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_and_hitl_record_survive_restart_and_resume_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoints.sqlite"
            hitl_path = root / "hitl.sqlite"
            config = {"configurable": {"thread_id": "restart-thread"}}

            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
                graph = _build_graph(saver)
                first = await graph.ainvoke({}, config)
                self.assertIn("__interrupt__", first)
                snapshot = await graph.aget_state(config)
                native_interrupt = snapshot.interrupts[0]
                checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]

            hitl_repository = HITLRepository(hitl_path)
            hitl_repository.initialize()
            hitl_service = HITLService(hitl_repository)
            interaction = hitl_service.create_or_get_interaction(
                interaction_id="restart-interaction",
                thread_id="restart-thread",
                run_id="restart-run",
                action=HITLAction.APPROVE,
                target_type="publication_approval",
                target_id="approval-restart",
                target_version=1,
            )
            hitl_service.bind_runtime(
                interaction.interaction_id,
                interrupt_id=native_interrupt.id,
                checkpoint_id=checkpoint_id,
            )
            hitl_service.approve(
                interaction.interaction_id,
                decision_actor="user",
            )
            hitl_repository.close()

            # New graph/checkpointer and new HITL repository objects represent
            # a process restart, not merely a second request in one process.
            reopened_repository = HITLRepository(hitl_path)
            reopened_repository.initialize()
            reopened_service = HITLService(reopened_repository)
            restored = reopened_service.get_interaction(interaction.interaction_id)
            self.assertEqual(restored.status, HITLStatus.APPROVED)
            self.assertEqual(restored.interrupt_id, native_interrupt.id)
            self.assertEqual(restored.checkpoint_id, checkpoint_id)

            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
                graph = _build_graph(saver)
                resumed = await resume_agent_after_hitl(
                    graph,
                    thread_id="restart-thread",
                    interaction_id=restored.interaction_id,
                    decision="approved",
                    expected_interrupt_id=restored.interrupt_id,
                )

            self.assertTrue(resumed["resumed"])
            self.assertEqual(resumed["interrupt_id"], native_interrupt.id)
            reopened_service.start_resume(restored.interaction_id)
            reopened_service.mark_resumed(restored.interaction_id)
            self.assertEqual(
                reopened_service.get_interaction(restored.interaction_id).status,
                HITLStatus.RESUMED,
            )
            reopened_repository.close()


if __name__ == "__main__":
    unittest.main()
