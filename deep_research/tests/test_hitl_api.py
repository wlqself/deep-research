import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_research.handlers.hitl import router
from deep_research.hitl.models import HITLAction, HITLStatus
from deep_research.hitl.repository import HITLRepository
from deep_research.hitl.service import HITLService


class HITLApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = HITLRepository(
            Path(self.temp_dir.name) / "hitl.sqlite"
        )
        self.repository.initialize()
        self.service = HITLService(self.repository)
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.state.hitl_service = self.service
        self.client = TestClient(self.app)

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def create_interaction(
        self,
        *,
        interaction_id="interaction-1",
        thread_id="thread-1",
        run_id="run-1",
        target_id="approval-1",
    ):
        return self.service.create_or_get_interaction(
            interaction_id=interaction_id,
            thread_id=thread_id,
            run_id=run_id,
            action=HITLAction.APPROVE,
            target_type="publication_approval",
            target_id=target_id,
            target_version=1,
        )

    def test_list_is_thread_scoped(self):
        self.create_interaction()
        self.create_interaction(
            interaction_id="interaction-2",
            thread_id="thread-2",
            run_id="run-2",
            target_id="approval-2",
        )

        response = self.client.get(
            "/hitl/interactions",
            params={"thread_id": "thread-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["interaction_id"] for item in response.json()],
            ["interaction-1"],
        )

    def test_list_reconciles_runtime_binding(self):
        interaction = self.create_interaction()
        self.service.approve(interaction.interaction_id, decision_actor="user")

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(
                    config={"configurable": {"checkpoint_id": "cp-1"}},
                    interrupts=(
                        SimpleNamespace(
                            id="interrupt-1",
                            value={
                                "kind": "publication_approval",
                                "interaction_id": "interaction-1",
                            },
                        ),
                    ),
                    tasks=(),
                )

        self.app.state.agent = FakeAgent()
        response = self.client.get(
            "/hitl/interactions",
            params={"thread_id": "thread-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["interrupt_id"], "interrupt-1")
        self.assertEqual(response.json()[0]["checkpoint_id"], "cp-1")

    def test_list_removes_resolved_interaction_with_missing_interrupt(self):
        interaction = self.create_interaction()
        self.service.approve(interaction.interaction_id, decision_actor="user")

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(interrupts=(), tasks=())

        self.app.state.agent = FakeAgent()
        response = self.client.get(
            "/hitl/interactions",
            params={"thread_id": "thread-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])
        self.assertEqual(
            self.service.get_interaction("interaction-1").status,
            HITLStatus.STALE,
        )

    def test_get_rejects_cross_thread_access_without_leaking_details(self):
        self.create_interaction()

        response = self.client.get(
            "/hitl/interactions/interaction-1",
            params={"thread_id": "thread-2"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            {"detail": {"error_code": "hitl_interaction_not_found"}},
        )

    def test_approve_is_explicit_and_records_user_actor(self):
        self.create_interaction()

        response = self.client.post(
            "/hitl/interactions/interaction-1/approve",
            params={"thread_id": "thread-1"},
            json={"decision_reason": "reviewed"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], HITLStatus.APPROVED.value)
        self.assertEqual(response.json()["decision_actor"], "user")

    def test_reject_requires_reason_and_then_changes_state(self):
        self.create_interaction()

        missing_reason = self.client.post(
            "/hitl/interactions/interaction-1/reject",
            params={"thread_id": "thread-1"},
            json={"decision_reason": ""},
        )
        self.assertEqual(missing_reason.status_code, 422)

        rejected = self.client.post(
            "/hitl/interactions/interaction-1/reject",
            params={"thread_id": "thread-1"},
            json={"decision_reason": "需要修改"},
        )
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(rejected.json()["status"], HITLStatus.REJECTED.value)

    def test_reject_without_reason_is_allowed(self):
        self.create_interaction()

        response = self.client.post(
            "/hitl/interactions/interaction-1/reject",
            params={"thread_id": "thread-1"},
            json={},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["decision_reason"])

    def test_repeated_decision_returns_stable_state_error(self):
        self.create_interaction()
        self.client.post(
            "/hitl/interactions/interaction-1/approve",
            params={"thread_id": "thread-1"},
            json={},
        )

        response = self.client.post(
            "/hitl/interactions/interaction-1/approve",
            params={"thread_id": "thread-1"},
            json={},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {"detail": {"error_code": "hitl_invalid_state_transition"}},
        )

    def test_unavailable_service_returns_safe_error(self):
        self.app.state.hitl_service = None

        response = self.client.get(
            "/hitl/interactions",
            params={"thread_id": "thread-1"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"detail": {"error_code": "hitl_service_unavailable"}},
        )

    def test_resume_transitions_approved_interaction_through_resuming(self):
        interaction = self.create_interaction()
        self.service.approve(
            interaction.interaction_id,
            decision_actor="user",
        )

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(
                    config={
                        "configurable": {
                            "checkpoint_id": "checkpoint-1",
                        },
                    },
                    values={},
                    interrupts=(
                        SimpleNamespace(
                            id="interrupt-1",
                            value={
                                "kind": "publication_approval",
                                "interaction_id": "interaction-1",
                            },
                        ),
                    ),
                    tasks=(),
                )

            async def ainvoke(self, value, *, config, context):
                return {
                    "messages": [
                        SimpleNamespace(content="审批已确认，Agent 已继续执行。"),
                    ],
                }

        self.app.state.agent = FakeAgent()

        response = self.client.post(
            "/hitl/interactions/interaction-1/resume",
            params={"thread_id": "thread-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["resumed"])
        self.assertEqual(
            self.service.get_interaction("interaction-1").status,
            HITLStatus.RESUMED,
        )
        persisted = self.service.get_interaction("interaction-1")
        self.assertEqual(persisted.interrupt_id, "interrupt-1")
        self.assertEqual(persisted.checkpoint_id, "checkpoint-1")

    def test_missing_graph_interrupt_marks_interaction_stale(self):
        interaction = self.create_interaction()
        self.service.approve(
            interaction.interaction_id,
            decision_actor="user",
        )

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(interrupts=(), tasks=())

        self.app.state.agent = FakeAgent()

        response = self.client.post(
            "/hitl/interactions/interaction-1/resume",
            params={"thread_id": "thread-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["resumed"])
        self.assertEqual(
            response.json()["error_code"],
            "hitl_interaction_stale",
        )
        self.assertEqual(
            self.service.get_interaction("interaction-1").status,
            HITLStatus.STALE,
        )

    def test_rejected_interaction_can_resume_agent(self):
        interaction = self.create_interaction()
        self.service.reject(
            interaction.interaction_id,
            decision_actor="user",
            decision_reason="需要修改",
        )

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(
                    values={},
                    interrupts=(
                        SimpleNamespace(
                            value={
                                "kind": "publication_approval",
                                "interaction_id": "interaction-1",
                            },
                        ),
                    ),
                    tasks=(),
                )

            async def ainvoke(self, value, *, config, context):
                return {"messages": [SimpleNamespace(content="已记录修改请求。明白。 ")]}

        self.app.state.agent = FakeAgent()

        response = self.client.post(
            "/hitl/interactions/interaction-1/resume",
            params={"thread_id": "thread-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["resumed"])
        self.assertEqual(
            self.service.get_interaction("interaction-1").status,
            HITLStatus.RESUMED,
        )


if __name__ == "__main__":
    unittest.main()
