import hashlib
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_research.handlers.hitl import router
from deep_research.hitl.decision_service import HITLDecisionService
from deep_research.hitl.models import HITLAction, HITLStatus
from deep_research.hitl.repository import HITLRepository
from deep_research.hitl.service import HITLService
from deep_research.publishing.artifacts import ArtifactSnapshot
from deep_research.publishing.models import PublicationChannel
from deep_research.publishing.repository import PublishingRepository
from deep_research.publishing.service import PublishingService


class FakeArtifactService:
    async def read(self, *, thread_id, artifact_id):
        content = f"# {artifact_id}\n\nBody."
        encoded = content.encode("utf-8")
        return ArtifactSnapshot(
            source_thread_id=thread_id,
            source_artifact_id=artifact_id,
            workspace_path=f"/final/{artifact_id}.md",
            filename=f"{artifact_id}.md",
            size_bytes=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
            markdown_content=content,
        )


class HITLDecisionServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)

        self.publishing_repository = PublishingRepository(
            root / "publishing.sqlite"
        )
        self.publishing_repository.initialize()
        self.publishing_service = PublishingService(
            self.publishing_repository,
            FakeArtifactService(),
        )

        self.hitl_repository = HITLRepository(root / "hitl.sqlite")
        self.hitl_repository.initialize()
        self.hitl_service = HITLService(self.hitl_repository)
        self.decision_service = HITLDecisionService(
            self.hitl_service,
            self.publishing_service,
        )

    def tearDown(self):
        self.hitl_repository.close()
        self.publishing_repository.close()
        self.temp_dir.cleanup()

    async def create_target(self, action=HITLAction.APPROVE):
        article = await self.publishing_service.create_article_from_artifact(
            thread_id="thread-1",
            artifact_id="artifact-1",
            title="文章 A",
            slug="article-a",
        )
        self.publishing_service.approve_article(article.article_id)
        approval = self.publishing_service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            actor="application",
        )
        interaction = self.hitl_service.create_or_get_interaction(
            thread_id="thread-1",
            run_id="run-1",
            action=action,
            target_type="publication_approval",
            target_id=approval.approval_id,
            target_version=approval.article_version,
        )
        return approval, interaction

    async def test_approve_synchronizes_both_domains(self):
        approval, interaction = await self.create_target()

        result = self.decision_service.approve_publication(
            interaction.interaction_id,
            thread_id="thread-1",
            decision_actor="user",
        )

        self.assertEqual(result.interaction.status, HITLStatus.APPROVED)
        self.assertEqual(result.publication_approval.status.value, "approved")
        self.assertEqual(
            self.publishing_repository.get_approval_request(
                approval.approval_id,
            ).status.value,
            "approved",
        )

    async def test_repeated_approve_reconciles_without_duplicate_decision(self):
        _, interaction = await self.create_target()

        first = self.decision_service.approve_publication(
            interaction.interaction_id,
            thread_id="thread-1",
            decision_actor="user",
        )
        replay = self.decision_service.approve_publication(
            interaction.interaction_id,
            thread_id="thread-1",
            decision_actor="user",
        )

        self.assertEqual(
            first.interaction.interaction_id,
            replay.interaction.interaction_id,
        )
        self.assertEqual(replay.interaction.status, HITLStatus.APPROVED)
        self.assertEqual(replay.publication_approval.status.value, "approved")

    async def test_api_approve_uses_the_bridge_and_returns_both_states(self):
        _, interaction = await self.create_target()
        app = FastAPI()
        app.include_router(router)
        app.state.hitl_service = self.hitl_service
        app.state.hitl_decision_service = self.decision_service

        response = TestClient(app).post(
            f"/hitl/interactions/{interaction.interaction_id}/approve",
            params={"thread_id": "thread-1"},
            json={},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "approved")
        self.assertEqual(
            response.json()["publication_approval_status"],
            "approved",
        )

    async def test_reject_synchronizes_both_domains_without_reason(self):
        approval, interaction = await self.create_target(HITLAction.REJECT)

        result = self.decision_service.reject_publication(
            interaction.interaction_id,
            thread_id="thread-1",
            decision_actor="user",
        )

        self.assertEqual(result.interaction.status, HITLStatus.REJECTED)
        self.assertEqual(result.publication_approval.status.value, "rejected")
        self.assertIsNone(result.interaction.decision_reason)
        self.assertEqual(
            self.publishing_repository.get_approval_request(
                approval.approval_id,
            ).status.value,
            "rejected",
        )

    async def test_api_reject_uses_the_bridge_and_is_replayable(self):
        _, interaction = await self.create_target(HITLAction.REJECT)
        app = FastAPI()
        app.include_router(router)
        app.state.hitl_service = self.hitl_service
        app.state.hitl_decision_service = self.decision_service
        client = TestClient(app)

        response = client.post(
            f"/hitl/interactions/{interaction.interaction_id}/reject",
            params={"thread_id": "thread-1"},
            json={},
        )
        replay = client.post(
            f"/hitl/interactions/{interaction.interaction_id}/reject",
            params={"thread_id": "thread-1"},
            json={},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(response.json()["status"], "rejected")
        self.assertEqual(
            replay.json()["publication_approval_status"],
            "rejected",
        )

    async def test_request_changes_rejects_an_approval_card_interaction(self):
        approval, interaction = await self.create_target(HITLAction.APPROVE)

        result = self.decision_service.reject_publication(
            interaction.interaction_id,
            thread_id="thread-1",
            decision_actor="user",
            decision_reason="第二段需要重新组织",
        )

        self.assertEqual(result.interaction.action, HITLAction.APPROVE)
        self.assertEqual(result.interaction.status, HITLStatus.REJECTED)
        self.assertEqual(
            result.interaction.decision_reason,
            "第二段需要重新组织",
        )
        self.assertEqual(result.publication_approval.status.value, "rejected")
        self.assertEqual(
            self.publishing_repository.get_approval_request(
                approval.approval_id,
            ).status.value,
            "rejected",
        )

    async def test_api_request_changes_uses_reject_endpoint_for_approval_card(self):
        _, interaction = await self.create_target(HITLAction.APPROVE)
        app = FastAPI()
        app.include_router(router)
        app.state.hitl_service = self.hitl_service
        app.state.hitl_decision_service = self.decision_service

        response = TestClient(app).post(
            f"/hitl/interactions/{interaction.interaction_id}/reject",
            params={"thread_id": "thread-1"},
            json={"decision_reason": "请重新组织第二段"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "rejected")
        self.assertEqual(
            response.json()["publication_approval_status"],
            "rejected",
        )


if __name__ == "__main__":
    unittest.main()
