import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deep_research.publishing.artifacts import ArtifactSnapshot
from deep_research.publishing.intent_resolver import (
    PublishingIntentResolutionStatus,
    PublishingIntentResolver,
)
from deep_research.publishing.intents import PublishingIntent
from deep_research.publishing.models import PublicationChannel
from deep_research.publishing.models import PublicationStatus
from deep_research.publishing.service import (
    PublicationDeliveryUnknownError,
    PublicationExecutionError,
    PublicationResult,
)
from deep_research.publishing.repository import PublishingRepository
from deep_research.publishing.service import PublishingService
from deep_research.publishing.attachments import ImageAttachmentService
from deep_research.hitl.repository import HITLRepository
from deep_research.hitl.service import HITLService
from deep_research.agent.tool_events import (
    publishing_approval_status_preview,
    publishing_interrupt_preview,
)
from deep_research.tools.publishing import (
    build_publication_approval_status_tool,
    build_request_publication_approval_tool,
    build_resolve_publication_intent_tool,
)
from deep_research.hitl.models import HITLAction


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


class FailingWeChatPublisher:
    channel = PublicationChannel.WECHAT_OFFICIAL_ACCOUNT

    def publish(self, article):
        raise PublicationExecutionError("wechat_title_too_long")


class UnknownWeChatPublisher:
    channel = PublicationChannel.WECHAT_OFFICIAL_ACCOUNT

    def publish(self, article):
        raise PublicationDeliveryUnknownError(
            "wechat_publish_request_failed",
            external_id="draft-media-1",
        )


class PublishingIntentTests(unittest.IsolatedAsyncioTestCase):
    THREAD_ID = "thread-intent"

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = PublishingRepository(
            Path(self.temp_dir.name) / "publishing.sqlite"
        )
        self.repository.initialize()
        self.service = PublishingService(
            self.repository,
            FakeArtifactService(),
        )
        self.hitl_repository = HITLRepository(
            Path(self.temp_dir.name) / "hitl.sqlite"
        )
        self.hitl_repository.initialize()
        self.hitl_service = HITLService(self.hitl_repository)

    def tearDown(self):
        self.hitl_repository.close()
        self.repository.close()
        self.temp_dir.cleanup()

    async def create_pending_approval(
        self,
        title,
        artifact_id,
        channel=PublicationChannel.LOCAL_STATIC_SITE,
    ):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id=artifact_id,
            title=title,
            slug=title.casefold().replace(" ", "-"),
        )
        self.service.approve_article(article.article_id)
        return self.service.request_publication_approval(
            article.article_id,
            channel=channel,
            actor="agent",
        )

    async def test_llm_hint_resolves_one_current_thread_target(self):
        approval = await self.create_pending_approval(
            "父子 Chunk 研究", "artifact-one"
        )

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(action="approve", target_hint="父子 Chunk"),
            thread_id=self.THREAD_ID,
        )

        self.assertEqual(resolution.status, PublishingIntentResolutionStatus.RESOLVED)
        self.assertEqual(resolution.target.approval_id, approval.approval_id)

    async def test_same_hint_for_two_articles_is_ambiguous(self):
        await self.create_pending_approval("研究报告 A", "artifact-a")
        await self.create_pending_approval("研究报告 B", "artifact-b")

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(action="approve", target_hint="研究报告"),
            thread_id=self.THREAD_ID,
        )

        self.assertEqual(resolution.status, PublishingIntentResolutionStatus.AMBIGUOUS)
        self.assertEqual(len(resolution.candidates), 2)

    async def test_channel_hint_selects_one_platform_target(self):
        await self.create_pending_approval(
            "多平台文章",
            "artifact-local",
            channel=PublicationChannel.LOCAL_STATIC_SITE,
        )
        wechat_approval = await self.create_pending_approval(
            "多平台文章",
            "artifact-wechat",
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(
                action="approve",
                target_hint="多平台文章",
                channel="wechat_official_account",
            ),
            thread_id=self.THREAD_ID,
        )

        self.assertEqual(resolution.status, PublishingIntentResolutionStatus.RESOLVED)
        self.assertEqual(resolution.target.approval_id, wechat_approval.approval_id)
        self.assertEqual(
            resolution.target.channel,
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )

    async def test_request_publication_resolves_approved_article_without_approval(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-request",
            title="待请求发布的文章",
            slug="article-request",
        )
        self.service.approve_article(article.article_id)

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(
                action="request_publication",
                target_hint="待请求发布",
                channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            ),
            thread_id=self.THREAD_ID,
        )

        self.assertEqual(resolution.status, PublishingIntentResolutionStatus.RESOLVED)
        self.assertEqual(resolution.target.article_id, article.article_id)
        self.assertIsNone(resolution.target.approval_id)

    async def test_request_publication_resolves_draft_for_exact_channel_review(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-draft-request",
            title="待审核的公众号文章",
            slug="draft-wechat-article",
        )

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(
                action="request_publication",
                target_hint="待审核的公众号文章",
                channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            ),
            thread_id=self.THREAD_ID,
        )

        self.assertEqual(
            resolution.status,
            PublishingIntentResolutionStatus.RESOLVED,
        )
        self.assertEqual(resolution.target.article_id, article.article_id)

    async def test_request_publication_tool_creates_approval_and_hitl_card(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-tool-request",
            title="工具请求发布的文章",
            slug="article-tool-request",
        )
        tool = build_request_publication_approval_tool(
            self.service,
            self.hitl_service,
        )
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}},
            tool_call_id="request-publication-call-1",
        )

        result = await tool.coroutine(
            runtime=runtime,
            target_hint="工具请求发布",
            channel="wechat_official_account",
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["approval_required"])
        self.assertEqual(result["interaction_status"], "pending")
        approval_id = result["target"]["approval_id"]
        approval = self.repository.get_approval_request(approval_id)
        self.assertEqual(
            approval.channel,
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )
        interaction = self.hitl_repository.get(result["interaction_id"])
        self.assertEqual(interaction.target_id, approval_id)
        self.assertEqual(
            self.service.get_article(article.article_id).status.value,
            "draft",
        )

    async def test_explicit_generated_attachment_ids_skip_image_selection(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-generated-image",
            title="生成图片直接发布",
            slug="generated-image-publication",
        )
        attachment_service = ImageAttachmentService(
            self.repository,
            Path(self.temp_dir.name) / "attachments",
        )
        attachment = attachment_service.save(
            thread_id=self.THREAD_ID,
            filename="generated.png",
            content_type="image/png",
            content=b"generated-image",
        )
        tool = build_request_publication_approval_tool(
            self.service,
            image_attachment_service=attachment_service,
        )

        result = await tool.coroutine(
            runtime=SimpleNamespace(
                config={"configurable": {"thread_id": self.THREAD_ID}},
                tool_call_id="generated-publication-call",
            ),
            target_hint=article.title,
            channel="xiaohongshu",
            attachment_ids=[attachment.attachment_id],
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["approval_required"])
        self.assertNotIn("attachment_selection_required", result)
        self.assertEqual(
            result["target"]["attachment_ids"],
            [attachment.attachment_id],
        )

    async def test_generated_attachment_id_typo_is_repaired_from_server_result(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-generated-image-typo",
            title="生成图片 ID 容错",
            slug="generated-image-id-typo",
        )
        attachment_service = ImageAttachmentService(
            self.repository,
            Path(self.temp_dir.name) / "attachments-typo",
        )
        attachment = attachment_service.save(
            thread_id=self.THREAD_ID,
            filename="generated.png",
            content_type="image/png",
            content=b"generated-image-typo",
        )
        actual = attachment.attachment_id
        typo = actual[:17] + actual[18] + actual[17] + actual[19:]
        tool = build_request_publication_approval_tool(
            self.service,
            image_attachment_service=attachment_service,
        )

        result = await tool.coroutine(
            runtime=SimpleNamespace(
                config={"configurable": {"thread_id": self.THREAD_ID}},
                context=SimpleNamespace(generated_attachment_ids=(actual,)),
                state={},
                tool_call_id="generated-publication-typo-call",
            ),
            target_hint=article.title,
            channel="xiaohongshu",
            attachment_ids=[typo],
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["attachment_ids"], [actual])
        self.assertEqual(result["target"]["attachment_ids"], [actual])

    async def test_request_publication_rejects_invalid_wechat_title_before_hitl(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-long-wechat-title",
            title="A" * 33,
            slug="long-wechat-title",
        )
        tool = build_request_publication_approval_tool(
            self.service,
            self.hitl_service,
        )
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}},
            tool_call_id="invalid-wechat-title-request",
        )

        result = await tool.coroutine(
            runtime=runtime,
            target_hint=article.title,
            channel="wechat_official_account",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "wechat_title_too_long")
        self.assertFalse(result["retryable"])
        self.assertTrue(result["repairable"])
        self.assertEqual(result["repair_action"], "revise_article_for_publication")
        self.assertEqual(result["field"], "title")
        self.assertEqual(result["max_length"], 32)
        self.assertIn("maximum is 32", result["message"])
        self.assertEqual(
            self.repository.list_approval_requests(article_id=article.article_id),
            [],
        )
        self.assertEqual(
            self.hitl_repository.list_for_thread(self.THREAD_ID),
            [],
        )

    async def test_request_publication_marks_xiaohongshu_length_error_repairable(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-long-xiaohongshu-title",
            title="这是一个明确超过二十个字符的小红书标题示例",
            slug="long-xiaohongshu-title",
        )
        tool = build_request_publication_approval_tool(self.service)
        result = await tool.coroutine(
            runtime=SimpleNamespace(
                config={"configurable": {"thread_id": self.THREAD_ID}},
            ),
            target_hint=article.title,
            channel="xiaohongshu",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "xiaohongshu_title_too_long")
        self.assertFalse(result["retryable"])
        self.assertTrue(result["repairable"])
        self.assertEqual(result["field"], "title")
        self.assertEqual(result["max_length"], 20)
        self.assertEqual(
            result["repair_action"],
            "revise_article_for_publication",
        )

    async def test_request_publication_not_found_is_non_retryable(self):
        tool = build_request_publication_approval_tool(self.service)
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}},
        )

        result = await tool.coroutine(
            runtime=runtime,
            target_hint="missing article",
            channel="wechat_official_account",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["resolution_status"], "not_found")
        self.assertEqual(
            result["error_code"],
            "publication_target_not_found",
        )
        self.assertFalse(result["retryable"])

    async def test_resume_resolves_failed_publication_for_explicit_retry(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-failed-resume",
            title="Failed publication article",
            slug="failed-publication-article",
        )
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )
        self.service.approve_publication_request(
            approval.approval_id,
            decision_actor="user",
        )
        failed = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            idempotency_key="failed-resume-1",
            publisher=FailingWeChatPublisher(),
        )
        self.assertEqual(failed.status, PublicationStatus.FAILED)

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(
                action="resume",
                target_hint=article.title,
                channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            ),
            thread_id=self.THREAD_ID,
        )

        self.assertEqual(resolution.status, PublishingIntentResolutionStatus.RESOLVED)
        self.assertEqual(
            resolution.target.publication_status,
            PublicationStatus.FAILED,
        )
        self.assertEqual(
            resolution.target.publication_error_code,
            "wechat_title_too_long",
        )

    async def test_resume_resolves_delivery_unknown_for_explicit_retry(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-unknown-resume",
            title="Unknown publication article",
            slug="delivery-unknown-publication-article",
        )
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )
        self.service.approve_publication_request(
            approval.approval_id,
            decision_actor="user",
        )
        unknown = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            idempotency_key="unknown-resume-1",
            publisher=UnknownWeChatPublisher(),
        )
        self.assertEqual(unknown.status, PublicationStatus.DELIVERY_UNKNOWN)

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(
                action="resume",
                target_hint=article.title,
                channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            ),
            thread_id=self.THREAD_ID,
        )

        self.assertEqual(resolution.status, PublishingIntentResolutionStatus.RESOLVED)
        self.assertEqual(
            resolution.target.publication_status,
            PublicationStatus.DELIVERY_UNKNOWN,
        )
        self.assertEqual(
            resolution.target.publication_id,
            unknown.publication_id,
        )
        self.assertEqual(
            resolution.target.publication_error_code,
            "wechat_publish_request_failed",
        )

    def test_native_interrupt_preview_exposes_initial_wechat_approval_card(self):
        preview = publishing_interrupt_preview(
            {
                "kind": "publication_approval",
                "ok": True,
                "action": "request_publication",
                "resolution_status": "resolved",
                "target": {
                    "approval_id": "approval-wechat-1",
                    "article_id": "article-wechat-1",
                    "article_title": "微信公众号文章",
                    "article_slug": "wechat-article",
                    "article_version": 1,
                    "channel": "wechat_official_account",
                    "approval_status": "pending",
                    "publication_status": None,
                },
                "requires_user_confirmation": True,
                "interaction_id": "interaction-wechat-1",
                "interaction_status": "pending",
            }
        )

        self.assertTrue(preview["native_interrupt"])
        self.assertEqual(preview["action"], "approve")
        self.assertEqual(
            preview["target"]["channel"],
            "wechat_official_account",
        )

    async def test_request_publication_requires_channel(self):
        tool = build_request_publication_approval_tool(self.service)
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}},
        )

        result = await tool.coroutine(
            runtime=runtime,
            target_hint="文章",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error_code"],
            "publication_channel_required",
        )

    async def test_request_publication_stops_repeating_after_native_resume(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-native-resume",
            title="原生审批恢复文章",
            slug="native-resume-article",
        )
        self.service.approve_article(article.article_id)
        tool = build_request_publication_approval_tool(
            self.service,
            self.hitl_service,
            enable_native_interrupt=True,
        )
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}},
            tool_call_id="request-publication-native-resume",
        )

        with patch(
            "deep_research.tools.publishing.interrupt",
            return_value={"decision": "approved"},
        ):
            result = await tool.coroutine(
                runtime=runtime,
                target_hint="原生审批恢复文章",
                channel="wechat_official_account",
            )

        self.assertTrue(result["resumed"])
        self.assertEqual(result["resume_decision"], "approved")
        self.assertFalse(result["approval_required"])
        self.assertFalse(result["requires_user_confirmation"])
        self.assertIn("Do not request approval again", result["message"])

    async def test_other_thread_target_is_not_visible(self):
        await self.create_pending_approval("本线程文章", "artifact-local")
        other = await self.service.create_article_from_artifact(
            thread_id="other-thread",
            artifact_id="artifact-other",
            title="其他线程文章",
            slug="other-thread-article",
        )
        self.service.approve_article(other.article_id)
        self.service.request_publication_approval(
            other.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            actor="agent",
        )

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(action="approve", target_hint="其他线程"),
            thread_id=self.THREAD_ID,
        )

        self.assertEqual(resolution.status, PublishingIntentResolutionStatus.NOT_FOUND)

    async def test_tool_resolves_without_approving(self):
        approval = await self.create_pending_approval(
            "需要确认的文章", "artifact-confirm"
        )
        tool = build_resolve_publication_intent_tool(self.service)
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}}
        )

        result = await tool.coroutine(
            action="approve",
            runtime=runtime,
            target_hint="需要确认",
            channel="local_static_site",
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["requires_user_confirmation"])
        self.assertEqual(result["target"]["approval_id"], approval.approval_id)
        self.assertEqual(result["channel"], "local_static_site")
        self.assertEqual(
            self.repository.get_approval_request(approval.approval_id).status.value,
            "pending",
        )

    async def test_status_tool_surfaces_recoverable_hitl_card(self):
        approval = await self.create_pending_approval(
            "可恢复审批的文章", "artifact-recovery"
        )
        interaction = self.hitl_service.create_or_get_interaction(
            thread_id=self.THREAD_ID,
            run_id="recovery-run-1",
            action=HITLAction.APPROVE,
            target_type="publication_approval",
            target_id=approval.approval_id,
            target_version=approval.article_version,
        )
        tool = build_publication_approval_status_tool(
            self.service,
            self.hitl_service,
        )
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}},
        )

        result = await tool.coroutine(runtime=runtime)

        self.assertTrue(result["ok"])
        self.assertEqual(result["approvals"][0]["interaction_id"], interaction.interaction_id)
        self.assertEqual(result["approvals"][0]["interaction_status"], "pending")
        self.assertEqual(result["recovery_cards"][0]["interaction_id"], interaction.interaction_id)
        self.assertTrue(result["recovery_cards"][0]["requires_user_confirmation"])

    async def test_request_approval_persists_explicit_attachment_ids(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-explicit-image",
            title="明确配图文章",
            slug="explicit-image-article",
        )
        self.service.approve_article(article.article_id)
        tool = build_request_publication_approval_tool(self.service)
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}},
        )

        result = await tool.coroutine(
            runtime=runtime,
            target_hint="明确配图文章",
            channel="xiaohongshu",
            attachment_ids=["image-a", "image-b", "image-a"],
        )

        self.assertTrue(result["ok"])
        approval = self.repository.get_approval_request(
            result["target"]["approval_id"]
        )
        self.assertEqual(approval.attachment_ids, ("image-a", "image-b"))
        self.assertEqual(result["attachment_ids"], ["image-a", "image-b"])

    async def test_status_tool_surfaces_publish_button_after_approval(self):
        approval = await self.create_pending_approval(
            "已批准等待发布的文章", "artifact-approved-recovery"
        )
        interaction = self.hitl_service.create_or_get_interaction(
            thread_id=self.THREAD_ID,
            run_id="approved-recovery-run",
            action=HITLAction.APPROVE,
            target_type="publication_approval",
            target_id=approval.approval_id,
            target_version=approval.article_version,
        )
        self.service.approve_publication_request(
            approval.approval_id,
            decision_actor="user",
        )
        self.hitl_service.approve(
            interaction.interaction_id,
            decision_actor="user",
        )

        tool = build_publication_approval_status_tool(
            self.service,
            self.hitl_service,
        )
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}},
        )

        result = await tool.coroutine(runtime=runtime)

        card = result["recovery_cards"][0]
        self.assertEqual(card["action"], "resume")
        self.assertTrue(card["requires_user_confirmation"])
        self.assertEqual(card["target"]["publication_status"], None)

    def test_status_preview_keeps_only_safe_recovery_fields(self):
        preview = publishing_approval_status_preview(
            {
                "output": {
                    "ok": True,
                    "recovery_cards": [
                        {
                            "action": "approve",
                            "resolution_status": "resolved",
                            "target": {
                                "approval_id": "approval-1",
                                "article_id": "article-1",
                                "article_title": "文章 A",
                                "article_slug": "article-a",
                                "article_version": 1,
                                "channel": "local_static_site",
                                "approval_status": "pending",
                                "publication_status": None,
                            },
                            "candidates": [],
                            "requires_user_confirmation": True,
                            "interaction_id": "interaction-1",
                            "interaction_status": "pending",
                            "workspace_path": "C:/private/secret.md",
                        },
                    ],
                },
            }
        )

        self.assertEqual(len(preview["cards"]), 1)
        self.assertEqual(
            preview["cards"][0]["interaction_id"],
            "interaction-1",
        )
        self.assertNotIn("workspace_path", preview["cards"][0])

    async def test_tool_creates_idempotent_hitl_interaction_for_unique_target(self):
        approval = await self.create_pending_approval(
            "需要持久化审批的文章", "artifact-hitl"
        )
        tool = build_resolve_publication_intent_tool(
            self.service,
            self.hitl_service,
        )
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": self.THREAD_ID}},
            tool_call_id="resolve-call-1",
        )

        first = await tool.coroutine(
            action="approve",
            runtime=runtime,
            target_hint="需要持久化审批",
        )
        replay = await tool.coroutine(
            action="approve",
            runtime=runtime,
            target_hint="需要持久化审批",
        )

        self.assertTrue(first["ok"])
        self.assertEqual(first["interaction_status"], "pending")
        self.assertEqual(
            first["interaction_id"],
            replay["interaction_id"],
        )
        interaction = self.hitl_repository.get(first["interaction_id"])
        self.assertEqual(interaction.thread_id, self.THREAD_ID)
        self.assertEqual(interaction.target_id, approval.approval_id)
        self.assertEqual(interaction.target_version, approval.article_version)


if __name__ == "__main__":
    unittest.main()
