import subprocess
import unittest
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


class FrontendPublishingIntentContractTests(unittest.TestCase):
    def test_publishing_intent_frontend_scripts_have_valid_syntax(self):
        scripts = [
            STATIC_DIR / "js" / "research-stream.js",
            STATIC_DIR / "js" / "publishing.js",
            STATIC_DIR / "js" / "attachments.js",
            STATIC_DIR / "js" / "knowledge.js",
        ]
        for script in scripts:
            result = subprocess.run(
                ["node", "--check", str(script)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_stream_and_card_contract_is_present(self):
        stream = (STATIC_DIR / "js" / "research-stream.js").read_text(
            encoding="utf-8",
        )
        publishing = (STATIC_DIR / "js" / "publishing.js").read_text(
            encoding="utf-8",
        )
        styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

        self.assertIn('event.type === "publishing_intent"', stream)
        self.assertIn('event.type === "publication_attachment_selection"', stream)
        self.assertIn("renderPublicationAttachmentSelectionCard(event)", stream)
        self.assertIn("renderPublishingIntentCard(event)", stream)
        self.assertIn("function renderPublishingIntentCard(event)", publishing)
        self.assertIn(
            "function renderPublicationAttachmentSelectionCard(event)",
            publishing,
        )
        self.assertIn("submitResearchWithAttachmentSelection", publishing)
        self.assertIn("attachment_selection_confirmed", stream)
        self.assertIn("globalThis.renderPublishingIntentCard", publishing)
        self.assertIn("function executePublishingIntent", publishing)
        self.assertIn("/hitl/interactions/", publishing)
        self.assertIn("publishingIntentInteractionId", publishing)
        self.assertIn("intentSignature", publishing)
        self.assertIn("existingCard.remove()", publishing)
        self.assertIn("/publishing/approval-requests/", publishing)
        self.assertIn("/publishing/articles/", publishing)
        self.assertIn('"request_changes"', publishing)
        self.assertIn("修改文章", publishing)
        self.assertIn("/hitl/interactions/", publishing)
        self.assertIn("/resume?thread_id=", publishing)
        self.assertIn("wechat_official_account", publishing)
        self.assertIn("xiaohongshu", publishing)
        self.assertIn("douyin", publishing)
        self.assertIn("channel: target.channel", publishing)
        self.assertIn('result?.status === "failed"', publishing)
        self.assertIn("wechat_title_too_long", publishing)
        self.assertIn("publishingPublicationSucceeded", publishing)
        self.assertIn('target.publication_status === "delivery_unknown"', publishing)
        self.assertIn("target.publication_id", publishing)
        self.assertIn("confirm_delivery_unknown: isDeliveryUnknown", publishing)
        self.assertIn(
            "/publishing/publications/${encodeURIComponent(target.publication_id)}/retry",
            publishing,
        )
        self.assertIn("publishingIntentIdempotencyKeys.delete", publishing)
        self.assertIn("publishing-intent-card", styles)
        self.assertIn('pending: { action: "approve", requires: true }', publishing)
        self.assertIn('rejected: { action: "resume", requires: true }', publishing)
        self.assertIn('approved: { action: "resume", requires: true }', publishing)
        self.assertIn('resuming: { action: "resume", requires: false }', publishing)
        self.assertIn('failed: { action: "resume", requires: true }', publishing)
        self.assertIn("approval?.workflow_status || approval?.status", publishing)
        self.assertIn("const isResuming = status === \"resuming\"", publishing)
        self.assertIn("const isFailed = status === \"failed\"", publishing)
        self.assertIn("function renderPublishingHITLRecoveryError", publishing)
        self.assertIn("重试恢复审批", publishing)
        self.assertIn("hitl_recovery_invalid_response", publishing)
        self.assertIn("data-publishing-hitl-recovery-error", publishing)

    def test_publication_approval_pins_attachment_snapshot(self):
        prompt = (STATIC_DIR.parent / "prompts" / "supervisor.py").read_text(
            encoding="utf-8",
        )
        tools = (STATIC_DIR.parent / "tools" / "publishing.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("attachment_ids 原样传入", prompt)
        self.assertIn("固化到发布审批快照", prompt)
        self.assertIn('attachment_ids: list[str] | None = None', tools)
        self.assertIn("publication_attachment_mismatch", (STATIC_DIR.parent / "publishing" / "service_publications.py").read_text(encoding="utf-8"))

    def test_publishing_preview_accepts_all_supported_channels(self):
        tool_events = (STATIC_DIR.parent / "agent" / "tool_events.py").read_text(
            encoding="utf-8",
        )
        self.assertIn('"xiaohongshu"', tool_events)
        self.assertIn('"douyin"', tool_events)

    def test_stream_does_not_crash_when_publishing_card_script_is_unavailable(self):
        stream = (STATIC_DIR / "js" / "research-stream.js").read_text(
            encoding="utf-8",
        )
        self.assertIn(
            'typeof renderPublishingIntentCard === "function"',
            stream,
        )
        self.assertIn("发布确认卡片组件尚未加载，请刷新页面后重试", stream)

    def test_intent_card_uses_safe_text_and_explicit_action(self):
        publishing = (STATIC_DIR / "js" / "publishing.js").read_text(
            encoding="utf-8",
        )

        self.assertIn("body.textContent", publishing)
        self.assertIn("title.textContent", publishing)
        self.assertIn('confirmButton.type = "button"', publishing)
        self.assertIn("requires_user_confirmation", publishing)
        self.assertIn("decision_reason: reason || null", publishing)
        self.assertIn("不会自动选择文章", publishing)

    def test_wechat_attachment_and_preview_contract_is_present(self):
        index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        attachments = (STATIC_DIR / "js" / "attachments.js").read_text(
            encoding="utf-8",
        )
        styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
        publishing = (STATIC_DIR / "js" / "publishing.js").read_text(
            encoding="utf-8",
        )

        self.assertIn('id="image-upload-button"', index)
        self.assertIn('id="image-file-input"', index)
        self.assertIn('accept="image/jpeg,image/png,image/gif,.jpg,.jpeg,.png,.gif"', index)
        self.assertIn("new FormData()", attachments)
        self.assertIn('formData.append("thread_id", currentThreadId)', attachments)
        self.assertIn('formData.append("file", file', attachments)
        self.assertIn('"/publishing/attachments/images"', attachments)
        self.assertIn(
            "/publishing/attachments/${encodeURIComponent(attachmentId)}/wechat-cover",
            attachments,
        )
        self.assertIn("attachment_id", attachments)
        self.assertIn("当前微信公众号封面", attachments)
        self.assertIn('method: "DELETE"', attachments)
        self.assertIn("共享图片库", attachments)
        self.assertIn("选择为本次发布配图", attachments)
        self.assertIn("setPublicationAttachmentIds", attachments)
        self.assertIn('id="message-attachments"', index)
        self.assertIn('id="image-library-open"', index)
        self.assertIn('id="image-library-dialog"', index)
        self.assertIn('id="image-attachment-detail-dialog"', index)
        self.assertIn('id="image-attachment-detail"', index)
        self.assertIn('id="publishing-insert-image-button"', index)
        self.assertIn("pendingMessageAttachments", attachments)
        self.assertIn("uniqueImageAttachments", attachments)
        self.assertIn("content_sha256", attachments)
        self.assertIn("getPendingMessageAttachmentIds", attachments)
        self.assertIn("clearPendingMessageAttachmentIds", attachments)
        self.assertIn("附加到下一条消息", attachments)
        self.assertIn("automaticallyAttachedIds", (STATIC_DIR / "js" / "research-stream.js").read_text(encoding="utf-8"))
        self.assertIn("analysis_status", attachments)
        self.assertIn("analyze_uploaded_image", (STATIC_DIR.parent / "tools" / "publishing.py").read_text(encoding="utf-8"))
        self.assertIn("renderImageAnalysisCard", publishing)
        self.assertIn("帮我解读", publishing)
        self.assertIn("提取文字", publishing)
        self.assertIn('class="image-library-heading"', index)
        self.assertIn("chat-attachment-preview", (STATIC_DIR / "js" / "research-ui.js").read_text(encoding="utf-8"))
        self.assertIn("hasPendingImageUploads", attachments)
        self.assertIn("restorePendingMessageAttachmentIds", attachments)
        self.assertIn("openInlineImagePicker", attachments)
        self.assertIn("插入正文", attachments)
        self.assertIn("contextmenu", attachments)
        self.assertIn("openImageAttachmentDetail", attachments)
        self.assertIn("image-attachment-context-menu", attachments)
        self.assertIn("image-attachment-tile", attachments)
        self.assertIn("分析结果", attachments)
        self.assertIn("appendChatAttachmentPreviews", (STATIC_DIR / "js" / "thread-view.js").read_text(encoding="utf-8"))
        self.assertIn("appendConversationTurn(question, requestAttachmentIds)", (STATIC_DIR / "js" / "research-stream.js").read_text(encoding="utf-8"))
        self.assertIn("请先上传或选择微信公众号封面", publishing)
        self.assertIn("发布模式：草稿", publishing)
        self.assertIn("发布模式：正式发布", publishing)
        self.assertIn("平台草稿已创建，尚未公开", publishing)
        self.assertIn("delivery_unknown", publishing)
        self.assertIn("refreshPublishingPublication", publishing)
        self.assertIn("retryPublishingPublication", publishing)
        self.assertIn("publishingStatusPollIntervalMs", publishing)
        self.assertIn("insertPublishingImage", publishing)
        self.assertIn("attachment://${attachmentId}", publishing)
        self.assertIn("restartPublishingStatusPolling", publishing)
        self.assertIn("自动同步中", publishing)
        self.assertIn("stopPublishingStatusPolling", publishing)
        self.assertIn("/publishing/publications/${encodeURIComponent(current.publication_id)}/refresh", publishing)
        self.assertIn("/publishing/publications/${encodeURIComponent(existing.publication_id)}/retry", publishing)
        self.assertIn("确认无误后才会重新提交", publishing)
        self.assertIn("publishing-history-actions", styles)
        self.assertIn('approval.status !== "approved"', publishing)

    def test_knowledge_library_has_left_click_file_actions(self):
        knowledge = (STATIC_DIR / "js" / "knowledge.js").read_text(
            encoding="utf-8",
        )
        styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
        self.assertIn("knowledge-file-row", knowledge)
        self.assertIn("knowledge-row-action", knowledge)
        self.assertIn("openKnowledgeDocumentDetail", knowledge)
        self.assertIn("showKnowledgeContextMenu", knowledge)
        self.assertIn("loadKnowledgeDocumentPreview", knowledge)
        self.assertIn("knowledge-document-content-preview", knowledge)
        self.assertIn("knowledge-file-row", styles)
        self.assertIn("knowledge-document-content", styles)


if __name__ == "__main__":
    unittest.main()
