        const publishingOpenButton = document.querySelector(
            "#publishing-open",
        );
        const publishingDialog = document.querySelector(
            "#publishing-dialog",
        );
        const publishingCloseButton = document.querySelector(
            "#publishing-close",
        );
        const publishingRefreshButton = document.querySelector(
            "#publishing-refresh",
        );
        const publishingStatus = document.querySelector(
            "#publishing-status",
        );
        const publishingArtifactSelect = document.querySelector(
            "#publishing-artifact-select",
        );
        const publishingImportButton = document.querySelector(
            "#publishing-import-button",
        );
        const publishingArticleList = document.querySelector(
            "#publishing-article-list",
        );
        const publishingArticleCount = document.querySelector(
            "#publishing-article-count",
        );
        const publishingForm = document.querySelector(
            "#publishing-form",
        );
        const publishingEditorState = document.querySelector(
            "#publishing-editor-state",
        );
        const publishingEmptyEditor = document.querySelector(
            "#publishing-empty-editor",
        );
        const publishingTitle = document.querySelector(
            "#publishing-title",
        );
        const publishingSlug = document.querySelector(
            "#publishing-slug",
        );
        const publishingExcerpt = document.querySelector(
            "#publishing-excerpt",
        );
        const publishingTags = document.querySelector(
            "#publishing-tags",
        );
        const publishingMarkdown = document.querySelector(
            "#publishing-markdown",
        );
        const publishingInsertImageButton = document.querySelector(
            "#publishing-insert-image-button",
        );
        const publishingPreviewButton = document.querySelector(
            "#publishing-preview-button",
        );
        const publishingPreview = document.querySelector(
            "#publishing-preview",
        );
        const publishingPreviewClose = document.querySelector(
            "#publishing-preview-close",
        );
        const publishingPreviewContent = document.querySelector(
            "#publishing-preview-content",
        );
        const publishingSaveButton = document.querySelector(
            "#publishing-save-button",
        );
        const publishingApproveButton = document.querySelector(
            "#publishing-approve-button",
        );
        const publishingPublishButton = document.querySelector(
            "#publishing-publish-button",
        );
        const publishingApprovalPanel = document.querySelector(
            "#publishing-approval-panel",
        );
        const publishingApprovalState = document.querySelector(
            "#publishing-approval-state",
        );
        const publishingApprovalReasonField = document.querySelector(
            "#publishing-approval-reason-field",
        );
        const publishingApprovalReason = document.querySelector(
            "#publishing-approval-reason",
        );
        const publishingRequestApprovalButton = document.querySelector(
            "#publishing-request-approval-button",
        );
        const publishingApprovalApproveButton = document.querySelector(
            "#publishing-approval-approve-button",
        );
        const publishingApprovalRejectButton = document.querySelector(
            "#publishing-approval-reject-button",
        );
        const publishingChannel = document.querySelector(
            "#publishing-channel",
        );
        const publishingPreviewTitle = document.querySelector(
            "#publishing-preview-title",
        );
        const publishingPreviewExcerpt = document.querySelector(
            "#publishing-preview-excerpt",
        );
        const publishingPreviewMode = document.querySelector(
            "#publishing-preview-mode",
        );
        const publishingPreviewCoverName = document.querySelector(
            "#publishing-preview-cover-name",
        );
        const publishingPreviewCoverEmpty = document.querySelector(
            "#publishing-preview-cover-empty",
        );
        const publishingPreviewCover = document.querySelector(
            "#publishing-preview-cover",
        );
        const publishingHistoryState = document.querySelector(
            "#publishing-history-state",
        );
        const publishingHistoryList = document.querySelector(
            "#publishing-history-list",
        );

        const publishingArticles = new Map();
        const publishingApprovals = new Map();
        const publishingIdempotencyKeys = new Map();
        const publishingIntentIdempotencyKeys = new Map();
        const hitlDecisionIdempotencyKeys = new Map();
        const publishingIntentCards = new Set();
        const publicationAttachmentSelectionCards = new Set();
        let publishingSelectedArticleId = null;
        let publishingSelectedChannel = publishingChannel?.value || "local_static_site";
        let publishingWechatPreview = {
            configured: false,
            publish_mode: null,
        };
        let publishingBusy = false;
        let publishingIntentBusy = false;
        const publishingStatusPollIntervalMs = 10000;
        let publishingStatusPollTimer = null;
        let publishingStatusPollInFlight = false;
        const publishingDecisionReasonRequired = false;

        const publishingStatusLabels = {
            draft: "草稿",
            approved: "已批准",
            publishing: "已提交正式发布，等待平台状态确认",
            drafted: "平台草稿已创建，尚未公开",
            published: "平台已确认发布成功",
            failed: "发布失败",
            delivery_unknown: "请求结果不确定，请人工到对应平台核对后决定是否重试",
        };

        const publishingErrorLabels = {
            wechat_cover_not_configured: "请先上传或选择微信公众号封面",
            wechat_cover_attachment_failed: "设置微信公众号封面失败",
            wechat_image_not_configured: "微信公众号配图服务尚未配置",
            wechat_image_not_found: "微信公众号配图不存在",
            wechat_image_invalid: "微信公众号配图文件无效或格式不支持",
            publication_failed: "\u53d1\u5e03\u5931\u8d25",
            wechat_title_too_long: "\u5fae\u4fe1\u6587\u7ae0\u6807\u9898\u8d85\u51fa 32 \u4e2a\u5b57\u7b26\u7684\u9650\u5236",
            wechat_author_too_long: "\u5fae\u4fe1\u6587\u7ae0\u4f5c\u8005\u540d\u8d85\u51fa\u9650\u5236",
            wechat_digest_too_long: "\u5fae\u4fe1\u6587\u7ae0\u6458\u8981\u8d85\u51fa\u9650\u5236",
            wechat_token_rejected: "\u5fae\u4fe1 access token \u88ab\u62d2\u7edd\uff0c\u8bf7\u68c0\u67e5 AppID/Secret \u548c IP \u767d\u540d\u5355",
            wechat_token_request_failed: "\u5fae\u4fe1 access token \u8bf7\u6c42\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5\u7f51\u7edc",
            wechat_token_response_invalid: "\u5fae\u4fe1 access token \u54cd\u5e94\u683c\u5f0f\u65e0\u6548",
            wechat_draft_rejected: "\u5fae\u4fe1\u62d2\u7edd\u8349\u7a3f\u8bf7\u6c42",
            wechat_draft_request_failed: "\u5fae\u4fe1\u8349\u7a3f\u8bf7\u6c42\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5\u7f51\u7edc",
            xiaohongshu_not_configured: "小红书发布服务尚未配置",
            xiaohongshu_title_too_long: "小红书标题超过 20 个字符",
            xiaohongshu_content_empty: "小红书正文不能为空",
            xiaohongshu_content_too_long: "小红书正文超过 1000 个字符，请精简后重试",
            xiaohongshu_image_count_invalid: "小红书图片数量不符合限制",
            xiaohongshu_image_required: "小红书至少需要选择一张配图",
            xiaohongshu_image_not_found: "小红书图片文件不存在",
            xiaohongshu_service_unreachable: "小红书发布服务无法连接",
            xiaohongshu_publish_receipt_missing: "小红书未返回可追踪的发布任务",
            douyin_not_configured: "抖音发布服务尚未配置",
            douyin_title_too_long: "抖音标题超过配置长度限制",
            douyin_content_empty: "抖音正文不能为空",
            douyin_image_count_invalid: "抖音图片数量不符合限制",
            douyin_image_required: "抖音图文至少需要选择一张配图",
            douyin_image_not_found: "抖音图片文件不存在",
            douyin_service_unreachable: "抖音发布服务无法连接",
            douyin_publish_receipt_missing: "抖音未返回可追踪的发布任务",
            article_not_found: "文章不存在",
            artifact_not_found: "研究报告不存在",
            artifact_integrity_failed: "研究报告完整性校验失败",
            article_not_editable: "当前文章不可编辑",
            article_not_approval_ready: "文章当前状态不能提交发布审批",
            article_not_publishable: "文章尚未批准或不可发布",
            approval_not_pending: "审批请求已经处理",
            approval_request_conflict: "发布审批请求创建冲突",
            approval_request_not_found: "发布审批请求不存在",
            approval_target_stale: "审批请求对应的文章版本已变化",
            invalid_state_transition: "文章状态不允许此操作",
            invalid_publishing_request: "发布请求无效",
            idempotency_conflict: "重复发布请求不一致",
            publication_approval_required: "请先完成发布审批",
            static_write_failed: "本地站点写入失败",
            unsupported_publication_channel: "当前发布渠道不可用",
            publication_not_found: "发布记录不存在",
            delivery_unknown_requires_manual_confirmation: "结果不确定的记录需要先人工核对对应平台",
            delivery_unknown_confirmation_required: "请先确认对应平台未发布，再执行重试",
            hitl_graph_not_interrupted: "审批已记录，但没有找到可恢复的 Agent 运行，请重新发起发布审批",
            hitl_graph_unavailable: "审批已记录，但 Agent 当前不可恢复，请重试本次会话",
            hitl_resume_failed: "审批已记录，但发布流程恢复失败，请重试",
            hitl_pending: "当前会话正在等待审批，请先处理会话中的审批卡片",
            hitl_interaction_stale: "这张审批卡片已经失效，请重新发起发布审批",
            hitl_interaction_expired: "这张审批卡片已经过期，请重新发起发布审批",
            hitl_idempotency_conflict: "这次审批请求的幂等键与原请求不一致",
            hitl_decision_in_progress: "这次审批请求正在处理中，请稍候刷新",
        };

        function setPublishingStatus(message, state = "") {
            if (!publishingStatus) {
                return;
            }

            publishingStatus.textContent = message || "";
            publishingStatus.dataset.state = state;
        }

        function publishingThreadId() {
            return typeof threadId === "string" ? threadId : "";
        }

        function publishingDate(value) {
            if (typeof value !== "string" || !value) {
                return "时间未知";
            }

            const parsed = new Date(value);
            if (Number.isNaN(parsed.getTime())) {
                return "时间未知";
            }

            return new Intl.DateTimeFormat("zh-CN", {
                dateStyle: "medium",
                timeStyle: "short",
            }).format(parsed);
        }

        function publishingSafeError(error) {
            const candidate = (
                error && typeof error.code === "string"
                    ? error.code
                    : "publishing_operation_failed"
            );
            const code = /^[a-z0-9_]{1,80}$/i.test(candidate)
                ? candidate
                : "publishing_operation_failed";
            return `${publishingErrorLabels[code] || "操作失败"}（错误代码：${code}）`;
        }

        function publishingPublicationFailure(publication) {
            const code = (
                publication && typeof publication.error_code === "string" &&
                publication.error_code.trim()
                    ? publication.error_code.trim()
                    : "publication_failed"
            );
            const error = new Error(code);
            error.code = code;
            return error;
        }

        function publishingPublicationSucceeded(publication) {
            return Boolean(
                publication &&
                ["drafted", "publishing", "published", "delivery_unknown"].includes(
                    publication.status,
                ),
            );
        }

        function publishingPublicationMessage(publication) {
            const channel = publishingChannelLabel(publication?.channel);
            if (publication?.status === "drafted") {
                return `${channel}草稿已创建，尚未公开`;
            }
            if (publication?.status === "publishing") {
                return `已提交${channel}发布，等待状态确认`;
            }
            if (publication?.status === "published") {
                return publication.public_url
                    ? `${channel}已确认发布成功：${publication.public_url}`
                    : `${channel}已确认发布成功`;
            }
            if (publication?.status === "delivery_unknown") {
                return `请求结果不确定，请人工核对${channel}后决定是否重试`;
            }
            return "发布失败";
        }

        async function publishingRequest(url, options = {}) {
            const response = await fetch(url, {
                ...options,
                headers: {
                    Accept: "application/json",
                    ...(options.body
                        ? { "Content-Type": "application/json" }
                        : {}),
                    ...(options.headers || {}),
                },
            });

            let payload = null;
            try {
                payload = await response.json();
            } catch {
                payload = null;
            }

            if (!response.ok) {
                const detail = payload && payload.detail;
                const code = (
                    detail && typeof detail.error_code === "string"
                        ? detail.error_code
                        : "publishing_operation_failed"
                );
                const error = new Error(code);
                error.code = code;
                throw error;
            }

            return payload;
        }

        function renderPublishingHITLRecoveryError(error) {
            const message = error?.code || error?.message || "unknown_error";
            const existing = conversation?.querySelector(
                "[data-publishing-hitl-recovery-error]",
            );
            existing?.remove();

            const assistant = publishingIntentCardAssistant();
            if (!assistant) {
                const topStatus = document.querySelector("#status");
                if (topStatus) {
                    topStatus.textContent = `审批卡片恢复失败（${message}），请重试`;
                }
                return;
            }

            const card = document.createElement("section");
            card.className = "publishing-intent-card publishing-hitl-recovery-error";
            card.dataset.publishingHitlRecoveryError = "true";
            card.setAttribute("role", "alert");
            const title = document.createElement("h3");
            title.textContent = "审批卡片恢复失败";
            const body = document.createElement("p");
            body.textContent = `当前待处理的发布审批未能恢复（${message}）。请检查服务后重试。`;
            const retry = document.createElement("button");
            retry.type = "button";
            retry.className = "publishing-action publishing-action-primary";
            retry.textContent = "重试恢复审批";
            retry.addEventListener("click", () => {
                retry.disabled = true;
                body.textContent = "正在重新恢复审批卡片...";
                void restorePendingPublishingHITL();
            });
            card.append(title, body, retry);
            assistant.append(card);
            conversation.scrollTop = conversation.scrollHeight;
        }

        async function restorePendingPublishingHITL() {
            const currentThreadId = publishingThreadId();
            if (!currentThreadId || !conversation) {
                return;
            }
            conversation.querySelector(
                "[data-publishing-hitl-recovery-error]",
            )?.remove();
            try {
                const interactions = await publishingRequest(
                    `/hitl/interactions?thread_id=${encodeURIComponent(currentThreadId)}`,
                );
                if (!Array.isArray(interactions)) {
                    const error = new Error("hitl_recovery_invalid_response");
                    error.code = "hitl_recovery_invalid_response";
                    throw error;
                }
                for (const interaction of interactions) {
                    if (
                        !interaction ||
                        interaction.thread_id !== currentThreadId ||
                        !interaction.target
                    ) {
                        continue;
                    }
                    // Rebuild every recoverable lifecycle state.  An
                    // approved interaction still has an interrupted graph;
                    // resuming means the browser must offer the continue
                    // action. RESUMING is shown without a button so a refresh
                    // cannot issue a second resume.
                    const recoverable = {
                        pending: { action: "approve", requires: true },
                        rejected: { action: "resume", requires: true },
                        approved: { action: "resume", requires: true },
                        resuming: { action: "resume", requires: false },
                        failed: { action: "resume", requires: true },
                    }[interaction.status];
                    if (!recoverable) {
                        continue;
                    }
                    renderPublishingIntentCard({
                        type: "publishing_intent",
                        action: recoverable.action,
                        resolution_status: "resolved",
                        requires_user_confirmation: recoverable.requires,
                        interaction_id: interaction.interaction_id,
                        interaction_status: interaction.status,
                        target: interaction.target,
                        candidates: [],
                    });
                }
            } catch (error) {
                renderPublishingHITLRecoveryError(error);
            }
        }

        const publishingChannelLabels = {
            local_static_site: "本地静态站",
            wechat_official_account: "微信公众号",
            xiaohongshu: "小红书",
            douyin: "抖音",
        };

        function publishingChannelLabel(channel) {
            return publishingChannelLabels[channel] || "目标平台";
        }

        function renderImageAnalysisCard(attachment) {
            if (!attachment || typeof attachment.attachment_id !== "string") {
                return;
            }
            let assistant = publishingIntentCardAssistant();
            if (!assistant && conversation) {
                const turn = document.createElement("article");
                turn.className = "conversation-turn image-analysis-turn";
                assistant = document.createElement("div");
                assistant.className = "chat-message assistant";
                const label = document.createElement("div");
                label.className = "chat-label";
                label.textContent = "研究助手";
                assistant.append(label);
                turn.append(assistant);
                conversation.append(turn);
                document.querySelector("#conversation-empty")?.setAttribute("hidden", "hidden");
            }
            if (!assistant) {
                return;
            }
            const existing = assistant.querySelector(
                `[data-image-analysis-id="${CSS.escape(attachment.attachment_id)}"]`,
            );
            if (existing) {
                return;
            }

            const card = document.createElement("section");
            card.className = "image-analysis-card";
            card.dataset.imageAnalysisId = attachment.attachment_id;
            card.setAttribute("aria-label", "图片分析结果");

            const heading = document.createElement("div");
            heading.className = "publishing-intent-card-heading";
            const title = document.createElement("h3");
            title.textContent = "图片分析完成";
            const type = document.createElement("span");
            type.className = "publishing-intent-card-state";
            type.textContent = attachment.analysis_type || "图片";
            heading.append(title, type);

            const summary = document.createElement("p");
            summary.className = "publishing-intent-card-body";
            summary.textContent = attachment.analysis_summary || "已完成图片解析。";
            card.append(heading, summary);

            if (attachment.analysis_ocr_text) {
                const details = document.createElement("details");
                const summaryLabel = document.createElement("summary");
                summaryLabel.textContent = "查看识别文字";
                const text = document.createElement("pre");
                text.className = "image-analysis-ocr";
                text.textContent = attachment.analysis_ocr_text;
                details.append(summaryLabel, text);
                card.append(details);
            }

            const hint = document.createElement("p");
            hint.className = "publishing-intent-card-status";
            hint.textContent = "需要我继续处理这张图片吗？";
            card.append(hint);

            const actions = document.createElement("div");
            actions.className = "publishing-intent-card-actions";
            for (const [label, request] of [
                ["帮我解读", "请详细解读这张图片，并告诉我可以如何处理。"],
                ["提取文字", "请提取这张图片中的完整文字，并保持原有结构。"],
                ["整理成文章", "请根据这张图片的内容整理一篇文章草稿。"],
            ]) {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "publishing-action";
                button.textContent = label;
                button.addEventListener("click", () => {
                    if (typeof globalThis.submitResearchWithAttachmentSelection !== "function") {
                        hint.textContent = "会话提交组件尚未加载，请刷新页面后重试。";
                        return;
                    }
                    for (const actionButton of actions.querySelectorAll("button")) {
                        actionButton.disabled = true;
                    }
                    hint.textContent = "正在把图片交给 LLM 处理...";
                    const submitted = globalThis.submitResearchWithAttachmentSelection({
                        question: request,
                        attachmentIds: [attachment.attachment_id],
                        confirmed: false,
                    });
                    if (!submitted) {
                        for (const actionButton of actions.querySelectorAll("button")) {
                            actionButton.disabled = false;
                        }
                        hint.textContent = "当前还有请求进行中，请稍后再试。";
                    }
                });
                actions.append(button);
            }
            card.append(actions);
            assistant.append(card);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function renderGeneratedImageCard(event) {
            if (!event || !Array.isArray(event.images) || !event.images.length) {
                return;
            }
            let assistant = publishingIntentCardAssistant();
            if (!assistant && conversation) {
                const turn = document.createElement("article");
                turn.className = "conversation-turn image-generation-turn";
                assistant = document.createElement("div");
                assistant.className = "chat-message assistant";
                const label = document.createElement("div");
                label.className = "chat-label";
                label.textContent = "研究助手";
                assistant.append(label);
                turn.append(assistant);
                conversation.append(turn);
                document.querySelector("#conversation-empty")?.setAttribute("hidden", "hidden");
            }
            if (!assistant) {
                return;
            }
            const ids = event.images
                .map((image) => image?.attachment_id)
                .filter((id) => typeof id === "string" && id.trim());
            const key = ids.join(",");
            if (!key || assistant.querySelector(`[data-image-generation-id="${CSS.escape(key)}"]`)) {
                return;
            }

            const card = document.createElement("section");
            card.className = "image-analysis-card image-generation-card";
            card.dataset.imageGenerationId = key;
            card.setAttribute("aria-label", "AI 生成图片");
            const heading = document.createElement("div");
            heading.className = "publishing-intent-card-heading";
            const title = document.createElement("h3");
            title.textContent = "AI 图片已生成";
            const state = document.createElement("span");
            state.className = "publishing-intent-card-state";
            state.textContent = event.model || "图片生成";
            heading.append(title, state);
            card.append(heading);

            const gallery = document.createElement("div");
            gallery.className = "image-generation-gallery";
            for (const image of event.images) {
                if (!image || typeof image.attachment_id !== "string") {
                    continue;
                }
                const preview = document.createElement("img");
                preview.className = "image-generation-preview";
                preview.alt = image.filename || "AI 生成图片";
                preview.src = `/publishing/attachments/images/${encodeURIComponent(image.attachment_id)}/content?thread_id=${encodeURIComponent(publishingThreadId())}`;
                gallery.append(preview);
            }
            card.append(gallery);
            const hint = document.createElement("p");
            hint.className = "publishing-intent-card-status";
            hint.textContent = "图片已进入共享图片库；如果本轮要发布，LLM 会直接使用这些图片，无需再次选择。";
            card.append(hint);
            assistant.append(card);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function publishingIntentTarget(target) {
            if (!target || typeof target !== "object") {
                return null;
            }

            const requiredStrings = [
                "approval_id",
                "article_id",
                "article_title",
                "article_slug",
            ];
            if (
                requiredStrings.some(
                    (field) => (
                        typeof target[field] !== "string" ||
                        !target[field].trim()
                    ),
                ) ||
                !Number.isInteger(target.article_version) ||
                !Object.prototype.hasOwnProperty.call(
                    publishingChannelLabels,
                    target.channel,
                )
            ) {
                return null;
            }

            return target;
        }

        function publishingIntentKey(event) {
            const target = publishingIntentTarget(event?.target);
            if (!target) {
                return [
                    event?.action || "none",
                    event?.resolution_status || "unknown",
                    JSON.stringify(event?.candidates || []),
                ].join(":");
            }

            // A single approval target must have one visible card.  The
            // action may change from approve to resume after a failed or
            // uncertain delivery, but that is a state transition of the same
            // card rather than a second HITL request.
            return [
                target.approval_id || target.article_id,
                target.article_version,
            ].join(":");
        }

        function publishingIntentLabel(action) {
            return {
                approve: "批准发布",
                reject: "拒绝发布",
                request_changes: "要求修改",
                resume: "继续发布",
                status: "查看审批状态",
                none: "发布操作",
            }[action] || "发布操作";
        }

        function beginPublishingRevisionConversation(target, reason) {
            const question = document.querySelector("#question");
            if (!question || !target) {
                return;
            }

            const normalizedReason = typeof reason === "string"
                ? reason.trim()
                : "";
            question.value = normalizedReason
                ? `请修改文章《${target.article_title}》：${normalizedReason}`
                : `请修改文章《${target.article_title}》，我会在下一条消息中说明具体修改要求。`;
            question.focus();
        }

        function appendPublishingResumeAnswer(answer) {
            if (typeof answer !== "string" || !answer.trim()) {
                return;
            }

            const assistant = publishingIntentCardAssistant();
            if (!assistant) {
                return;
            }

            const message = document.createElement("p");
            message.className = "publishing-intent-resume-answer";
            message.textContent = answer;
            assistant.append(message);
            const topStatus = document.querySelector("#status");
            if (topStatus) {
                topStatus.textContent = "完成";
            }
            const traceStatuses = document.querySelectorAll(".trace-status");
            const latestTraceStatus = traceStatuses[traceStatuses.length - 1];
            if (latestTraceStatus) {
                latestTraceStatus.textContent = "已完成";
            }
            conversation.scrollTop = conversation.scrollHeight;
        }

        function publishingIntentInteractionId(event) {
            const interactionId = event?.interaction_id;
            return (
                typeof interactionId === "string" && interactionId.trim()
                    ? interactionId.trim()
                    : null
            );
        }

        function publishingIntentCardAssistant() {
            const turns = conversation?.querySelectorAll(
                ".conversation-turn",
            );
            if (!turns || turns.length === 0) {
                return null;
            }

            return turns[turns.length - 1].querySelector(
                ".chat-message.assistant",
            );
        }

        function finishPublishingIntentCard(card, message, state) {
            if (!card) {
                return;
            }

            card.dataset.state = state;
            const status = card.querySelector(
                ".publishing-intent-card-status",
            );
            if (status) {
                status.textContent = message;
            }

            for (const button of card.querySelectorAll("button")) {
                button.disabled = true;
            }
            const reason = card.querySelector(
                ".publishing-intent-reason",
            );
            if (reason) {
                reason.disabled = true;
            }
        }

        function publishingIntentIdempotencyKey(target) {
            const key = [
                target.article_id,
                target.article_version,
            ].join(":");
            const existing = publishingIntentIdempotencyKeys.get(key);
            if (existing) {
                return existing;
            }

            const random = (
                globalThis.crypto &&
                typeof globalThis.crypto.randomUUID === "function"
                    ? globalThis.crypto.randomUUID()
                    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
            );
            const idempotencyKey = `conversation-${target.article_id}-v${target.article_version}-${random}`;
            publishingIntentIdempotencyKeys.set(key, idempotencyKey);
            return idempotencyKey;
        }

        function hitlDecisionIdempotencyKey(interactionId, action) {
            // Legacy compatibility endpoint remains available server-side:
            // /hitl/interactions/{interaction_id}/resume?thread_id=
            const mapKey = `${interactionId}:${action}`;
            const existing = hitlDecisionIdempotencyKeys.get(mapKey);
            if (existing) {
                return existing;
            }
            const random = (
                globalThis.crypto &&
                typeof globalThis.crypto.randomUUID === "function"
                    ? globalThis.crypto.randomUUID()
                    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
            );
            const key = `conversation-hitl-${interactionId}-${action}-${random}`;
            hitlDecisionIdempotencyKeys.set(mapKey, key);
            return key;
        }

        async function executePublishingIntent(
            card,
            action,
            target,
            interactionId,
        ) {
            if (
                publishingIntentBusy ||
                !card ||
                !target ||
                ![
                    "approve",
                    "reject",
                    "request_changes",
                    "resume",
                ].includes(action)
            ) {
                return;
            }

            if (
                ["approve", "reject", "request_changes"].includes(action) &&
                (!interactionId || !publishingThreadId())
            ) {
                const unavailableStatus = card.querySelector(
                    ".publishing-intent-card-status",
                );
                if (unavailableStatus) {
                    unavailableStatus.textContent =
                        "审批记录暂不可用，请重新发起审批。";
                }
                return;
            }

            if (["reject", "request_changes"].includes(action)) {
                const reason = card.querySelector(
                    ".publishing-intent-reason",
                )?.value.trim() || "";
                if (publishingDecisionReasonRequired && !reason) {
                    const status = card.querySelector(
                        ".publishing-intent-card-status",
                    );
                    if (status) {
                        status.textContent = "请先填写拒绝理由。";
                    }
                    card.querySelector(
                        ".publishing-intent-reason",
                    )?.focus();
                    return;
                }
            }

            publishingIntentBusy = true;
            for (const button of card.querySelectorAll("button")) {
                button.disabled = true;
            }
            const status = card.querySelector(
                ".publishing-intent-card-status",
            );
            if (status) {
                status.textContent = `正在${publishingIntentLabel(action)}...`;
            }

            try {
                let result;
                if (["approve", "reject", "request_changes"].includes(action)) {
                    const decisionResult = await publishingRequest(
                        `/hitl/interactions/${encodeURIComponent(interactionId)}/decide?thread_id=${encodeURIComponent(publishingThreadId())}`,
                        {
                            method: "POST",
                            body: JSON.stringify({
                                decision: action,
                                decision_reason: card.querySelector(
                                    ".publishing-intent-reason",
                                )?.value.trim() || null,
                                idempotency_key: hitlDecisionIdempotencyKey(
                                    interactionId,
                                    action,
                                ),
                            }),
                        },
                    );
                    result = decisionResult.resume;
                } else {
                    const retryablePublicationStatuses = [
                        "failed",
                        "delivery_unknown",
                    ];
                    const isPublicationRetry = (
                        action === "resume" &&
                        retryablePublicationStatuses.includes(
                            target.publication_status,
                        )
                    );
                    if (isPublicationRetry && !target.publication_id) {
                        const missingPublicationError = new Error(
                            "publication_id_missing",
                        );
                        missingPublicationError.code = "publication_id_missing";
                        throw missingPublicationError;
                    }

                    const isDeliveryUnknown = (
                        target.publication_status === "delivery_unknown"
                    );
                    const requestPath = isPublicationRetry
                        ? `/publishing/publications/${encodeURIComponent(target.publication_id)}/retry`
                        : `/publishing/articles/${encodeURIComponent(target.article_id)}/publish`;
                    const requestBody = isPublicationRetry
                        ? {
                            idempotency_key: newPublishingIdempotencyKey({
                                article_id: target.article_id,
                                version: target.article_version,
                            }),
                            confirm_delivery_unknown: isDeliveryUnknown,
                        }
                        : {
                            channel: target.channel,
                            idempotency_key:
                                publishingIntentIdempotencyKey(target),
                        };
                    result = await publishingRequest(
                        requestPath,
                        {
                            method: "POST",
                            body: JSON.stringify(requestBody),
                        },
                    );
                    if (result?.status === "failed") {
                        throw publishingPublicationFailure(result);
                    }
                    if (!publishingPublicationSucceeded(result)) {
                        throw publishingPublicationFailure(result);
                    }
                }

                if (
                    ["approve", "reject", "request_changes"].includes(action)
                ) {
                    const resumeResult = result;
                    if (!resumeResult?.resumed) {
                        const resumeError = new Error(
                            resumeResult?.error_code || "hitl_resume_failed",
                        );
                        resumeError.code = resumeResult?.error_code || "hitl_resume_failed";
                        throw resumeError;
                    }
                    appendPublishingResumeAnswer(resumeResult.answer);
                    if (
                        resumeResult.next_intent &&
                        typeof renderPublishingIntentCard === "function"
                    ) {
                        renderPublishingIntentCard(resumeResult.next_intent);
                    }
                }

                const reason = card.querySelector(
                    ".publishing-intent-reason",
                )?.value.trim() || "";
                const successMessage = action === "approve"
                    ? `HITL 批准已记录，Agent 已继续执行：《${target.article_title}》。`
                    : action === "reject"
                        ? `HITL 拒绝已记录，Agent 已继续处理：《${target.article_title}》。`
                        : action === "request_changes"
                            ? `已记录修改请求：《${target.article_title}》。请在输入框中补充或确认修改要求。`
                            : `《${target.article_title}》${publishingPublicationMessage(result)}`;
                finishPublishingIntentCard(card, successMessage, "success");

                if (action === "request_changes") {
                    beginPublishingRevisionConversation(target, reason);
                }

                if (publishingDialog?.open) {
                    await refreshPublishingCenter();
                }
            } catch (error) {
                publishingIntentBusy = false;
                if (error?.code === "hitl_interaction_stale") {
                    finishPublishingIntentCard(
                        card,
                        "这张审批卡片已失效，请重新发起发布审批。",
                        "stale",
                    );
                    return;
                }
                publishingIntentCards.delete(card.dataset.intentKey);
                if (target && action === "resume") {
                    publishingIntentIdempotencyKeys.delete(
                        [target.article_id, target.article_version].join(":"),
                    );
                }
                for (const button of card.querySelectorAll("button")) {
                    button.disabled = false;
                }
                if (status) {
                    status.textContent = publishingSafeError(error);
                }
                card.dataset.state = "error";
                return;
            }

            publishingIntentBusy = false;
        }

        function renderPublicationAttachmentSelectionCard(event) {
            if (!event || typeof event !== "object") {
                return;
            }
            const assistant = publishingIntentCardAssistant();
            if (!assistant) {
                return;
            }
            const articleId = typeof event.article_id === "string"
                ? event.article_id.trim()
                : "";
            const articleTitle = typeof event.article_title === "string"
                ? event.article_title.trim()
                : "这篇文章";
            const key = `attachments:${articleId}:${event.article_version || ""}:${event.channel || ""}`;
            if (!articleId || publicationAttachmentSelectionCards.has(key)) {
                return;
            }
            publicationAttachmentSelectionCards.add(key);

            const card = document.createElement("section");
            card.className = "publishing-intent-card publication-attachment-selection-card";
            card.dataset.state = "pending";
            card.setAttribute("aria-label", "选择文章配图");

            const heading = document.createElement("div");
            heading.className = "publishing-intent-card-heading";
            const title = document.createElement("h3");
            title.textContent = "选择本次文章配图";
            const state = document.createElement("span");
            state.className = "publishing-intent-card-state";
            state.textContent = publishingChannelLabel(event.channel);
            heading.append(title, state);

            const body = document.createElement("p");
            body.className = "publishing-intent-card-body";
            body.textContent = `《${articleTitle}》已整理完成，请从共享图片库选择本次发布使用的图片。`;
            card.append(heading, body);

            const list = document.createElement("div");
            list.className = "publication-attachment-selection-list";
            const selectedIds = new Set(
                Array.isArray(event.attachment_ids)
                    ? event.attachment_ids.filter(
                        (item) => typeof item === "string" && item.trim(),
                    )
                    : [],
            );
            const images = Array.isArray(event.attachments) ? event.attachments : [];
            for (const image of images) {
                if (!image || typeof image.attachment_id !== "string") {
                    continue;
                }
                const label = document.createElement("label");
                label.className = "publication-attachment-option";
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.value = image.attachment_id;
                checkbox.checked = selectedIds.has(image.attachment_id);
                const preview = document.createElement("img");
                preview.className = "publication-attachment-option-preview";
                preview.alt = image.filename || "文章配图";
                preview.src = `/publishing/attachments/images/${encodeURIComponent(image.attachment_id)}/content?thread_id=${encodeURIComponent(publishingThreadId())}`;
                const filename = document.createElement("span");
                filename.textContent = image.filename || "未命名图片";
                label.append(checkbox, preview, filename);
                list.append(label);
            }
            if (images.length === 0) {
                const empty = document.createElement("p");
                empty.className = "publishing-intent-card-status";
                empty.textContent = "图片库目前为空，也可以暂不使用配图继续发布。";
                list.append(empty);
            }
            card.append(list);

            const status = document.createElement("p");
            status.className = "publishing-intent-card-status";
            status.setAttribute("role", "status");
            status.setAttribute("aria-live", "polite");
            status.textContent = "选择完成后，点击确认，LLM 会继续处理发布流程。";

            const actions = document.createElement("div");
            actions.className = "publishing-intent-card-actions";
            const confirm = document.createElement("button");
            confirm.type = "button";
            confirm.className = "publishing-action publishing-action-primary";
            confirm.textContent = "确认选择并继续";
            confirm.addEventListener("click", () => {
                const attachmentIds = Array.from(
                    card.querySelectorAll(
                        ".publication-attachment-option input[type=checkbox]:checked",
                    ),
                ).map((input) => input.value);
                if (typeof globalThis.setPublicationAttachmentIds === "function") {
                    globalThis.setPublicationAttachmentIds(attachmentIds);
                }
                if (typeof globalThis.submitResearchWithAttachmentSelection !== "function") {
                    status.textContent = "会话提交组件尚未加载，请刷新页面后重试。";
                    return;
                }
                confirm.disabled = true;
                skip.disabled = true;
                status.textContent = "正在把图片选择交给 LLM...";
                const submitted = globalThis.submitResearchWithAttachmentSelection({
                    question: `我已完成《${articleTitle}》的配图选择，请继续处理发布流程。`,
                    attachmentIds,
                    confirmed: true,
                });
                if (!submitted) {
                    confirm.disabled = false;
                    skip.disabled = false;
                    status.textContent = "当前还有请求进行中，请稍后再试。";
                } else {
                    card.dataset.state = "success";
                }
            });
            const skip = document.createElement("button");
            skip.type = "button";
            skip.className = "publishing-action";
            skip.textContent = "暂不使用配图";
            skip.addEventListener("click", () => {
                if (typeof globalThis.setPublicationAttachmentIds === "function") {
                    globalThis.setPublicationAttachmentIds([]);
                }
                if (typeof globalThis.submitResearchWithAttachmentSelection !== "function") {
                    status.textContent = "会话提交组件尚未加载，请刷新页面后重试。";
                    return;
                }
                confirm.disabled = true;
                skip.disabled = true;
                status.textContent = "正在继续发布流程...";
                const submitted = globalThis.submitResearchWithAttachmentSelection({
                    question: `我已确认《${articleTitle}》本次暂不使用配图，请继续处理发布流程。`,
                    attachmentIds: [],
                    confirmed: true,
                });
                if (!submitted) {
                    confirm.disabled = false;
                    skip.disabled = false;
                    status.textContent = "当前还有请求进行中，请稍后再试。";
                } else {
                    card.dataset.state = "success";
                }
            });
            actions.append(confirm, skip);
            card.append(actions, status);
            assistant.append(card);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function renderPublishingIntentCard(event) {
            if (!event || typeof event !== "object") {
                return;
            }

            const assistant = publishingIntentCardAssistant();
            if (!assistant) {
                return;
            }

            const target = publishingIntentTarget(event.target);
            if (
                Array.isArray(event.attachment_ids) &&
                event.attachment_ids.length > 0 &&
                typeof globalThis.setPublicationAttachmentIds === "function"
            ) {
                globalThis.setPublicationAttachmentIds(event.attachment_ids);
            }
            const resolutionStatus = event.resolution_status;
            const action = event.action;
            const interactionId = publishingIntentInteractionId(event);
            const interactionStatus = event.interaction_status;
            const key = publishingIntentKey(event);
            // An approval may be re-emitted by the native interrupt preview,
            // the status tool, or a later recovery turn. Search the whole
            // conversation, otherwise the same target gets one card per
            // assistant message (often leaving an old card without buttons).
            const existingCard = Array.from(
                conversation.querySelectorAll(".publishing-intent-card"),
            ).find((candidate) => (
                candidate.dataset.intentKey === key &&
                candidate.dataset.state === "pending"
            ));
            const signature = JSON.stringify({
                interactionId: interactionId || null,
                interactionStatus: interactionStatus || null,
                resolutionStatus: resolutionStatus || null,
                requiresUserConfirmation: event.requires_user_confirmation === true,
                approvalStatus: target?.approval_status || null,
                publicationStatus: target?.publication_status || null,
            });
            if (existingCard && existingCard.dataset.intentSignature === signature) {
                return;
            }
            if (existingCard) {
                existingCard.remove();
            }
            publishingIntentCards.add(key);

            const card = document.createElement("section");
            card.className = "publishing-intent-card";
            card.dataset.state = "pending";
            card.dataset.intentKey = key;
            card.dataset.intentSignature = signature;
            if (interactionId) {
                card.dataset.interactionId = interactionId;
            }
            card.setAttribute("aria-label", "发布操作确认");

            const heading = document.createElement("div");
            heading.className = "publishing-intent-card-heading";
            const title = document.createElement("h3");
            title.textContent = "发布操作确认";
            const state = document.createElement("span");
            state.className = "publishing-intent-card-state";
            state.textContent = publishingIntentLabel(action);
            heading.append(title, state);

            const body = document.createElement("p");
            body.className = "publishing-intent-card-body";

            if (target?.publication_status === "delivery_unknown" && target) {
                body.textContent = `\u300a${target.article_title}\u300b\u4e0a\u6b21\u53d1\u5e03\u7ed3\u679c\u672a\u77e5\uff08${target.publication_error_code || "delivery_unknown"}\uff09\u3002\u8bf7\u786e\u8ba4\u5e73\u53f0\u6ca1\u6709\u53d1\u5e03\u6210\u529f\u540e\uff0c\u518d\u70b9\u51fb\u7ee7\u7eed\u91cd\u8bd5\uff1b\u5982\u679c\u5e73\u53f0\u5df2\u53d1\u5e03\uff0c\u91cd\u8bd5\u53ef\u80fd\u9020\u6210\u91cd\u590d\u5185\u5bb9\u3002`;
            } else if (target?.publication_status === "failed" && target) {
                const failureCode = target.publication_error_code || "publication_failed";
                body.textContent = `\u300a${target.article_title}\u300b\u7684\u53d1\u5e03\u5931\u8d25\uff08${failureCode}\uff09\u3002\u53ef\u4ee5\u70b9\u51fb\u7ee7\u7eed\u91cd\u8bd5\uff1b\u5982\u679c\u9700\u8981\u4fee\u6539\u6587\u7ae0\uff0c\u8bf7\u5148\u4fee\u6539\u5e76\u91cd\u65b0\u63d0\u4ea4\u5ba1\u6279\u3002`;
            } else if (interactionStatus === "approved" && target) {
                body.textContent = `\u6b64\u524d\u5df2\u6279\u51c6\u300a${target.article_title}\u300b\uff0c\u5f53\u524d\u7b49\u5f85\u540e\u7eed\u53d1\u5e03\u3002`;
            } else if (interactionStatus === "failed" && target) {
                body.textContent = `\u300a${target.article_title}\u300b\u7684\u5ba1\u6279\u540e\u7eed\u5904\u7406\u5931\u8d25\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5\u3002`;
            } else if (interactionStatus === "resuming" && target) {
                body.textContent = `\u300a${target.article_title}\u300b\u7684\u5ba1\u6279\u5df2\u786e\u8ba4\uff0c\u6b63\u5728\u7ee7\u7eed\u5904\u7406\uff0c\u8bf7\u7a0d\u5019\u3002`;
            } else if (resolutionStatus === "resolved" && target) {
                const channelLabel = publishingChannelLabel(target.channel);
                body.textContent = action === "resume"
                    ? `是否继续发布《${target.article_title}》到${channelLabel}？`
                    : action === "reject"
                        ? `是否拒绝《${target.article_title}》的发布审批？`
                        : `是否批准《${target.article_title}》发布到${channelLabel}？`;
            } else if (resolutionStatus === "ambiguous") {
                body.textContent = "匹配到多篇文章，请在对话中提供更准确的标题或 slug。";
            } else if (resolutionStatus === "not_found") {
                body.textContent = "没有找到可以执行该发布操作的文章，请提供更准确的信息。";
            } else if (
                resolutionStatus === "read_only" &&
                Array.isArray(event.candidates) &&
                event.candidates.length > 0
            ) {
                body.textContent = "当前线程有以下待处理的发布操作：";
            } else {
                body.textContent = "当前线程中没有待处理的发布审批。";
            }

            const status = document.createElement("p");
            status.className = "publishing-intent-card-status";
            status.setAttribute("role", "status");
            status.setAttribute("aria-live", "polite");

            card.append(heading, body);

            const candidates = Array.isArray(event.candidates)
                ? event.candidates
                    .map(publishingIntentTarget)
                    .filter(Boolean)
                : [];
            if (candidates.length > 0 && resolutionStatus !== "resolved") {
                const list = document.createElement("ul");
                list.className = "publishing-intent-candidates";
                for (const candidate of candidates) {
                    const item = document.createElement("li");
                    item.textContent = `《${candidate.article_title}》 · v${candidate.article_version}`;
                    list.append(item);
                }
                card.append(list);
            }

            if (
                resolutionStatus === "resolved" &&
                target &&
                event.requires_user_confirmation === true &&
                (action === "resume" || interactionId)
            ) {
                if (["approve", "reject"].includes(action)) {
                    const reason = document.createElement("textarea");
                    reason.className = "publishing-intent-reason";
                    reason.rows = 2;
                    reason.maxLength = 500;
                    reason.placeholder = action === "approve"
                        ? "修改意见（可选）"
                        : "拒绝理由（可选）";
                    reason.setAttribute(
                        "aria-label",
                        action === "approve" ? "修改意见" : "拒绝理由",
                    );
                    card.append(reason);
                }

                const actions = document.createElement("div");
                actions.className = "publishing-intent-card-actions";
                const confirmButton = document.createElement("button");
                confirmButton.type = "button";
                confirmButton.className = "publishing-action publishing-action-primary";
                confirmButton.textContent = action === "resume"
                    ? "确认继续发布"
                    : action === "reject"
                        ? "确认拒绝"
                        : "确认批准";
                confirmButton.addEventListener(
                    "click",
                    () => void executePublishingIntent(
                        card,
                        action,
                        target,
                        interactionId,
                    ),
                );
                actions.append(confirmButton);

                if (action === "approve") {
                    const changesButton = document.createElement("button");
                    changesButton.type = "button";
                    changesButton.className = "publishing-action";
                    changesButton.textContent = "修改文章";
                    changesButton.addEventListener(
                        "click",
                        () => void executePublishingIntent(
                            card,
                            "request_changes",
                            target,
                            interactionId,
                        ),
                    );
                    actions.append(changesButton);
                }
                card.append(actions);
            }

            if (target?.publication_status === "failed") {
                status.textContent = "\u53d1\u5e03\u5931\u8d25\uff0c\u7b49\u5f85\u786e\u8ba4\u540e\u53ef\u91cd\u8bd5\uff1b\u5982\u9700\u4fee\u6539\u8bf7\u91cd\u65b0\u63d0\u4ea4\u5ba1\u6279";
            } else if (target?.publication_status === "delivery_unknown") {
                status.textContent = "\u6295\u9012\u7ed3\u679c\u672a\u77e5\uff0c\u786e\u8ba4\u5e73\u53f0\u672a\u53d1\u5e03\u540e\u518d\u91cd\u8bd5";
            } else if (interactionStatus === "approved") {
                status.textContent = "\u5ba1\u6279\u5df2\u6062\u590d\uff1a\u5df2\u6279\u51c6\uff0c\u7b49\u5f85\u540e\u7eed\u53d1\u5e03\u3002";
            } else if (interactionStatus === "failed") {
                status.textContent = "\u5ba1\u6279\u8bb0\u5f55\u5df2\u6062\u590d\uff0c\u4f46\u540e\u7eed\u5904\u7406\u5931\u8d25\u3002";
            } else if (interactionStatus === "resuming") {
                status.textContent = "\u5ba1\u6279\u6062\u590d\u5904\u7406\u4e2d\uff0c\u8bf7\u7a0d\u5019\uff0c\u4e0d\u8981\u91cd\u590d\u70b9\u51fb\u3002";
            } else if (resolutionStatus === "resolved") {
                status.textContent = (
                    event.requires_user_confirmation === true &&
                    ["approve", "reject"].includes(action) &&
                    !interactionId
                )
                    ? "审批记录尚未就绪，请重新发起审批。"
                    : "等待你的明确确认。";
            } else if (resolutionStatus === "ambiguous") {
                status.textContent = "不会自动选择文章。";
            } else if (resolutionStatus === "not_found") {
                status.textContent = "未执行任何操作。";
            } else {
                status.textContent = "仅展示状态，不执行任何操作。";
            }
            card.append(status);
            assistant.append(card);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function showPublishingDialog() {
            if (!publishingDialog) {
                return;
            }

            if (typeof publishingDialog.showModal === "function") {
                publishingDialog.showModal();
            } else {
                publishingDialog.setAttribute("open", "");
            }

            void refreshPublishingCenter();
        }

        function closePublishingDialog() {
            if (!publishingDialog) {
                return;
            }

            stopPublishingStatusPolling();

            if (typeof publishingDialog.close === "function") {
                publishingDialog.close();
            } else {
                publishingDialog.removeAttribute("open");
            }
        }

        function publishingStatusLabel(status) {
            return publishingStatusLabels[status] || "未知状态";
        }

        function currentPublishingApproval(article) {
            const approvals = publishingApprovals.get(article?.article_id) || [];
            return approvals.find((approval) => (
                approval &&
                article &&
                approval.article_version === article.version &&
                approval.channel === publishingSelectedChannel
            )) || null;
        }

        function renderPublishingApproval(article) {
            if (!publishingApprovalPanel) {
                return;
            }

            const eligible = article && ["approved", "failed"].includes(
                article.status,
            );
            publishingApprovalPanel.hidden = !eligible;
            if (!eligible) {
                if (publishingApprovalState) {
                    publishingApprovalState.textContent = "";
                }
                return;
            }

            const approval = currentPublishingApproval(article);
            // HITL is the canonical workflow state. The legacy approval
            // status is only a projection for old records.
            const status = approval?.workflow_status || approval?.status || "none";
            const isPending = status === "pending";
            const isResuming = status === "resuming";
            const isFailed = status === "failed";
            const isApproved = ["approved", "resumed"].includes(status);
            const isRejected = status === "rejected";

            if (publishingApprovalState) {
                publishingApprovalState.textContent = isPending
                    ? "等待用户审批"
                    : isResuming
                        ? "审批已确认，正在恢复 Agent"
                        : isFailed
                            ? "审批已记录，后续恢复失败"
                            : isApproved
                        ? `已批准（版本 v${approval.article_version}）`
                        : isRejected
                            ? "已拒绝，可重新提交"
                            : "尚未提交发布审批";
            }
            if (publishingRequestApprovalButton) {
                publishingRequestApprovalButton.hidden = (
                    isPending || isApproved || isResuming || isFailed
                );
            }
            if (publishingApprovalApproveButton) {
                publishingApprovalApproveButton.hidden = !isPending;
            }
            if (publishingApprovalRejectButton) {
                publishingApprovalRejectButton.hidden = !isPending;
            }
            if (publishingApprovalReasonField) {
                publishingApprovalReasonField.hidden = !isPending;
            }
            if (publishingPublishButton) {
                publishingPublishButton.hidden = !isApproved;
                publishingPublishButton.textContent = publishingSelectedChannel === "wechat_official_account"
                    ? "提交微信公众号发布"
                    : publishingSelectedChannel === "xiaohongshu"
                        ? "提交小红书发布"
                        : publishingSelectedChannel === "douyin"
                            ? "提交抖音发布"
                            : "发布到本地站点";
            }
        }

        function renderPublishingArticleList() {
            if (!publishingArticleList) {
                return;
            }

            publishingArticleList.replaceChildren();
            const articles = [...publishingArticles.values()];

            if (publishingArticleCount) {
                publishingArticleCount.textContent = `${articles.length} 篇`;
            }

            if (articles.length === 0) {
                const empty = document.createElement("li");
                empty.className = "publishing-empty";
                empty.textContent = "暂无文章草稿";
                publishingArticleList.append(empty);
                return;
            }

            for (const article of articles) {
                if (!article || typeof article.article_id !== "string") {
                    continue;
                }

                const item = document.createElement("li");
                item.className = "publishing-article-item";
                item.dataset.selected = (
                    article.article_id === publishingSelectedArticleId
                        ? "true"
                        : "false"
                );

                const button = document.createElement("button");
                button.type = "button";
                button.className = "publishing-article-select";
                button.addEventListener("click", () => {
                    selectPublishingArticle(article.article_id);
                });

                const title = document.createElement("strong");
                title.className = "publishing-article-title";
                title.textContent = article.title || "未命名文章";

                const meta = document.createElement("span");
                meta.className = "publishing-article-meta";
                meta.textContent = [
                    publishingStatusLabel(article.status),
                    `v${Number(article.version) || 1}`,
                    publishingDate(article.updated_at),
                ].join(" · ");

                button.append(title, meta);
                item.append(button);
                publishingArticleList.append(item);
            }
        }

        function publishingTagsValue() {
            return (publishingTags?.value || "")
                .split(/[，,]/)
                .map((tag) => tag.trim())
                .filter(Boolean)
                .slice(0, 32);
        }

        function setPublishingEditorDisabled(disabled) {
            for (const field of [
                publishingTitle,
                publishingSlug,
                publishingExcerpt,
                publishingTags,
                publishingMarkdown,
                publishingInsertImageButton,
            ]) {
                if (field) {
                    field.disabled = disabled;
                }
            }
        }

        function currentWechatPreviewAttachment() {
            const local = globalThis.getWechatCoverPreview;
            if (typeof local === "function") {
                const attachment = local();
                if (attachment?.is_active) {
                    return attachment;
                }
            }
            if (publishingWechatPreview.attachment_id) {
                return publishingWechatPreview;
            }
            return null;
        }

        function renderPublishingWechatPreview() {
            if (!publishingPreviewTitle) {
                return;
            }
            const article = publishingArticles.get(publishingSelectedArticleId);
            publishingPreviewTitle.textContent = article?.title || publishingTitle?.value || "未命名文章";
            publishingPreviewExcerpt.textContent = `摘要：${article?.excerpt || publishingExcerpt?.value || "无"}`;
            const mode = publishingWechatPreview.publish_mode;
            publishingPreviewMode.textContent = mode === "publish"
                ? "发布模式：正式发布"
                : mode === "draft"
                    ? "发布模式：草稿"
                    : "发布模式：暂不可用";

            const attachment = currentWechatPreviewAttachment();
            const currentThreadId = publishingThreadId();
            if (attachment?.attachment_id && currentThreadId) {
                publishingPreviewCoverName.textContent = `封面文件名：${attachment.filename || "未命名图片"}`;
                publishingPreviewCoverEmpty.textContent = "";
                publishingPreviewCover.hidden = false;
                publishingPreviewCover.src = attachment.local_url || (
                    `/publishing/attachments/images/${encodeURIComponent(attachment.attachment_id)}/content?thread_id=${encodeURIComponent(currentThreadId)}`
                );
                publishingPreviewCover.alt = attachment.filename || "微信公众号封面预览";
                publishingPreviewCover.onerror = () => {
                    publishingPreviewCover.hidden = true;
                };
            } else {
                publishingPreviewCover.hidden = true;
                publishingPreviewCover.removeAttribute("src");
                publishingPreviewCoverName.textContent = "";
                publishingPreviewCoverEmpty.textContent = "请先上传或选择微信公众号封面";
            }
        }

        async function loadPublishingWechatPreview() {
            const currentThreadId = publishingThreadId();
            publishingWechatPreview = { configured: false, publish_mode: null };
            if (!currentThreadId) {
                return;
            }
            try {
                const preview = await publishingRequest(
                    `/publishing/wechat/preview?thread_id=${encodeURIComponent(currentThreadId)}`,
                );
                if (preview && typeof preview === "object") {
                    publishingWechatPreview = preview;
                }
                renderPublishingWechatPreview();
            } catch {
                // The preview remains safe and explicit when the optional
                // WeChat service is not configured.
                renderPublishingWechatPreview();
            }
        }

        function renderPublishingEditor(article, options = {}) {
            if (!article || typeof article.article_id !== "string") {
                stopPublishingStatusPolling();
                publishingForm.hidden = true;
                publishingEmptyEditor.hidden = false;
                if (publishingEditorState) {
                    publishingEditorState.textContent = "未选择文章";
                }
                if (publishingHistoryState) {
                    publishingHistoryState.textContent = "选择文章后查看";
                }
                if (publishingHistoryList) {
                    publishingHistoryList.replaceChildren();
                }
                return;
            }

            publishingForm.hidden = false;
            publishingEmptyEditor.hidden = true;
            publishingTitle.value = article.title || "";
            publishingSlug.value = article.slug || "";
            publishingExcerpt.value = article.excerpt || "";
            publishingTags.value = Array.isArray(article.tags)
                ? article.tags.join(", ")
                : "";
            publishingMarkdown.value = article.markdown_content || "";

            const isDraft = article.status === "draft";
            setPublishingEditorDisabled(!isDraft);
            publishingSaveButton.hidden = !isDraft;
            publishingApproveButton.hidden = !isDraft;
            publishingPublishButton.hidden = true;
            publishingPreviewButton.disabled = false;

            if (publishingEditorState) {
                publishingEditorState.textContent = [
                    publishingStatusLabel(article.status),
                    `版本 v${Number(article.version) || 1}`,
                ].join(" · ");
            }

            if (publishingPreview) {
                publishingPreview.hidden = true;
            }

            renderPublishingApproval(article);
            renderPublishingWechatPreview();
            if (options.loadRemote !== false) {
                void loadPublishingHistory(article.article_id);
                void loadPublishingApprovals(article.article_id);
            }
        }

        function selectPublishingArticle(articleId) {
            const article = publishingArticles.get(articleId);
            if (!article) {
                return;
            }

            publishingSelectedArticleId = articleId;
            renderPublishingArticleList();
            renderPublishingEditor(article);
            restartPublishingStatusPolling(articleId);
        }

        async function runPublishingHistoryAction(
            publication,
            action,
            confirmationMessage,
        ) {
            const article = publishingArticles.get(publishingSelectedArticleId);
            if (
                !article ||
                !publication ||
                typeof publication.publication_id !== "string" ||
                publishingBusy
            ) {
                return;
            }
            if (confirmationMessage && !window.confirm(confirmationMessage)) {
                return;
            }

            publishingBusy = true;
            setPublishingStatus("正在更新发布记录...", "");
            try {
                const result = await action(article, publication);
                if (result?.status === "failed") {
                    throw publishingPublicationFailure(result);
                }
                const refreshed = await publishingRequest(
                    `/publishing/articles/${encodeURIComponent(article.article_id)}`,
                );
                publishingArticles.set(refreshed.article_id, refreshed);
                renderPublishingArticleList();
                renderPublishingEditor(refreshed);
                await loadPublishingHistory(article.article_id);
                setPublishingStatus(
                    publishingPublicationMessage(result),
                    result?.status === "delivery_unknown" ? "warning" : "success",
                );
            } catch (error) {
                await loadPublishingHistory(article.article_id);
                setPublishingStatus(publishingSafeError(error), "error");
            } finally {
                publishingBusy = false;
            }
        }

        async function refreshPublishingPublication(publication) {
            await runPublishingHistoryAction(
                publication,
                (_article, current) => publishingRequest(
                    `/publishing/publications/${encodeURIComponent(current.publication_id)}/refresh`,
                    { method: "POST" },
                ),
            );
        }

        async function retryPublishingPublication(publication) {
            const article = publishingArticles.get(publishingSelectedArticleId);
            if (!article) {
                return;
            }
            const isUnknown = publication.status === "delivery_unknown";
            const confirmationMessage = isUnknown
                ? "请先到微信后台确认这篇文章没有发布成功。确认无误后才会重新提交；如果原请求其实已成功，重试可能产生重复发布。继续吗？"
                : "确认重新发布这篇文章吗？系统会生成一条新的发布记录。";
            await runPublishingHistoryAction(
                publication,
                (current, existing) => publishingRequest(
                    `/publishing/publications/${encodeURIComponent(existing.publication_id)}/retry`,
                    {
                        method: "POST",
                        body: JSON.stringify({
                            idempotency_key: newPublishingIdempotencyKey(current),
                            confirm_delivery_unknown: isUnknown,
                        }),
                    },
                ),
                confirmationMessage,
            );
        }

        async function loadPublishingHistory(articleId, options = {}) {
            if (!publishingHistoryList || !publishingHistoryState) {
                return null;
            }

            if (options.silent !== true) {
                publishingHistoryList.replaceChildren();
                publishingHistoryState.textContent = "加载中...";
            }

            try {
                const publications = await publishingRequest(
                    `/publishing/articles/${encodeURIComponent(articleId)}/publications`,
                );

                if (!Array.isArray(publications)) {
                    throw new Error("publishing_operation_failed");
                }

                publishingHistoryList.replaceChildren();
                publishingHistoryState.textContent = (
                    `${publications.length} 条记录` +
                    (publishingHasAsyncInFlight(publications)
                        ? " · 自动同步中"
                        : "")
                );
                if (publications.length === 0) {
                    const empty = document.createElement("li");
                    empty.className = "publishing-empty";
                    empty.textContent = "暂无发布记录";
                    publishingHistoryList.append(empty);
                    return publications;
                }

                for (const publication of publications) {
                    if (
                        !publication ||
                        typeof publication.publication_id !== "string"
                    ) {
                        continue;
                    }

                    const item = document.createElement("li");
                    item.className = "publishing-history-item";

                    const heading = document.createElement("strong");
                    heading.textContent = [
                        publishingChannelLabel(publication.channel),
                        publishingStatusLabel(publication.status),
                    ].join(" · ");

                    const meta = document.createElement("span");
                    meta.textContent = [
                        `版本 v${Number(publication.article_version) || 1}`,
                        publishingDate(publication.updated_at),
                    ].join(" · ");

                    item.append(heading, meta);

                    if (publication.public_url) {
                        const link = document.createElement("a");
                        link.className = "publishing-public-url";
                        link.textContent = "打开发布地址";
                        const safeUrl = safePublishingUrl(
                            publication.public_url,
                        );
                        if (safeUrl) {
                            link.href = safeUrl;
                            link.target = "_blank";
                            link.rel = "noopener noreferrer";
                            item.append(link);
                        }
                    }

                    if (publication.error_code) {
                        const error = document.createElement("span");
                        error.className = "publishing-history-error";
                        error.textContent = `失败代码：${publication.error_code}`;
                        item.append(error);
                    }

                    if (publication.status === "delivery_unknown") {
                        const warning = document.createElement("span");
                        warning.className = "publishing-history-warning";
                        warning.textContent = `请先人工核对${publishingChannelLabel(publication.channel)}后台，再决定是否重试`;
                        item.append(warning);
                    }

                    const actions = document.createElement("span");
                    actions.className = "publishing-history-actions";
                    if (
                        ["wechat_official_account", "xiaohongshu", "douyin"].includes(publication.channel) &&
                        publication.status === "publishing"
                    ) {
                        const refresh = document.createElement("button");
                        refresh.type = "button";
                        refresh.className = "publishing-history-action";
                        refresh.textContent = "刷新状态";
                        refresh.addEventListener(
                            "click",
                            () => void refreshPublishingPublication(publication),
                        );
                        actions.append(refresh);
                    }
                    if (
                        publication.status === "failed" ||
                        publication.status === "delivery_unknown"
                    ) {
                        const retry = document.createElement("button");
                        retry.type = "button";
                        retry.className = "publishing-history-action";
                        retry.textContent = "重新发布";
                        retry.addEventListener(
                            "click",
                            () => void retryPublishingPublication(publication),
                        );
                        actions.append(retry);
                    }
                    if (actions.childElementCount > 0) {
                        item.append(actions);
                    }

                    publishingHistoryList.append(item);
                }
                return publications;
            } catch (error) {
                publishingHistoryState.textContent = "加载失败";
                if (options.silent !== true) {
                    setPublishingStatus(publishingSafeError(error), "error");
                }
                return null;
            }
        }

        function publishingDialogIsOpen() {
            return Boolean(
                publishingDialog?.open || publishingDialog?.hasAttribute("open"),
            );
        }

        function stopPublishingStatusPolling() {
            if (publishingStatusPollTimer !== null) {
                window.clearInterval(publishingStatusPollTimer);
                publishingStatusPollTimer = null;
            }
            publishingStatusPollInFlight = false;
        }

        function publishingHasAsyncInFlight(publications) {
            return publications.some(
                (publication) => (
                    ["wechat_official_account", "xiaohongshu", "douyin"].includes(publication?.channel) &&
                    publication?.status === "publishing"
                ),
            );
        }

        async function pollPublishingStatus(articleId) {
            if (
                !publishingDialogIsOpen() ||
                publishingSelectedArticleId !== articleId ||
                publishingStatusPollInFlight ||
                publishingBusy
            ) {
                return;
            }

            publishingStatusPollInFlight = true;
            try {
                let publications = await loadPublishingHistory(articleId, {
                    silent: true,
                });
                if (!Array.isArray(publications)) {
                    return;
                }

                const inFlightAsync = publications.find(
                    (publication) => (
                        ["wechat_official_account", "xiaohongshu", "douyin"].includes(publication?.channel) &&
                        publication?.status === "publishing"
                    ),
                );
                if (inFlightAsync) {
                    await publishingRequest(
                        `/publishing/publications/${encodeURIComponent(inFlightAsync.publication_id)}/refresh`,
                        { method: "POST" },
                    );
                    const checked = await loadPublishingHistory(articleId, {
                        silent: true,
                    });
                    if (Array.isArray(checked)) {
                        publications = checked;
                    }
                }

                const latestAsync = publications.find(
                    (publication) => (
                        ["wechat_official_account", "xiaohongshu", "douyin"].includes(publication?.channel)
                    ),
                );
                if (latestAsync) {
                    setPublishingStatus(
                        publishingPublicationMessage(latestAsync),
                        latestAsync.status === "delivery_unknown"
                            ? "warning"
                            : latestAsync.status === "published"
                                ? "success"
                                : latestAsync.status === "failed"
                                    ? "error"
                                    : "",
                    );
                }

                const hasAsyncInFlight = publishingHasAsyncInFlight(publications);
                const refreshed = await publishingRequest(
                    `/publishing/articles/${encodeURIComponent(articleId)}`,
                );
                if (publishingSelectedArticleId !== articleId) {
                    return;
                }
                publishingArticles.set(refreshed.article_id, refreshed);
                renderPublishingArticleList();
                if (publishingEditorState) {
                    publishingEditorState.textContent = [
                        publishingStatusLabel(refreshed.status),
                        `版本 v${Number(refreshed.version) || 1}`,
                    ].join(" · ");
                }
                renderPublishingApproval(refreshed);
                if (!hasAsyncInFlight) {
                    stopPublishingStatusPolling();
                }
            } catch (error) {
                setPublishingStatus(
                    `自动同步失败：${publishingSafeError(error)}`,
                    "error",
                );
            } finally {
                publishingStatusPollInFlight = false;
            }
        }

        function restartPublishingStatusPolling(articleId) {
            stopPublishingStatusPolling();
            if (!articleId || !publishingDialogIsOpen()) {
                return;
            }
            publishingStatusPollTimer = window.setInterval(
                () => void pollPublishingStatus(articleId),
                publishingStatusPollIntervalMs,
            );
        }

        async function loadPublishingApprovals(articleId) {
            if (!articleId) {
                return;
            }

            try {
                const approvals = await publishingRequest(
                    `/publishing/articles/${encodeURIComponent(articleId)}/approval-requests`,
                );
                if (!Array.isArray(approvals)) {
                    throw new Error("publishing_operation_failed");
                }
                publishingApprovals.set(articleId, approvals);
                if (publishingSelectedArticleId === articleId) {
                    renderPublishingApproval(
                        publishingArticles.get(articleId),
                    );
                }
            } catch (error) {
                if (publishingSelectedArticleId === articleId) {
                    setPublishingStatus(publishingSafeError(error), "error");
                }
            }
        }

        function safePublishingUrl(value) {
            if (typeof value !== "string" || !value.trim()) {
                return null;
            }

            try {
                const parsed = new URL(value, window.location.origin);
                if (!['http:', 'https:'].includes(parsed.protocol)) {
                    return null;
                }
                return parsed.href;
            } catch {
                return null;
            }
        }

        async function loadPublishingArtifacts() {
            if (!publishingArtifactSelect || !publishingImportButton) {
                return;
            }

            publishingArtifactSelect.replaceChildren();
            publishingImportButton.disabled = true;
            const currentThreadId = publishingThreadId();

            if (!currentThreadId) {
                const option = document.createElement("option");
                option.textContent = "当前没有会话";
                publishingArtifactSelect.append(option);
                return;
            }

            try {
                const payload = await publishingRequest(
                    `/threads/${encodeURIComponent(currentThreadId)}/artifacts`,
                );
                const artifacts = Array.isArray(payload?.artifacts)
                    ? payload.artifacts
                    : [];

                if (artifacts.length === 0) {
                    const option = document.createElement("option");
                    option.textContent = "当前会话暂无研究报告";
                    publishingArtifactSelect.append(option);
                    return;
                }

                for (const artifact of artifacts) {
                    if (
                        !artifact ||
                        typeof artifact.artifact_id !== "string" ||
                        !/^[0-9a-f]{32}$/i.test(artifact.artifact_id)
                    ) {
                        continue;
                    }

                    const option = document.createElement("option");
                    option.value = artifact.artifact_id;
                    option.textContent = (
                        typeof artifact.filename === "string" &&
                        artifact.filename
                            ? artifact.filename
                            : "未命名研究报告"
                    );
                    publishingArtifactSelect.append(option);
                }

                publishingImportButton.disabled = (
                    publishingArtifactSelect.options.length === 0
                );
            } catch (error) {
                const option = document.createElement("option");
                option.textContent = "研究报告加载失败";
                publishingArtifactSelect.append(option);
                setPublishingStatus(publishingSafeError(error), "error");
            }
        }

        async function loadPublishingArticles() {
            const articles = await publishingRequest(
                "/publishing/articles",
            );
            if (!Array.isArray(articles)) {
                throw new Error("publishing_operation_failed");
            }

            publishingArticles.clear();
            for (const article of articles) {
                if (article && typeof article.article_id === "string") {
                    publishingArticles.set(article.article_id, article);
                }
            }

            if (
                !publishingSelectedArticleId ||
                !publishingArticles.has(publishingSelectedArticleId)
            ) {
                publishingSelectedArticleId = (
                    articles[0]?.article_id || null
                );
            }

            renderPublishingArticleList();
            renderPublishingEditor(
                publishingSelectedArticleId
                    ? publishingArticles.get(publishingSelectedArticleId)
                    : null,
            );
        }

        async function refreshPublishingCenter() {
            if (publishingBusy) {
                return;
            }

            stopPublishingStatusPolling();
            publishingBusy = true;
            setPublishingStatus("正在加载发布中心...");
            try {
                await Promise.all([
                    loadPublishingArticles(),
                    loadPublishingArtifacts(),
                    loadPublishingWechatPreview(),
                ]);
                setPublishingStatus("");
            } catch (error) {
                setPublishingStatus(publishingSafeError(error), "error");
                renderPublishingArticleList();
            } finally {
                publishingBusy = false;
                if (publishingDialogIsOpen()) {
                    restartPublishingStatusPolling(publishingSelectedArticleId);
                }
            }
        }

        async function importPublishingArticle() {
            const artifactId = publishingArtifactSelect?.value || "";
            const currentThreadId = publishingThreadId();
            if (!artifactId || !currentThreadId || publishingBusy) {
                return;
            }

            publishingBusy = true;
            publishingImportButton.disabled = true;
            setPublishingStatus("正在创建文章草稿...");
            try {
                const article = await publishingRequest(
                    "/publishing/articles/from-artifact",
                    {
                        method: "POST",
                        body: JSON.stringify({
                            thread_id: currentThreadId,
                            artifact_id: artifactId,
                        }),
                    },
                );
                publishingArticles.set(article.article_id, article);
                publishingSelectedArticleId = article.article_id;
                renderPublishingArticleList();
                renderPublishingEditor(article);
                setPublishingStatus("文章草稿已创建，等待编辑和人工审批", "success");
            } catch (error) {
                setPublishingStatus(publishingSafeError(error), "error");
            } finally {
                publishingBusy = false;
                await loadPublishingArtifacts();
            }
        }

        function publishingFormPayload() {
            return {
                title: publishingTitle.value.trim(),
                slug: publishingSlug.value.trim(),
                excerpt: publishingExcerpt.value.trim(),
                tags: publishingTagsValue(),
                markdown_content: publishingMarkdown.value,
            };
        }

        async function savePublishingArticle(event) {
            event.preventDefault();
            const article = publishingArticles.get(
                publishingSelectedArticleId,
            );
            if (!article || article.status !== "draft" || publishingBusy) {
                return;
            }

            publishingBusy = true;
            setPublishingStatus("正在保存草稿...");
            try {
                const updated = await publishingRequest(
                    `/publishing/articles/${encodeURIComponent(article.article_id)}`,
                    {
                        method: "PATCH",
                        body: JSON.stringify(publishingFormPayload()),
                    },
                );
                publishingArticles.set(updated.article_id, updated);
                renderPublishingArticleList();
                renderPublishingEditor(updated);
                setPublishingStatus("草稿已保存", "success");
            } catch (error) {
                setPublishingStatus(publishingSafeError(error), "error");
            } finally {
                publishingBusy = false;
            }
        }

        async function approvePublishingArticle() {
            const article = publishingArticles.get(
                publishingSelectedArticleId,
            );
            if (
                !article ||
                article.status !== "draft" ||
                publishingBusy ||
                !window.confirm("确认批准这篇文章吗？批准后将不能直接编辑原版本。")
            ) {
                return;
            }

            publishingBusy = true;
            setPublishingStatus("正在批准文章...");
            try {
                const updated = await publishingRequest(
                    `/publishing/articles/${encodeURIComponent(article.article_id)}/approve`,
                    { method: "POST" },
                );
                publishingArticles.set(updated.article_id, updated);
                renderPublishingArticleList();
                renderPublishingEditor(updated);
                setPublishingStatus("文章已批准，可以发布", "success");
            } catch (error) {
                setPublishingStatus(publishingSafeError(error), "error");
            } finally {
                publishingBusy = false;
            }
        }

        async function requestPublishingApproval() {
            const article = publishingArticles.get(
                publishingSelectedArticleId,
            );
            if (
                !article ||
                !["approved", "failed"].includes(article.status) ||
                publishingBusy ||
                !window.confirm("确认提交这篇文章的发布审批吗？")
            ) {
                return;
            }

            publishingBusy = true;
            setPublishingStatus("正在提交发布审批...");
            try {
                await publishingRequest(
                    `/publishing/articles/${encodeURIComponent(article.article_id)}/approval-requests`,
                    {
                        method: "POST",
                        body: JSON.stringify({
                            channel: publishingSelectedChannel,
                        }),
                    },
                );
                await loadPublishingApprovals(article.article_id);
                setPublishingStatus("发布审批已提交，等待用户确认", "success");
            } catch (error) {
                setPublishingStatus(publishingSafeError(error), "error");
            } finally {
                publishingBusy = false;
            }
        }

        async function approvePublishingRequest() {
            const article = publishingArticles.get(
                publishingSelectedArticleId,
            );
            const approval = currentPublishingApproval(article);
            if (
                !approval ||
                approval.status !== "pending" ||
                publishingBusy ||
                !window.confirm("确认批准这篇文章发布吗？批准后将允许执行发布。")
            ) {
                return;
            }

            publishingBusy = true;
            setPublishingStatus("正在批准发布审批...");
            try {
                await publishingRequest(
                    `/publishing/approval-requests/${encodeURIComponent(approval.approval_id)}/approve`,
                    {
                        method: "POST",
                        body: JSON.stringify({}),
                    },
                );
                await loadPublishingApprovals(article.article_id);
                setPublishingStatus("发布审批已批准，可以发布", "success");
            } catch (error) {
                setPublishingStatus(publishingSafeError(error), "error");
            } finally {
                publishingBusy = false;
            }
        }

        async function rejectPublishingRequest() {
            const article = publishingArticles.get(
                publishingSelectedArticleId,
            );
            const approval = currentPublishingApproval(article);
            const reason = publishingApprovalReason?.value.trim() || "";
            if (
                !approval ||
                approval.status !== "pending" ||
                publishingBusy
            ) {
                return;
            }
            if (publishingDecisionReasonRequired) {
                setPublishingStatus("请填写拒绝理由", "error");
                publishingApprovalReason?.focus();
                return;
            }
            if (!window.confirm("确认拒绝这篇文章发布吗？")) {
                return;
            }

            publishingBusy = true;
            setPublishingStatus("正在拒绝发布审批...");
            try {
                await publishingRequest(
                    `/publishing/approval-requests/${encodeURIComponent(approval.approval_id)}/reject`,
                    {
                        method: "POST",
                        body: JSON.stringify({
                            decision_reason: reason || null,
                        }),
                    },
                );
                if (publishingApprovalReason) {
                    publishingApprovalReason.value = "";
                }
                await loadPublishingApprovals(article.article_id);
                setPublishingStatus("发布审批已拒绝", "success");
            } catch (error) {
                setPublishingStatus(publishingSafeError(error), "error");
            } finally {
                publishingBusy = false;
            }
        }

        function publishingIdempotencyKey(article) {
            const existing = publishingIdempotencyKeys.get(article.article_id);
            if (existing) {
                return existing;
            }

            const random = (
                globalThis.crypto &&
                typeof globalThis.crypto.randomUUID === "function"
                    ? globalThis.crypto.randomUUID()
                    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
            );
            const key = `ui-${article.article_id}-v${article.version}-${random}`;
            publishingIdempotencyKeys.set(article.article_id, key);
            return key;
        }

        function newPublishingIdempotencyKey(article) {
            const random = (
                globalThis.crypto &&
                typeof globalThis.crypto.randomUUID === "function"
                    ? globalThis.crypto.randomUUID()
                    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
            );
            return `ui-retry-${article.article_id}-v${article.version}-${random}`;
        }

        async function publishPublishingArticle() {
            const article = publishingArticles.get(
                publishingSelectedArticleId,
            );
            const approval = currentPublishingApproval(article);
            if (
                !article ||
                !["approved", "failed"].includes(article.status) ||
                !approval ||
                approval.status !== "approved" ||
                publishingBusy ||
                !window.confirm(
                    `确认发布这篇文章到${publishingChannelLabel(publishingSelectedChannel)}吗？此操作会生成一条发布记录。`,
                )
            ) {
                return;
            }

            publishingBusy = true;
            setPublishingStatus(`正在提交到${publishingChannelLabel(publishingSelectedChannel)}...`);
            try {
                const publication = await publishingRequest(
                    `/publishing/articles/${encodeURIComponent(article.article_id)}/publish`,
                    {
                        method: "POST",
                        body: JSON.stringify({
                            channel: publishingSelectedChannel,
                            idempotency_key: publishingIdempotencyKey(article),
                            attachment_ids: typeof globalThis.getSelectedPublicationAttachmentIds === "function"
                                ? globalThis.getSelectedPublicationAttachmentIds()
                                : [],
                        }),
                    },
                );
                if (publication?.status === "failed") {
                    throw publishingPublicationFailure(publication);
                }
                if (!publishingPublicationSucceeded(publication)) {
                    throw publishingPublicationFailure(publication);
                }
                const updated = await publishingRequest(
                    `/publishing/articles/${encodeURIComponent(article.article_id)}`,
                );
                publishingArticles.set(updated.article_id, updated);
                renderPublishingArticleList();
                renderPublishingEditor(updated);
                setPublishingStatus(
                    publishingPublicationMessage(publication),
                    publication.status === "delivery_unknown" ? "warning" : "success",
                );
            } catch (error) {
                try {
                    const refreshed = await publishingRequest(
                        `/publishing/articles/${encodeURIComponent(article.article_id)}`,
                    );
                    publishingArticles.set(refreshed.article_id, refreshed);
                    if (refreshed.status === "failed") {
                        publishingIdempotencyKeys.delete(article.article_id);
                    }
                    renderPublishingArticleList();
                    renderPublishingEditor(refreshed);
                } catch {
                    // Keep the current editor usable if the status refresh fails.
                }
                setPublishingStatus(publishingSafeError(error), "error");
                await loadPublishingHistory(article.article_id);
            } finally {
                publishingBusy = false;
            }
        }

        function previewPublishingArticle() {
            if (!publishingPreview || !publishingPreviewContent) {
                return;
            }

            publishingPreview.hidden = false;
            if (typeof renderMarkdownInto === "function") {
                const currentThreadId = publishingThreadId();
                const previewMarkdown = currentThreadId
                    ? publishingMarkdown.value.replace(
                        /!\[([^\]]*)\]\(\s*attachment:\/\/([A-Za-z0-9_-]{1,128})\s*\)/g,
                        (_match, alt, attachmentId) => (
                            `![${alt}](/publishing/attachments/images/${encodeURIComponent(attachmentId)}/content?thread_id=${encodeURIComponent(currentThreadId)})`
                        ),
                    )
                    : publishingMarkdown.value;
                renderMarkdownInto(
                    publishingPreviewContent,
                    previewMarkdown,
                );
            } else {
                publishingPreviewContent.textContent = publishingMarkdown.value;
            }
        }

        function insertPublishingImage(attachment) {
            if (!publishingMarkdown || !attachment?.attachment_id) {
                return;
            }
            const attachmentId = String(attachment.attachment_id).trim();
            if (!attachmentId) {
                return;
            }
            const alt = String(attachment.filename || "图片")
                .replace(/[\[\]]/g, "")
                .trim() || "图片";
            const snippet = `![${alt}](attachment://${attachmentId})`;
            const start = Number.isInteger(publishingMarkdown.selectionStart)
                ? publishingMarkdown.selectionStart
                : publishingMarkdown.value.length;
            const end = Number.isInteger(publishingMarkdown.selectionEnd)
                ? publishingMarkdown.selectionEnd
                : start;
            const before = publishingMarkdown.value.slice(0, start);
            const after = publishingMarkdown.value.slice(end);
            const prefix = before && !/[\n\s]$/.test(before) ? "\n\n" : "";
            const suffix = after && !/^[\n\s]/.test(after) ? "\n\n" : "";
            publishingMarkdown.value = `${before}${prefix}${snippet}${suffix}${after}`;
            const cursor = before.length + prefix.length + snippet.length;
            publishingMarkdown.focus();
            publishingMarkdown.setSelectionRange(cursor, cursor);
            publishingMarkdown.dispatchEvent(new Event("input", { bubbles: true }));
        }

        publishingOpenButton?.addEventListener(
            "click",
            showPublishingDialog,
        );
        publishingCloseButton?.addEventListener(
            "click",
            closePublishingDialog,
        );
        publishingRefreshButton?.addEventListener(
            "click",
            () => void refreshPublishingCenter(),
        );
        publishingImportButton?.addEventListener(
            "click",
            () => void importPublishingArticle(),
        );
        publishingForm?.addEventListener(
            "submit",
            (event) => void savePublishingArticle(event),
        );
        publishingPreviewButton?.addEventListener(
            "click",
            previewPublishingArticle,
        );
        publishingInsertImageButton?.addEventListener("click", () => {
            const openPicker = globalThis.openInlineImagePicker;
            if (typeof openPicker !== "function") {
                setPublishingStatus("图片库暂不可用，请刷新页面后重试", "error");
                return;
            }
            openPicker(insertPublishingImage);
        });
        publishingPreviewClose?.addEventListener(
            "click",
            () => {
                publishingPreview.hidden = true;
            },
        );
        publishingApproveButton?.addEventListener(
            "click",
            () => void approvePublishingArticle(),
        );
        publishingPublishButton?.addEventListener(
            "click",
            () => void publishPublishingArticle(),
        );
        publishingRequestApprovalButton?.addEventListener(
            "click",
            () => void requestPublishingApproval(),
        );
        publishingApprovalApproveButton?.addEventListener(
            "click",
            () => void approvePublishingRequest(),
        );
        publishingApprovalRejectButton?.addEventListener(
            "click",
            () => void rejectPublishingRequest(),
        );
        publishingChannel?.addEventListener("change", () => {
            publishingSelectedChannel = [
                "local_static_site",
                "wechat_official_account",
                "xiaohongshu",
                "douyin",
            ].includes(publishingChannel.value)
                ? publishingChannel.value
                : "local_static_site";
            renderPublishingApproval(
                publishingArticles.get(publishingSelectedArticleId),
            );
        });
        document.addEventListener("wechat-cover-changed", (event) => {
            const changedThreadId = event.detail?.thread_id;
            if (changedThreadId && changedThreadId === publishingThreadId()) {
                const attachment = event.detail?.attachment;
                publishingWechatPreview = {
                    ...publishingWechatPreview,
                    attachment_id: attachment?.attachment_id || null,
                    filename: attachment?.filename || null,
                };
                renderPublishingWechatPreview();
            }
        });
        document.addEventListener("research:thread-changed", () => {
            if (publishingDialog?.open) {
                void refreshPublishingCenter();
            } else {
                void loadPublishingWechatPreview();
            }
        });
        document.addEventListener("research:thread-restored", (event) => {
            if (!event.detail?.thread_id || event.detail.thread_id === publishingThreadId()) {
                void restorePendingPublishingHITL();
            }
        });

        // Keep the event consumer resilient across classic-script load order
        // and make the cross-file contract explicit.
        globalThis.renderPublishingIntentCard = renderPublishingIntentCard;
        globalThis.restorePendingPublishingHITL = restorePendingPublishingHITL;
        globalThis.renderPublicationAttachmentSelectionCard = renderPublicationAttachmentSelectionCard;
        globalThis.renderImageAnalysisCard = renderImageAnalysisCard;
        globalThis.renderGeneratedImageCard = renderGeneratedImageCard;
        // thread-view initializes before this classic script on a hard
        // refresh, so run one delayed recovery in addition to the restore
        // event used by later thread switches.
        window.setTimeout(() => void restorePendingPublishingHITL(), 0);
