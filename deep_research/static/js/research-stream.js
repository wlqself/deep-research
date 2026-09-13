        let pendingAttachmentIds = null;
        let pendingAttachmentSelectionConfirmed = false;

        function formatToolStart(event) {
            const name = event.name || "unknown_tool";
            const input = event.input || {};

            if (name === "web_search") {
                return `开始 web_search - 查询：${input.query || ""}`;
            }

            if (name === "read_page") {
                return `开始 read_page - source_id：${input.source_id || ""}`;
            }

            if (name === "assess_research") {
                return `开始 assess_research - 下一步：${input.next_action || ""}`;
            }

            if (name === "write_todos") {
                return "更新研究计划";
            }

            if (name === "prepare_article_for_publication") {
                return "开始整理文章草稿";
            }

            return `开始 ${name}`;
        }

        newThreadButton.addEventListener(
            "click",
            startNewThread,
        );

        threadContextMenu?.addEventListener(
            "click",
            (event) => {
                const button = event.target.closest(
                    "[data-thread-action]",
                );

                if (!button || !contextThreadId) {
                    return;
                }

                const action = button.dataset.threadAction;
                const targetId = contextThreadId;
                closeThreadContextMenu();

                if (action === "rename") {
                    openThreadRenameDialog(targetId);
                } else if (action === "delete") {
                    void deleteThread(targetId);
                }
            },
        );

        threadRenameForm?.addEventListener(
            "submit",
            (event) => {
                void submitThreadRename(event);
            },
        );

        threadRenameCancel?.addEventListener(
            "click",
            closeThreadRenameDialog,
        );

        document.addEventListener("pointerdown", (event) => {
            if (
                threadContextMenu?.hidden === false &&
                !threadContextMenu.contains(event.target)
            ) {
                closeThreadContextMenu();
            }
        });

        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") {
                closeThreadContextMenu();
            }
        });

        knowledgeOpenButton.addEventListener(
            "click",
            openKnowledgeDialog,
        );

        memoryOpenButton?.addEventListener(
            "click",
            openMemoryDialog,
        );

        memoryCloseButton?.addEventListener(
            "click",
            closeMemoryDialog,
        );

        memorySearchButton?.addEventListener(
            "click",
            () => {
                memoryPage = 1;
                void loadMemories();
            },
        );

        memorySearchInput?.addEventListener(
            "keydown",
            (event) => {
                if (event.key !== "Enter") {
                    return;
                }

                event.preventDefault();
                memoryPage = 1;
                void loadMemories();
            },
        );

        memoryTabs?.addEventListener(
            "click",
            (event) => {
                const button = event.target.closest("[data-kind]");

                if (!button) {
                    return;
                }

                memoryKind = button.dataset.kind || "";
                memoryPage = 1;

                for (const tab of memoryTabs.querySelectorAll("[data-kind]")) {
                    tab.classList.toggle("is-active", tab === button);
                }

                void loadMemories();
            },
        );

        memoryPrevButton?.addEventListener(
            "click",
            () => {
                if (memoryPage <= 1) {
                    return;
                }

                memoryPage -= 1;
                void loadMemories();
            },
        );

        memoryNextButton?.addEventListener(
            "click",
            () => {
                if (memoryNextButton.disabled) {
                    return;
                }

                memoryPage += 1;
                void loadMemories();
            },
        );

        memoryClearButton?.addEventListener(
            "click",
            () => void clearMemories(),
        );

        memoryFormCancel?.addEventListener(
            "click",
            closeMemoryEditor,
        );

        memoryForm?.addEventListener(
            "submit",
            submitMemoryEdit,
        );

        knowledgeCloseButton.addEventListener(
            "click",
            closeKnowledgeDialog,
        );

        knowledgeFileInput.addEventListener("change", () => {
            const file = knowledgeFileInput.files?.[0];
            void uploadKnowledgeDocument(file);
        });

        knowledgeRefreshButton.addEventListener(
            "click",
            () => {
                void loadKnowledgeDocuments();
            },
        );

        threadSettingsButton.addEventListener(
            "click",
            () => {
                void openThreadSettings();
            },
        );

        settingsCloseButton.addEventListener("click", () => {
            if (typeof settingsDialog.close === "function") {
                settingsDialog.close();
            } else {
                settingsDialog.removeAttribute("open");
            }
        });

        renderThreadList();
        void restoreThreadView(threadId);
        void loadKnowledgeDocuments();

        for (const button of promptButtons) {
            button.addEventListener("click", () => {
                questionInput.value = button.dataset.prompt;
                questionInput.focus();
            });
        }

        questionInput.addEventListener("keydown", (event) => {
            if (
                event.key !== "Enter" ||
                event.shiftKey ||
                event.isComposing
            ) {
                return;
            }

            event.preventDefault();

            if (
                submitButton.disabled ||
                controller
            ) {
                return;
            }

            form.requestSubmit();
        });

        cancelButton.addEventListener("click", () => {
            if (!controller) {
                return;
            }

            cancellationRequested = true;
            markRunningTasksCancelled();
            recordActivityFromStreamEvent({ type: "cancel_requested" });
            status.textContent = "正在停止";
            traceStatus.textContent = "正在停止";
            controller?.abort();
        });

        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            const question = questionInput.value.trim();

            if (!question) {
                questionInput.focus();
                return;
            }

            if (
                typeof globalThis.hasPendingImageUploads === "function" &&
                globalThis.hasPendingImageUploads()
            ) {
                status.textContent = "图片仍在上传，请稍候再发送";
                questionInput.focus();
                return;
            }

            controller = new AbortController();
            const automaticallyAttachedIds = typeof globalThis.getPendingMessageAttachmentIds === "function"
                ? globalThis.getPendingMessageAttachmentIds()
                : [];
            // A publication/HITL selection is an explicit override.  A
            // stale empty selection must not hide images the user just
            // attached in the composer.
            const requestAttachmentIds = (
                Array.isArray(pendingAttachmentIds) &&
                pendingAttachmentSelectionConfirmed
            )
                ? pendingAttachmentIds
                : automaticallyAttachedIds;
            const requestAttachmentSelectionConfirmed = pendingAttachmentSelectionConfirmed;
            pendingAttachmentIds = null;
            pendingAttachmentSelectionConfirmed = false;
            if (typeof globalThis.clearPendingMessageAttachmentIds === "function") {
                globalThis.clearPendingMessageAttachmentIds();
            }
            cancellationRequested = false;
            setBusy(true);
            status.textContent = "正在检索";
            appendConversationTurn(question, requestAttachmentIds);
            questionInput.value = "";
            answer.textContent = "";
            resetTrace();
            resetPlan();
            resetTasks();
            resetMemoryNotice();

            try {
                const response = await fetch("/research/stream", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        question,
                        thread_id: threadId,
                        attachment_ids: requestAttachmentIds,
                        attachment_selection_confirmed: requestAttachmentSelectionConfirmed,
                    }),
                    signal: controller.signal,
                });

                if (!response.ok || !response.body) {
                    let errorCode = "research_request_failed";
                    let errorMessage = "Research request failed";
                    try {
                        const payload = await response.json();
                        const detail = payload?.detail;
                        if (typeof detail === "string") {
                            errorMessage = detail;
                        } else if (detail && typeof detail === "object") {
                            errorCode = detail.error_code || errorCode;
                            errorMessage = detail.message || errorMessage;
                        }
                    } catch {
                        // Keep the stable fallback for non-JSON server errors.
                    }
                    const requestError = new Error(errorMessage);
                    requestError.code = errorCode;
                    throw requestError;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = "";
                let pendingLine = "";
                let researchSucceeded = false;

                function consumeEventLine(line) {
                    const trimmedLine = line.trim();

                    if (!trimmedLine) {
                        return;
                    }

                    let event;

                    try {
                        event = JSON.parse(trimmedLine);
                    } catch {
                        addTraceItem("无法解析服务器事件", "failed");
                        return;
                    }

                    if (event.type === "activity") {
                        appendActivityEvent(event.activity);
                        return;
                    }

                    if (event.type === "plan_update") {
                        renderPlan(event.todos);
                        return;
                    }

                    if (
                        event.type === "subagent_start" ||
                        event.type === "subagent_end"
                    ) {
                        updateResearchTask(event);
                        return;
                    }

            if (event.type === "memory_compacted") {
                showMemoryCompactedNotice();
                return;
            }

            if (event.type === "memory_recall_started") {
                status.textContent = "正在检索长期记忆";
                addTraceItem("正在检索长期记忆", "running");
                return;
            }

            if (event.type === "memory_recall_completed") {
                addTraceItem("长期记忆检索完成", "completed");
                return;
            }

            if (event.type === "model_started") {
                if (event.agent_name === "main") {
                    status.textContent = "正在分析请求";
                    addTraceItem("正在分析请求", "running");
                }
                return;
            }

            if (event.type === "model_reasoning") {
                if (event.agent_name === "main") {
                    addTraceItem(String(event.message || "正在分析请求"), "running");
                }
                return;
            }

            if (event.type === "model_retrying") {
                addTraceItem(`模型响应异常，正在进行第 ${event.attempt || "?"} 次重试`, "running");
                return;
            }

            if (event.type === "first_token") {
                answer.classList.add("is-streaming");
                status.textContent = "正在生成回答";
                return;
            }

            if (event.type === "heartbeat") {
                status.textContent = "仍在处理中";
                return;
            }

            if (event.type === "model_completed") {
                if (event.agent_name === "main") {
                    addTraceItem("模型阶段完成", "completed");
                }
                return;
            }

            if (event.type === "model_failed") {
                addTraceItem("模型调用失败", "failed");
                status.textContent = "请求失败";
                return;
            }

            if (event.type === "waiting_for_user") {
                status.textContent = "等待用户操作";
                traceStatus.textContent = "等待确认";
                return;
            }

            if (event.type === "publishing_intent") {
                if (typeof renderPublishingIntentCard === "function") {
                    renderPublishingIntentCard(event);
                } else {
                    addTraceItem(
                        "发布确认卡片组件尚未加载，请刷新页面后重试",
                        "failed",
                    );
                }
                addTraceItem(
                    event.resolution_status === "resolved"
                        ? "已找到发布操作目标，等待用户确认"
                        : "发布操作未唯一确定，等待用户补充信息",
                    event.resolution_status === "resolved"
                        ? "running"
                        : "failed",
                );
                return;
            }

            if (event.type === "publication_attachment_selection") {
                if (typeof renderPublicationAttachmentSelectionCard === "function") {
                    renderPublicationAttachmentSelectionCard(event);
                } else {
                    addTraceItem(
                        "图片选择卡片组件尚未加载，请刷新页面后重试",
                        "failed",
                    );
                }
                addTraceItem("等待选择本次文章配图", "running");
                return;
            }

            if (event.type === "image_generated") {
                if (typeof renderGeneratedImageCard === "function") {
                    renderGeneratedImageCard(event);
                }
                addTraceItem("图片已生成并保存到图片库", "completed");
                return;
            }

            if (event.type === "text") {
                        fullText += event.text || "";
                        answer.textContent = fullText;
                        answer.scrollTop = answer.scrollHeight;
                        return;
                    }

                    if (event.type === "tool_start") {
                        addTraceItem(formatToolStart(event), "running");
                        return;
                    }

                    if (event.type === "tool_end") {
                        const name = event.name || "unknown_tool";
                        const failed = event.status === "failed";

                        if (name === "prepare_article_for_publication") {
                            addTraceItem(
                                failed
                                    ? "文章草稿整理失败"
                                    : "文章草稿整理完成",
                                failed ? "failed" : "completed",
                            );
                            return;
                        }

                        addTraceItem(
                            `${failed ? "失败" : "完成"} ${name}`,
                            failed ? "failed" : "completed",
                        );
                        return;
                    }

                    if (event.type === "artifact_saved") {
                        renderArtifactSaved(event);
                        return;
                    }

                    if (event.type === "error") {
                        const knownErrorMessages = {
                            model_repetition_detected: "模型输出出现重复，已停止本轮生成",
                            model_output_limit_exceeded: "模型输出超过安全长度，已停止本轮生成",
                            non_retryable_tool_loop_detected: "连续工具调用没有取得进展，已停止本轮处理",
                            publication_repair_exhausted: "文章自动修复后仍未通过平台校验，请修改后重试",
                            tool_retry_exhausted: "工具连续重试仍未成功，请稍后重试",
                            xiaohongshu_title_too_long: "小红书标题超过长度限制，请缩短标题后重试",
                            xiaohongshu_content_too_long: "小红书正文超过1000字限制，请精简正文后重试",
                            douyin_title_too_long: "抖音标题超过长度限制，请缩短标题后重试",
                            wechat_title_too_long: "微信公众号标题超过长度限制，请缩短标题后重试",
                        };
                        const errorMessage = knownErrorMessages[event.code]
                            || event.message
                            || "研究服务失败";
                        addTraceItem(
                            errorMessage,
                            "failed",
                        );
                        traceStatus.textContent = "失败";
                        status.textContent = "请求失败";
                        return;
                    }

                    if (event.type === "done") {
                        if (cancellationRequested) {
                            return;
                        }
                        const goalStatus = event.goal_status || "completed";
                        if (goalStatus === "completed") {
                            researchSucceeded = true;
                            traceStatus.textContent = "已完成";
                        } else if (goalStatus === "waiting_for_user") {
                            researchSucceeded = false;
                            traceStatus.textContent = "等待确认";
                            status.textContent = "等待用户确认";
                        } else {
                            researchSucceeded = false;
                            traceStatus.textContent = "目标未完成";
                            status.textContent = "处理结束，但目标未完成";
                        }
                        return;
                    }

                    if (event.type === "cancelled") {
                        cancellationRequested = true;
                        status.textContent = "已取消";
                        traceStatus.textContent = "已取消";
                    }
                }

                while (true) {
                    const { value, done } = await reader.read();

                    if (done) {
                        break;
                    }

                    pendingLine += decoder.decode(value, { stream: true });

                    const lines = pendingLine.split("\n");
                    pendingLine = lines.pop() || "";

                    for (const line of lines) {
                        consumeEventLine(line);
                    }
                }

                pendingLine += decoder.decode();

                if (pendingLine.trim()) {
                    consumeEventLine(pendingLine);
                }

                answer.classList.remove("is-streaming");
                renderMarkdown(fullText);

                if (
                    researchSucceeded &&
                    fullText.trim() &&
                    traceStatus.textContent !== "失败"
                ) {
                    persistSuccessfulThread(question);
                    status.textContent = "完成";
                }
            } catch (error) {
                if (error.name === "AbortError") {
                    cancellationRequested = true;
                    recordActivityFromStreamEvent({ type: "cancelled" });
                    status.textContent = "已取消";
                    traceStatus.textContent = "已取消";
                } else {
                    recordActivityFromStreamEvent({
                        type: "error",
                        error: error.message,
                    });
                    if (typeof globalThis.restorePendingMessageAttachmentIds === "function") {
                        globalThis.restorePendingMessageAttachmentIds(requestAttachmentIds);
                    }
                    if (error.code === "hitl_pending") {
                        status.textContent = "请先处理会话中的审批卡片";
                        traceStatus.textContent = "等待确认";
                    } else {
                        status.textContent = "请求失败";
                        traceStatus.textContent = "失败";
                    }
                    answer.classList.remove("is-streaming");
                    answer.textContent = "研究请求失败，请稍后重试。";
                }
            } finally {
                setBusy(false);
                controller = null;
            }
        });

        globalThis.submitResearchWithAttachmentSelection = ({
            question,
            attachmentIds,
            confirmed = true,
        } = {}) => {
            if (
                typeof question !== "string" ||
                !question.trim() ||
                controller ||
                submitButton.disabled
            ) {
                return false;
            }
            pendingAttachmentIds = Array.isArray(attachmentIds)
                ? attachmentIds.filter(
                    (item) => typeof item === "string" && item.trim(),
                )
                : [];
            pendingAttachmentSelectionConfirmed = confirmed === true;
            questionInput.value = question.trim();
            form.requestSubmit();
            return true;
        };
