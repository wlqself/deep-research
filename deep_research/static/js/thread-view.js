        function renderMarkdownInto(target, markdown) {
            if (
                typeof marked === "undefined" ||
                typeof DOMPurify === "undefined"
            ) {
                target.textContent = markdown;
                return;
            }

            try {
                const html = marked.parse(markdown, {
                    gfm: true,
                    breaks: false,
                });

                target.innerHTML = DOMPurify.sanitize(html, {
                    USE_PROFILES: { html: true },
                });

                for (const link of target.querySelectorAll("a")) {
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                }
            } catch {
                target.textContent = markdown;
            }
        }

        function createRestoredPlanPanel(todos) {
            if (!Array.isArray(todos) || todos.length === 0) {
                return null;
            }

            const labels = {
                pending: "待处理",
                in_progress: "进行中",
                completed: "已完成",
            };

            const panel = document.createElement("section");
            panel.className = "plan-panel";
            panel.setAttribute("aria-label", "研究计划");

            const heading = document.createElement("div");
            heading.className = "plan-heading";

            const title = document.createElement("h3");
            title.textContent = "研究计划";

            const status = document.createElement("span");
            status.className = "plan-status";

            const list = document.createElement("ol");
            list.className = "plan-list";

            let completedCount = 0;

            for (const todo of todos) {
                if (
                    !todo ||
                    typeof todo.content !== "string" ||
                    typeof todo.status !== "string"
                ) {
                    continue;
                }

                const item = document.createElement("li");
                item.className = "plan-item";
                item.dataset.status = todo.status;

                const content = document.createElement("span");
                content.textContent = todo.content;

                const itemStatus = document.createElement("span");
                itemStatus.className = "plan-item-status";
                itemStatus.textContent =
                    labels[todo.status] || "未知";

                item.append(content, itemStatus);
                list.append(item);

                if (todo.status === "completed") {
                    completedCount += 1;
                }
            }

            if (list.children.length === 0) {
                return null;
            }

            status.textContent =
                completedCount === list.children.length
                    ? `已完成 ${completedCount}/${list.children.length}`
                    : `进行中 ${completedCount}/${list.children.length}`;

            heading.append(title, status);
            panel.append(heading, list);
            return panel;
        }

        function createRestoredTaskPanel(tasks) {
            if (!Array.isArray(tasks)) {
                return null;
            }

            const terminalTasks = tasks.filter(
                (task) => (
                    task &&
                    typeof task === "object" &&
                    typeof task.task_id === "string" &&
                    typeof task.status === "string" &&
                    ["completed", "failed", "cancelled"].includes(
                        task.status,
                    )
                ),
            );

            if (terminalTasks.length === 0) {
                return null;
            }

            const labels = {
                completed: "已完成",
                failed: "失败",
                cancelled: "已取消",
            };
            const panel = document.createElement("section");
            panel.className = "task-panel";
            panel.setAttribute("aria-label", "研究任务");

            const heading = document.createElement("div");
            heading.className = "task-heading";

            const title = document.createElement("h3");
            title.textContent = "研究任务";

            const status = document.createElement("span");
            status.className = "task-status";
            const completedCount = terminalTasks.filter(
                (task) => task.status === "completed",
            ).length;
            status.textContent = `已结束 ${completedCount}/${terminalTasks.length}`;
            heading.append(title, status);

            const list = document.createElement("ul");
            list.className = "task-list";

            for (const task of terminalTasks) {
                const item = document.createElement("li");
                item.className = "task-item";
                item.dataset.status = task.status;

                const copy = document.createElement("div");
                copy.className = "task-copy";

                const owner = document.createElement("div");
                owner.className = "task-owner";
                const ownerName = (
                    task.agent_name === "researcher"
                        ? "Researcher"
                        : String(task.agent_name || "unknown")
                );
                owner.textContent = `已分配给 ${ownerName} 任务`;

                const description = document.createElement("div");
                description.className = "task-description";
                description.textContent = activitySummaryForTask(task);
                copy.append(owner, description);

                if (task.status === "failed") {
                    const error = document.createElement("div");
                    error.className = "task-error";
                    error.textContent = "研究任务失败";
                    copy.append(error);
                }

                const itemStatus = document.createElement("span");
                itemStatus.className = "task-item-status";
                itemStatus.textContent = labels[task.status] || "未知";

                item.append(copy, itemStatus);
                list.append(item);
            }

            panel.append(heading, list);
            return panel;
        }

        function createArtifactPanel(artifacts, id) {
            if (!Array.isArray(artifacts) || artifacts.length === 0) {
                return null;
            }

            const panel = document.createElement("section");
            panel.className = "artifact-panel";
            panel.setAttribute("aria-label", "已保存文件");

            const heading = document.createElement("h3");
            heading.className = "artifact-heading";
            heading.textContent = "已保存文件";

            const list = document.createElement("ul");
            list.className = "artifact-list";

            for (const artifact of artifacts) {
                if (
                    !artifact ||
                    typeof artifact !== "object" ||
                    !/^[0-9a-f]{32}$/i.test(
                        artifact.artifact_id || "",
                    ) ||
                    typeof artifact.filename !== "string"
                ) {
                    continue;
                }

                const rawUrl =
                    typeof artifact.download_url === "string" &&
                    artifact.download_url
                        ? artifact.download_url
                        : `/artifacts/${encodeURIComponent(id)}/${artifact.artifact_id}`;

                let safeUrl;

                try {
                    safeUrl = new URL(
                        rawUrl,
                        window.location.origin,
                    );
                } catch {
                    continue;
                }

                if (safeUrl.origin !== window.location.origin) {
                    continue;
                }

                const link = document.createElement("a");
                link.className = "artifact-link";
                link.href = `${safeUrl.pathname}${safeUrl.search}`;
                link.download = artifact.filename;
                link.textContent = artifact.filename;
                link.setAttribute(
                    "aria-label",
                    `下载 ${artifact.filename}`,
                );

                const item = document.createElement("li");
                item.append(link);
                list.append(item);
            }

            if (list.children.length === 0) {
                return null;
            }

            panel.append(heading, list);
            return panel;
        }

        function renderThreadSnapshot(snapshot, id) {
            const messages = Array.isArray(snapshot.messages)
                ? snapshot.messages
                : [];

            if (messages.length === 0) {
                throw new Error("Thread snapshot has no messages");
            }

            conversation.replaceChildren();
            conversationEmpty = null;
            setActivityEvents(snapshot.activity);

            const validMessages = messages.filter(
                (message) =>
                    message &&
                    (message.role === "user" ||
                        message.role === "assistant") &&
                    typeof message.content === "string" &&
                    message.content,
            );

            let latestAssistant = null;

            for (const message of validMessages) {
                const turn = document.createElement("article");
                turn.className = "conversation-turn";

                const messageElement = document.createElement("div");
                messageElement.className =
                    `chat-message ${message.role}`;

                const label = document.createElement("div");
                label.className = "chat-label";
                label.textContent =
                    message.role === "user" ? "你" : "研究助手";

                const content = document.createElement("div");
                content.className = "chat-content";

                if (message.role === "assistant") {
                    content.className = "answer-content";
                    content.setAttribute("role", "region");
                    content.setAttribute("aria-label", "研究回答");
                    renderMarkdownInto(content, message.content);
                    latestAssistant = messageElement;
                } else {
                    content.textContent = message.content;
                    if (typeof globalThis.appendChatAttachmentPreviews === "function") {
                        globalThis.appendChatAttachmentPreviews(
                            messageElement,
                            message.attachment_ids,
                            id,
                        );
                    }
                }

                messageElement.append(label, content);
                turn.append(messageElement);
                conversation.append(turn);
            }

            // A native HITL pause or a tool failure can checkpoint after the
            // user message but before LangGraph writes a final assistant
            // message. That is still a valid, recoverable conversation. Do
            // not turn this state into a generic "restore failed" screen.
            if (!latestAssistant) {
                const turn = document.createElement("article");
                turn.className = "conversation-turn";

                const messageElement = document.createElement("div");
                messageElement.className = "chat-message assistant";

                const label = document.createElement("div");
                label.className = "chat-label";
                label.textContent = "研究助手";

                const content = document.createElement("div");
                content.className = "answer-content";
                content.setAttribute("role", "region");
                content.setAttribute("aria-label", "研究回答");
                content.textContent =
                    "本轮处理尚未生成最终回复，但会话状态已恢复。";

                messageElement.append(label, content);
                turn.append(messageElement);
                conversation.append(turn);
                latestAssistant = messageElement;
            }

            const plan = createRestoredPlanPanel(snapshot.todos);
            if (plan) {
                latestAssistant.append(plan);
            }

            const tasks = createRestoredTaskPanel(snapshot.tasks);
            if (tasks) {
                latestAssistant.append(tasks);
            }

            const artifacts = createArtifactPanel(
                snapshot.artifacts,
                id,
            );
            if (artifacts) {
                latestAssistant.append(artifacts);
            }

            answer = latestAssistant.querySelector(
                ".answer-content",
            );
            planPanel = latestAssistant.querySelector(
                ".plan-panel",
            );
            planStatus = latestAssistant.querySelector(
                ".plan-status",
            );
            planList = latestAssistant.querySelector(
                ".plan-list",
            );
            taskPanel = null;
            taskStatus = null;
            taskList = null;
            memoryNotice = null;
            activeTasks.clear();
            tracePanel = null;
            traceStatus = null;
            traceList = null;
        }

        function replaceRestoredArtifactPanel(artifacts, id) {
            const latestAssistant = conversation.querySelector(
                ".conversation-turn:last-child .chat-message.assistant",
            );

            if (!latestAssistant) {
                return;
            }

            latestAssistant
                .querySelector(".artifact-panel")
                ?.remove();

            const panel = createArtifactPanel(
                artifacts,
                id,
            );

            if (panel) {
                latestAssistant.append(panel);
            }
        }

        async function restoreArtifactList(
            id,
            requestId,
        ) {
            try {
                const response = await fetch(
                    `/threads/${encodeURIComponent(id)}/artifacts`,
                    {
                        headers: {
                            Accept: "application/json",
                        },
                    },
                );

                if (
                    requestId !== threadRestoreRequest ||
                    !response.ok
                ) {
                    return;
                }

                const payload = await response.json();

                if (!Array.isArray(payload.artifacts)) {
                    return;
                }

                replaceRestoredArtifactPanel(
                    payload.artifacts,
                    id,
                );
            } catch {
                // The thread snapshot remains usable if the optional list
                // request fails.
            }
        }

        function renderSettingsSummary() {
            if (!settingsSummary) {
                return;
            }

            settingsSummary.textContent = (
                currentThreadSummary ||
                "当前会话尚未进行上下文整理。"
            );
        }

        async function openThreadSettings() {
            if (!settingsDialog || !settingsSummary) {
                return;
            }

            if (typeof settingsDialog.showModal === "function") {
                settingsDialog.showModal();
            } else {
                settingsDialog.setAttribute("open", "");
            }

            settingsSummary.textContent = "正在读取当前会话摘要...";

            try {
                const response = await fetch(
                    `/threads/${encodeURIComponent(threadId)}`,
                    {
                        headers: {
                            Accept: "application/json",
                        },
                    },
                );

                if (!response.ok) {
                    currentThreadSummary = null;
                    renderSettingsSummary();
                    return;
                }

                const snapshot = await response.json();
                currentThreadSummary = (
                    typeof snapshot.summary === "string" &&
                    snapshot.summary.trim()
                        ? snapshot.summary
                        : null
                );
                renderSettingsSummary();
            } catch {
                currentThreadSummary = null;
                renderSettingsSummary();
            }
        }

        async function restoreThreadView(id) {
            const requestId = ++threadRestoreRequest;
            resetConversationView();
            currentThreadSummary = null;
            status.textContent = "恢复对话";

            try {
                const response = await fetch(
                    `/threads/${encodeURIComponent(id)}`,
                    {
                        headers: {
                            Accept: "application/json",
                        },
                    },
                );

                if (requestId !== threadRestoreRequest) {
                    return;
                }

                if (response.status === 404) {
                    status.textContent = "就绪";
                    document.dispatchEvent(
                        new CustomEvent("research:thread-restored", {
                            detail: { thread_id: id },
                        }),
                    );
                    return;
                }

                if (!response.ok) {
                    throw new Error("Thread snapshot request failed");
                }

                const snapshot = await response.json();
                currentThreadSummary = (
                    typeof snapshot.summary === "string" &&
                    snapshot.summary.trim()
                        ? snapshot.summary
                        : null
                );
                renderThreadSnapshot(snapshot, id);
                await restoreArtifactList(
                    id,
                    requestId,
                );

                const currentThread = threads.get(id);
                if (
                    currentThread &&
                    typeof snapshot.title === "string" &&
                    snapshot.title.trim()
                ) {
                    currentThread.title = snapshot.title
                        .trim()
                        .slice(0, 80);
                    persistThreadIndex();
                    renderThreadList();
                }

                status.textContent = "就绪";
                document.dispatchEvent(
                    new CustomEvent("research:thread-restored", {
                        detail: { thread_id: id },
                    }),
                );
            } catch {
                if (requestId !== threadRestoreRequest) {
                    return;
                }

                resetConversationView();
                conversationEmpty.textContent =
                    "对话恢复失败，请稍后重试。";
                status.textContent = "恢复失败";
            }
        }

        function switchThread(id) {
            if (controller || id === threadId) {
                return;
            }

            threadId = id;
            persistThreadIndex();
            questionInput.value = "";
            renderThreadList();
            notifyResearchThreadChanged();
            void restoreThreadView(id);
        }
