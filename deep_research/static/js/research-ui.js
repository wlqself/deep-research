        function startNewThread() {
            if (controller) {
                return;
            }

            threadRestoreRequest += 1;
            threadId = createThreadId();
            currentThreadSummary = null;
            threads.set(threadId, {
                title: "新对话",
                updated_at: new Date().toISOString(),
            });
            persistThreadIndex();
            resetConversationView();
            questionInput.value = "";
            status.textContent = "就绪";
            renderThreadList();
            notifyResearchThreadChanged();
            questionInput.focus();
        }

        function appendChatAttachmentPreviews(messageElement, attachmentIds, currentThreadId) {
            const validAttachmentIds = Array.isArray(attachmentIds)
                ? [...new Set(attachmentIds.filter(
                    (attachmentId) => (
                        typeof attachmentId === "string" && attachmentId.trim()
                    ),
                ))]
                : [];
            if (validAttachmentIds.length === 0) {
                return;
            }

            const gallery = document.createElement("div");
            gallery.className = "chat-attachments";
            for (const attachmentId of validAttachmentIds) {
                const preview = document.createElement("img");
                preview.className = "chat-attachment-preview";
                preview.alt = "本条消息附加的图片";
                preview.src = `/publishing/attachments/images/${encodeURIComponent(attachmentId)}/content?thread_id=${encodeURIComponent(currentThreadId || "")}`;
                preview.addEventListener("error", () => {
                    preview.hidden = true;
                });
                gallery.append(preview);
            }
            messageElement.append(gallery);
        }

        globalThis.appendChatAttachmentPreviews = appendChatAttachmentPreviews;

        function appendConversationTurn(question, attachmentIds = []) {
            if (conversationEmpty) {
                conversationEmpty.hidden = true;
            }

            const turn = document.createElement("article");
            turn.className = "conversation-turn";

            const userMessage = document.createElement("div");
            userMessage.className = "chat-message user";

            const userLabel = document.createElement("div");
            userLabel.className = "chat-label";
            userLabel.textContent = "你";

            const userContent = document.createElement("div");
            userContent.className = "chat-content";
            userContent.textContent = question;
            userMessage.append(userLabel, userContent);

            appendChatAttachmentPreviews(userMessage, attachmentIds, threadId);

            const assistantMessage = document.createElement("div");
            assistantMessage.className = "chat-message assistant";

            const assistantLabel = document.createElement("div");
            assistantLabel.className = "chat-label";
            assistantLabel.textContent = "研究助手";

            planPanel = document.createElement("section");
            planPanel.className = "plan-panel";
            planPanel.hidden = true;
            planPanel.setAttribute("aria-label", "研究计划");

            const planHeading = document.createElement("div");
            planHeading.className = "plan-heading";

            const planTitle = document.createElement("h3");
            planTitle.textContent = "研究计划";

            planStatus = document.createElement("span");
            planStatus.className = "plan-status";
            planStatus.textContent = "等待中";

            planHeading.append(planTitle, planStatus);

            planList = document.createElement("ol");
            planList.className = "plan-list";
            planList.setAttribute("aria-live", "polite");
            planPanel.append(planHeading, planList);

            taskPanel = document.createElement("section");
            taskPanel.className = "task-panel";
            taskPanel.hidden = true;
            taskPanel.setAttribute("aria-label", "研究任务");

            const taskHeading = document.createElement("div");
            taskHeading.className = "task-heading";

            const taskTitle = document.createElement("h3");
            taskTitle.textContent = "研究任务";

            taskStatus = document.createElement("span");
            taskStatus.className = "task-status";
            taskStatus.textContent = "等待中";

            taskHeading.append(taskTitle, taskStatus);

            taskList = document.createElement("ul");
            taskList.className = "task-list";
            taskList.setAttribute("aria-live", "polite");

            taskPanel.append(taskHeading, taskList);

            memoryNotice = document.createElement("div");
            memoryNotice.className = "memory-notice";
            memoryNotice.hidden = true;
            memoryNotice.setAttribute("role", "status");
            memoryNotice.setAttribute("aria-live", "polite");

            tracePanel = document.createElement("details");
            tracePanel.className = "trace-panel";
            tracePanel.hidden = true;

            const traceSummary = document.createElement("summary");
            const traceTitle = document.createElement("span");
            traceTitle.textContent = "研究过程";

            traceStatus = document.createElement("span");
            traceStatus.className = "trace-status";
            traceStatus.textContent = "等待中";

            traceSummary.append(traceTitle, traceStatus);

            traceList = document.createElement("div");
            traceList.className = "trace-list";
            traceList.setAttribute("role", "log");
            traceList.setAttribute("aria-live", "polite");
            traceList.setAttribute(
                "aria-label",
                "研究工具调用记录",
            );

            tracePanel.append(traceSummary, traceList);

            answer = document.createElement("div");
            answer.className = "answer-content is-streaming";
            answer.setAttribute("role", "region");
            answer.setAttribute("aria-label", "研究回答");
            answer.textContent = "研究中...";

            assistantMessage.append(
                assistantLabel,
                planPanel,
                taskPanel,
                memoryNotice,
                tracePanel,
                answer,
            );
            turn.append(userMessage, assistantMessage);
            conversation.append(turn);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function persistSuccessfulThread(question) {
            const currentThread = threads.get(threadId);

            if (!currentThread) {
                return;
            }

            if (currentThread.title === "新对话") {
                currentThread.title = question
                    .trim()
                    .slice(0, 80);
            }

            currentThread.updated_at =
                new Date().toISOString();

            persistThreadIndex();
            renderThreadList();
        }

        function renderMarkdown(markdown) {
            renderMarkdownInto(answer, markdown);
        }

        function renderArtifactSaved(event) {
            if (!event.filename || !event.download_url) {
                return;
            }

            const assistantMessage = conversation.querySelector(
                ".conversation-turn:last-child .chat-message.assistant",
            );

            if (!assistantMessage) {
                return;
            }

            let artifactPanel = assistantMessage.querySelector(
                ".artifact-panel",
            );

            if (!artifactPanel) {
                artifactPanel = document.createElement("section");
                artifactPanel.className = "artifact-panel";
                artifactPanel.setAttribute(
                    "aria-label",
                    "已保存文件",
                );

                const heading = document.createElement("h3");
                heading.className = "artifact-heading";
                heading.textContent = "已保存文件";

                const list = document.createElement("ul");
                list.className = "artifact-list";

                artifactPanel.append(heading, list);
                assistantMessage.append(artifactPanel);
            }

            const safeUrl = new URL(
                event.download_url,
                window.location.origin,
            );

            if (safeUrl.origin !== window.location.origin) {
                return;
            }

            const link = document.createElement("a");
            link.className = "artifact-link";
            link.href = `${safeUrl.pathname}${safeUrl.search}`;
            link.download = event.filename;
            link.textContent = event.filename;
            link.setAttribute(
                "aria-label",
                `下载 ${event.filename}`,
            );

            const item = document.createElement("li");
            item.append(link);
            artifactPanel.querySelector(
                ".artifact-list",
            ).append(item);
        }

        function resetTrace() {
            if (!traceList || !tracePanel || !traceStatus) {
                return;
            }

            traceList.replaceChildren();
            tracePanel.hidden = false;
            tracePanel.open = true;
            traceStatus.textContent = "进行中";
        }

        function addTraceItem(text, state = "") {
            if (!traceList) {
                return;
            }

            const item = document.createElement("div");
            item.className = `trace-item ${state}`;
            item.textContent = text;
            traceList.append(item);
            traceList.scrollTop = traceList.scrollHeight;
        }

        function resetPlan() {
            if (!planList || !planPanel || !planStatus) {
                return;
            }

            planList.replaceChildren();
            planPanel.hidden = true;
            planStatus.textContent = "等待中";
        }

        function resetTasks() {
            activeTasks.clear();

            if (!taskPanel || !taskStatus || !taskList) {
                return;
            }

            taskList.replaceChildren();
            taskPanel.hidden = true;
            taskStatus.textContent = "等待中";
        }

        function resetMemoryNotice() {
            if (!memoryNotice) {
                return;
            }

            memoryNotice.hidden = true;
            memoryNotice.textContent = "";
        }

        function showMemoryCompactedNotice() {
            if (!memoryNotice) {
                return;
            }

            memoryNotice.hidden = false;
            memoryNotice.textContent = "当前会话已进行一次上下文整理。";
        }

        function renderResearchTasks() {
            if (!taskPanel || !taskStatus || !taskList) {
                return;
            }

            const labels = {
                running: "进行中",
                completed: "已完成",
                failed: "失败",
                cancelled: "已取消",
            };
            const tasks = Array.from(activeTasks.values());
            const completedCount = tasks.filter(
                (task) => task.status === "completed",
            ).length;
            const failedCount = tasks.filter(
                (task) => task.status === "failed",
            ).length;
            const cancelledCount = tasks.filter(
                (task) => task.status === "cancelled",
            ).length;

            taskList.replaceChildren();
            taskPanel.hidden = tasks.length === 0;

            for (const task of tasks) {
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
                        : task.agent_name
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

                const status = document.createElement("span");
                status.className = "task-item-status";
                status.textContent = labels[task.status] || "未知";

                item.append(copy, status);
                taskList.append(item);
            }

            if (tasks.length === 0) {
                taskStatus.textContent = "等待中";
            } else if (tasks.some((task) => task.status === "running")) {
                taskStatus.textContent = `进行中 ${tasks.length} 项`;
            } else if (failedCount > 0) {
                taskStatus.textContent = `失败 ${failedCount}/${tasks.length}`;
            } else if (cancelledCount > 0) {
                taskStatus.textContent = `已取消 ${cancelledCount}/${tasks.length}`;
            } else {
                taskStatus.textContent = `已完成 ${completedCount}/${tasks.length}`;
            }
        }

        function markRunningTasksCancelled() {
            for (const [taskId, task] of activeTasks.entries()) {
                if (task.status !== "running") {
                    continue;
                }

                activeTasks.set(taskId, {
                    ...task,
                    status: "cancelled",
                    error: "",
                });
            }

            renderResearchTasks();
        }

        function updateResearchTask(event) {
            if (
                !event ||
                typeof event.task_id !== "string" ||
                !event.task_id
            ) {
                return;
            }

            const previous = activeTasks.get(event.task_id) || {};
            const terminalStatuses = new Set([
                "completed",
                "failed",
                "cancelled",
            ]);
            const incomingStatus = (
                typeof event.status === "string" &&
                ["running", "completed", "failed", "cancelled"]
                    .includes(event.status)
                    ? event.status
                    : previous.status || "running"
            );
            const status = (
                terminalStatuses.has(previous.status)
                    ? previous.status
                    : cancellationRequested && incomingStatus === "running"
                        ? "cancelled"
                    : incomingStatus
            );

            activeTasks.set(event.task_id, {
                agent_name: (
                    typeof event.agent_name === "string"
                        ? event.agent_name
                        : previous.agent_name || "researcher"
                ),
                description: (
                    typeof event.description === "string"
                        ? event.description
                        : previous.description || ""
                ),
                status,
                finding_ids: (
                    Array.isArray(event.finding_ids)
                        ? event.finding_ids
                        : previous.finding_ids || []
                ),
                source_ids: (
                    Array.isArray(event.source_ids)
                        ? event.source_ids
                        : previous.source_ids || []
                ),
                error: (
                    status === "cancelled"
                        ? ""
                        : typeof event.error === "string"
                            ? event.error
                            : previous.error || ""
                ),
            });

            renderResearchTasks();
        }

        function renderPlan(todos) {
            if (!Array.isArray(todos)) {
                return;
            }

            planList.replaceChildren();
            planPanel.hidden = false;

            const labels = {
                pending: "待处理",
                in_progress: "进行中",
                completed: "已完成",
            };

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
                planList.append(item);

                if (todo.status === "completed") {
                    completedCount += 1;
                }
            }

            if (todos.length > 0 && completedCount === todos.length) {
                planStatus.textContent =
                    `已完成 ${completedCount}/${todos.length}`;
            } else {
                planStatus.textContent =
                    `进行中 ${completedCount}/${todos.length}`;
            }
        }
