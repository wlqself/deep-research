        let activityEvents = [];

        const activityStatusLabels = {
            running: "进行中",
            completed: "已完成",
            failed: "失败",
            cancelled: "已取消",
        };

        function isValidActivityEvent(event) {
            return Boolean(
                event &&
                typeof event === "object" &&
                typeof event.id === "string" &&
                typeof event.timestamp === "string" &&
                typeof event.kind === "string" &&
                typeof event.actor === "string" &&
                typeof event.status === "string" &&
                typeof event.label === "string" &&
                typeof event.summary === "string" &&
                typeof event.detail === "string",
            );
        }

        function activityTimestamp() {
            return new Date().toISOString();
        }

        function activityId() {
            if (
                globalThis.crypto &&
                typeof globalThis.crypto.randomUUID === "function"
            ) {
                return `client-${globalThis.crypto.randomUUID()}`;
            }

            return `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        }

        function createClientActivity(
            kind,
            actor,
            status,
            label,
            detail = "",
            elapsedMs = null,
            summary = label,
        ) {
            return {
                id: activityId(),
                run_id: "client",
                timestamp: activityTimestamp(),
                kind,
                actor,
                status,
                label,
                summary,
                detail,
                elapsed_ms: elapsedMs,
            };
        }

        function normalizeActivityEvents(events) {
            if (!Array.isArray(events)) {
                return [];
            }

            const seen = new Set();
            return events
                .filter((event) => {
                    if (!isValidActivityEvent(event) || seen.has(event.id)) {
                        return false;
                    }

                    seen.add(event.id);
                    return true;
                })
                .sort((left, right) => (
                    Date.parse(left.timestamp) - Date.parse(right.timestamp)
                ));
        }

        function formatActivityTime(value) {
            const timestamp = Date.parse(value);
            if (!Number.isFinite(timestamp)) {
                return "时间未知";
            }

            return new Intl.DateTimeFormat("zh-CN", {
                month: "2-digit",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
            }).format(timestamp);
        }

        function formatActivityElapsed(value) {
            if (
                typeof value !== "number" ||
                !Number.isFinite(value) ||
                value < 0
            ) {
                return "";
            }

            if (value < 1000) {
                return `${Math.round(value)} ms`;
            }

            return `${(value / 1000).toFixed(1)} s`;
        }

        function renderActivityLog(events = activityEvents) {
            activityEvents = normalizeActivityEvents(events);

            if (!activityList) {
                return;
            }

            activityList.replaceChildren();

            for (const event of activityEvents) {
                const item = document.createElement("li");
                item.className = `activity-item activity-${event.status}`;

                const marker = document.createElement("span");
                marker.className = "activity-marker";
                marker.setAttribute("aria-hidden", "true");

                const body = document.createElement("div");
                body.className = "activity-item-body";

                const heading = document.createElement("div");
                heading.className = "activity-item-heading";

                const label = document.createElement("strong");
                label.className = "activity-item-label";
                label.textContent = event.label;

                const state = document.createElement("span");
                state.className = "activity-item-state";
                state.textContent = (
                    activityStatusLabels[event.status] || event.status
                );
                heading.append(label, state);

                body.append(heading);

                if (event.detail) {
                    const detail = document.createElement("p");
                    detail.className = "activity-item-detail";
                    detail.textContent = event.detail;
                    body.append(detail);
                }

                const meta = document.createElement("div");
                meta.className = "activity-item-meta";
                const elapsed = formatActivityElapsed(event.elapsed_ms);
                meta.textContent = elapsed
                    ? `${formatActivityTime(event.timestamp)} · ${elapsed}`
                    : formatActivityTime(event.timestamp);
                body.append(meta);

                item.append(marker, body);
                activityList.append(item);
            }

            if (activityStatus) {
                activityStatus.textContent = activityEvents.length
                    ? `共 ${activityEvents.length} 条记录`
                    : "当前会话还没有研究日志";
            }

            if (activitySummary) {
                activitySummary.textContent = activityEvents.length
                    ? "问题、计划和执行步骤会按时间记录在这里"
                    : "记录当前会话的问题和执行过程";
            }
        }

        function resetActivityLog() {
            activityEvents = [];
            renderActivityLog();
        }

        function setActivityEvents(events) {
            renderActivityLog(events);
        }

        function appendActivityEvent(event) {
            if (!isValidActivityEvent(event)) {
                return;
            }

            renderActivityLog([...activityEvents, event]);
        }

        function activitySummaryForTask(task) {
            const agentName = String(task?.agent_name || "researcher");
            const activity = [...activityEvents]
                .reverse()
                .find(
                    (event) => (
                        event.kind === "agent" &&
                        event.actor === agentName &&
                        event.status === "running" &&
                        event.summary
                    ),
                );

            if (activity) {
                return activity.summary;
            }

            const description = String(task?.description || "");
            const compacted = description.replace(/\s+/g, " ").trim();

            if (compacted.length <= 140) {
                return compacted;
            }

            return `${compacted.slice(0, 139)}…`;
        }

        function recordQuestionActivity(question) {
            appendActivityEvent(
                createClientActivity(
                    "question",
                    "user",
                    "completed",
                    "用户提出问题",
                    question,
                ),
            );
            appendActivityEvent(
                createClientActivity(
                    "research",
                    "main",
                    "running",
                    "开始处理问题",
                ),
            );
        }

        function recordActivityFromStreamEvent(event) {
            if (!event || typeof event.type !== "string") {
                return;
            }

            const type = event.type;
            const agentName = String(event.agent_name || "researcher");
            const name = String(event.name || "unknown_tool");
            let activity = null;

            if (type === "plan_update") {
                const count = Array.isArray(event.todos)
                    ? event.todos.length
                    : 0;
                activity = createClientActivity(
                    "plan",
                    "main",
                    "completed",
                    "研究计划已更新",
                    `当前包含 ${count} 个步骤`,
                );
            } else if (type === "subagent_start") {
                activity = createClientActivity(
                    "agent",
                    agentName,
                    "running",
                    `${agentName} 开始执行任务`,
                    String(event.description || ""),
                );
            } else if (type === "subagent_end") {
                const state = String(event.status || "completed");
                const sourceCount = Array.isArray(event.source_ids)
                    ? event.source_ids.length
                    : 0;
                activity = createClientActivity(
                    "agent",
                    agentName,
                    state,
                    state === "completed"
                        ? `${agentName} 已完成任务`
                        : `${agentName} 任务${activityStatusLabels[state] || state}`,
                    state === "completed"
                        ? `关联 ${sourceCount} 个来源`
                        : String(event.error || "任务未完成"),
                );
            } else if (type === "tool_start") {
                if (name === "prepare_article_for_publication") {
                    activity = createClientActivity(
                        "publishing",
                        "main",
                        "running",
                        "开始整理文章草稿",
                        "正在根据研究 Artifact 准备 Article 草稿",
                    );
                } else {
                    activity = createClientActivity(
                        "tool",
                        "agent",
                        "running",
                        `开始使用 ${name}`,
                    );
                }
            } else if (type === "tool_end") {
                const state = String(event.status || "completed");
                if (name === "prepare_article_for_publication") {
                    const failed = state === "failed";
                    activity = createClientActivity(
                        "publishing",
                        "main",
                        state,
                        failed
                            ? "文章草稿整理失败"
                            : "文章草稿整理完成",
                        failed
                            ? "文章草稿未能准备完成"
                            : "草稿已准备完成，等待用户人工审批",
                        typeof event.elapsed_ms === "number"
                            ? event.elapsed_ms
                            : null,
                    );
                } else {
                    activity = createClientActivity(
                        "tool",
                        "agent",
                        state,
                        `${name} 执行结束`,
                        "",
                        typeof event.elapsed_ms === "number"
                            ? event.elapsed_ms
                            : null,
                    );
                }
            } else if (type === "memory_compacted") {
                activity = createClientActivity(
                    "memory",
                    "system",
                    "completed",
                    "上下文已整理",
                );
            } else if (type === "memory_recall_started" || type === "memory_recall_completed") {
                activity = createClientActivity(
                    "memory",
                    "system",
                    type === "memory_recall_started" ? "running" : "completed",
                    type === "memory_recall_started" ? "正在检索长期记忆" : "长期记忆检索完成",
                );
            } else if (type === "model_started" || type === "model_reasoning" || type === "model_retrying" || type === "model_completed" || type === "model_failed" || type === "heartbeat") {
                const failed = type === "model_failed";
                const completed = type === "model_completed";
                activity = createClientActivity(
                    "model",
                    String(event.agent_name || "main"),
                    failed ? "failed" : completed ? "completed" : "running",
                    type === "model_retrying"
                        ? `模型响应异常，正在进行第 ${event.attempt || "?"} 次重试`
                        : type === "model_reasoning"
                            ? String(event.message || "正在分析请求")
                            : type === "heartbeat"
                                ? "仍在处理中"
                                : failed
                                    ? "模型调用失败"
                                    : completed
                                        ? "模型阶段完成"
                                        : "正在分析请求",
                );
            } else if (type === "artifact_saved") {
                activity = createClientActivity(
                    "artifact",
                    "agent",
                    "completed",
                    "研究报告已保存",
                    String(event.filename || ""),
                );
            } else if (type === "done") {
                const goalStatus = event.goal_status || "completed";
                activity = createClientActivity(
                    "research",
                    "main",
                    goalStatus,
                    goalStatus === "waiting_for_user"
                        ? "等待用户确认"
                        : goalStatus === "completed_with_failure"
                            ? "处理结束，但目标未完成"
                            : "处理完成",
                    Array.isArray(event.error_codes)
                        ? event.error_codes.join(", ")
                        : "",
                );
            } else if (type === "cancelled") {
                activity = createClientActivity(
                    "research",
                    "main",
                    "cancelled",
                    "研究已取消",
                );
            } else if (type === "cancel_requested") {
                activity = createClientActivity(
                    "research",
                    "main",
                    "running",
                    "正在取消研究",
                );
            } else if (type === "error") {
                activity = createClientActivity(
                    "research",
                    "main",
                    "failed",
                    "研究失败",
                    String(event.error || "研究服务发生错误"),
                );
            }

            if (activity) {
                appendActivityEvent(activity);
            }
        }

        function openActivityDialog() {
            renderActivityLog();

            if (typeof activityDialog?.showModal === "function") {
                activityDialog.showModal();
            } else {
                activityDialog?.setAttribute("open", "");
            }
        }

        function closeActivityDialog() {
            if (typeof activityDialog?.close === "function") {
                activityDialog.close();
            } else {
                activityDialog?.removeAttribute("open");
            }
        }

        activityOpenButton?.addEventListener("click", openActivityDialog);
        activityCloseButton?.addEventListener("click", closeActivityDialog);

        renderActivityLog();
