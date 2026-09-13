        function createThreadId() {
            if (
                globalThis.crypto &&
                typeof globalThis.crypto.randomUUID === "function"
            ) {
                return globalThis.crypto.randomUUID();
            }

            return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(
                /[xy]/g,
                (character) => {
                    const random = Math.floor(Math.random() * 16);
                    const value =
                        character === "x"
                            ? random
                            : (random & 0x3) | 0x8;

                    return value.toString(16);
                },
            );
        }

        const THREAD_INDEX_KEY = "deep-research.threads.v1";
        const MAX_RECENT_THREADS = 20;
        const THREAD_ID_PATTERN =
            /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
        const CONVERSATION_PANEL_KEY = "deep-research.conversation-panel-open.v1";
        const conversationPanel = document.querySelector(
            "#conversation-panel",
        );
        const conversationToggle = document.querySelector(
            "#conversation-toggle",
        );

        function setConversationPanelExpanded(expanded) {
            const isExpanded = Boolean(expanded);
            if (conversationPanel) {
                conversationPanel.hidden = !isExpanded;
            }
            conversationToggle?.setAttribute(
                "aria-expanded",
                isExpanded ? "true" : "false",
            );
            try {
                localStorage.setItem(
                    CONVERSATION_PANEL_KEY,
                    isExpanded ? "true" : "false",
                );
            } catch {
                // Disabled storage must not break the conversation menu.
            }
        }

        function conversationPanelInitiallyExpanded() {
            try {
                return localStorage.getItem(CONVERSATION_PANEL_KEY) !== "false";
            } catch {
                return true;
            }
        }

        function isValidThreadId(value) {
            return (
                typeof value === "string" &&
                THREAD_ID_PATTERN.test(value)
            );
        }

        function parseUpdatedAt(value) {
            if (typeof value !== "string") {
                return 0;
            }

            const timestamp = Date.parse(value);
            return Number.isFinite(timestamp) ? timestamp : 0;
        }

        function loadThreadIndex() {
            try {
                const raw = localStorage.getItem(
                    THREAD_INDEX_KEY,
                );

                if (!raw) {
                    return null;
                }

                const parsed = JSON.parse(raw);

                if (
                    !parsed ||
                    !Array.isArray(parsed.threads)
                ) {
                    return null;
                }

                const entries = [];
                const seen = new Set();

                for (const item of parsed.threads) {
                    if (
                        !item ||
                        !isValidThreadId(item.thread_id) ||
                        seen.has(item.thread_id)
                    ) {
                        continue;
                    }

                    seen.add(item.thread_id);

                    entries.push({
                        thread_id: item.thread_id,
                        title:
                            typeof item.title === "string" &&
                            item.title.trim()
                                ? item.title.trim().slice(0, 80)
                                : "新对话",
                        updated_at:
                            typeof item.updated_at === "string"
                                ? item.updated_at
                                : new Date(0).toISOString(),
                    });
                }

                entries.sort(
                    (left, right) =>
                        parseUpdatedAt(right.updated_at) -
                        parseUpdatedAt(left.updated_at),
                );

                const activeThreadId =
                    isValidThreadId(parsed.active_thread_id) &&
                    entries.some(
                        (item) =>
                            item.thread_id ===
                            parsed.active_thread_id,
                    )
                        ? parsed.active_thread_id
                        : entries[0]?.thread_id || null;

                return {
                    active_thread_id: activeThreadId,
                    threads: entries.slice(
                        0,
                        MAX_RECENT_THREADS,
                    ),
                };
            } catch {
                return null;
            }
        }

        function persistThreadIndex() {
            const entries = [...threads.entries()]
                .filter(([id]) => isValidThreadId(id))
                .map(([id, thread]) => ({
                    thread_id: id,
                    title:
                        typeof thread.title === "string" &&
                        thread.title.trim()
                            ? thread.title.trim().slice(0, 80)
                            : "新对话",
                    updated_at:
                        typeof thread.updated_at === "string"
                            ? thread.updated_at
                            : new Date().toISOString(),
                }))
                .sort(
                    (left, right) =>
                        parseUpdatedAt(right.updated_at) -
                        parseUpdatedAt(left.updated_at),
                )
                .slice(0, MAX_RECENT_THREADS);

            localStorage.setItem(
                THREAD_INDEX_KEY,
                JSON.stringify({
                    active_thread_id: threadId,
                    threads: entries,
                }),
            );
        }

        const savedThreadIndex = loadThreadIndex();
        const threads = new Map(
            (savedThreadIndex?.threads || []).map(
                (item) => [
                    item.thread_id,
                    {
                        title: item.title,
                        updated_at: item.updated_at,
                    },
                ],
            ),
        );

        let threadId =
            savedThreadIndex?.active_thread_id ||
            createThreadId();

        if (!threads.has(threadId)) {
            threads.set(threadId, {
                title: "新对话",
                updated_at: new Date().toISOString(),
            });
        }

        persistThreadIndex();

        globalThis.getCurrentResearchThreadId = () => threadId;

        function notifyResearchThreadChanged() {
            document.dispatchEvent(
                new CustomEvent("research:thread-changed", {
                    detail: { thread_id: threadId },
                }),
            );
        }

        let threadRestoreRequest = 0;
        let contextThreadId = null;
        let renameThreadId = null;

        conversationToggle?.addEventListener("click", () => {
            setConversationPanelExpanded(Boolean(conversationPanel?.hidden));
        });
        setConversationPanelExpanded(conversationPanelInitiallyExpanded());

        function setBusy(isBusy) {
            submitButton.disabled = isBusy;
            submitButton.textContent = isBusy ? "…" : "↑";
            submitButton.classList.toggle("is-busy", isBusy);
            submitButton.setAttribute("aria-label", isBusy ? "研究中" : "发送");
            cancelButton.hidden = !isBusy;
            questionInput.disabled = isBusy;
        }

        function renderThreadList() {
            if (!threadList) {
                return;
            }
            threadList.replaceChildren();

            for (const [id, thread] of threads) {
                const item = document.createElement("button");
                item.className = "thread-item";
                item.type = "button";
                item.textContent = thread.title;

                if (id === threadId) {
                    item.classList.add("is-active");
                }

                item.addEventListener("click", () => {
                    switchThread(id);
                });

                item.addEventListener("contextmenu", (event) => {
                    event.preventDefault();
                    openThreadContextMenu(
                        id,
                        event.clientX,
                        event.clientY,
                    );
                });

                item.addEventListener("keydown", (event) => {
                    const isContextMenuKey = (
                        event.key === "ContextMenu" ||
                        (event.key === "F10" && event.shiftKey)
                    );

                    if (!isContextMenuKey) {
                        return;
                    }

                    event.preventDefault();
                    const rect = item.getBoundingClientRect();
                    openThreadContextMenu(
                        id,
                        rect.left,
                        rect.bottom,
                    );
                });

                threadList.append(item);
            }
        }

        function closeThreadContextMenu() {
            if (!threadContextMenu) {
                return;
            }

            threadContextMenu.hidden = true;
            contextThreadId = null;
        }

        function openThreadContextMenu(id, x, y) {
            if (
                controller ||
                !threadContextMenu ||
                !threads.has(id)
            ) {
                return;
            }

            contextThreadId = id;
            threadContextMenu.hidden = false;
            threadContextMenu.style.left = `${Math.max(8, x)}px`;
            threadContextMenu.style.top = `${Math.max(8, y)}px`;

            const rect = threadContextMenu.getBoundingClientRect();
            const left = Math.min(
                Math.max(8, x),
                Math.max(8, window.innerWidth - rect.width - 8),
            );
            const top = Math.min(
                Math.max(8, y),
                Math.max(8, window.innerHeight - rect.height - 8),
            );

            threadContextMenu.style.left = `${left}px`;
            threadContextMenu.style.top = `${top}px`;
            threadContextMenu
                .querySelector("[data-thread-action='rename']")
                ?.focus();
        }

        function openThreadRenameDialog(id) {
            const thread = threads.get(id);

            if (
                !threadRenameDialog ||
                !threadRenameInput ||
                !thread
            ) {
                return;
            }

            renameThreadId = id;
            threadRenameInput.value = thread.title || "";
            threadRenameError.textContent = "";
            closeThreadContextMenu();

            if (typeof threadRenameDialog.showModal === "function") {
                threadRenameDialog.showModal();
            } else {
                threadRenameDialog.setAttribute("open", "");
            }

            threadRenameInput.focus();
            threadRenameInput.select();
        }

        function closeThreadRenameDialog() {
            if (!threadRenameDialog) {
                return;
            }

            if (typeof threadRenameDialog.close === "function") {
                threadRenameDialog.close();
            } else {
                threadRenameDialog.removeAttribute("open");
            }

            renameThreadId = null;
            if (threadRenameError) {
                threadRenameError.textContent = "";
            }
        }

        async function submitThreadRename(event) {
            event.preventDefault();

            if (
                !renameThreadId ||
                !threadRenameInput ||
                !threadRenameSubmit
            ) {
                return;
            }

            const title = threadRenameInput.value.trim();

            if (!title) {
                threadRenameError.textContent = "请输入会话名称";
                threadRenameInput.focus();
                return;
            }

            if (title.length > 80) {
                threadRenameError.textContent = "会话名称不能超过 80 个字符";
                threadRenameInput.focus();
                return;
            }

            const targetId = renameThreadId;
            threadRenameSubmit.disabled = true;
            threadRenameError.textContent = "";

            try {
                const response = await fetch(
                    `/threads/${encodeURIComponent(targetId)}`,
                    {
                        method: "PATCH",
                        headers: {
                            "Content-Type": "application/json",
                            Accept: "application/json",
                        },
                        body: JSON.stringify({ title }),
                    },
                );

                if (!response.ok) {
                    throw new Error("thread rename failed");
                }

                const payload = await response.json();
                const savedTitle = (
                    typeof payload.title === "string" &&
                    payload.title.trim()
                        ? payload.title.trim().slice(0, 80)
                        : title
                );
                const thread = threads.get(targetId);

                if (thread) {
                    thread.title = savedTitle;
                    thread.updated_at = new Date().toISOString();
                }

                persistThreadIndex();
                renderThreadList();
                closeThreadRenameDialog();
                status.textContent = "会话名称已更新";
            } catch {
                threadRenameError.textContent = "重命名失败，请稍后重试";
            } finally {
                threadRenameSubmit.disabled = false;
            }
        }

        async function deleteThread(id) {
            const thread = threads.get(id);

            if (!thread || controller) {
                return;
            }

            if (!window.confirm(`确定删除“${thread.title}”吗？`)) {
                return;
            }

            closeThreadContextMenu();

            try {
                const response = await fetch(
                    `/threads/${encodeURIComponent(id)}`,
                    { method: "DELETE" },
                );

                if (!response.ok && response.status !== 404) {
                    throw new Error("thread delete failed");
                }

                const deletingCurrent = id === threadId;
                threads.delete(id);

                if (deletingCurrent) {
                    threadRestoreRequest += 1;
                    const nextThreadId = threads.keys().next().value;

                    if (typeof nextThreadId === "string") {
                        threadId = nextThreadId;
                        currentThreadSummary = null;
                        questionInput.value = "";
                        resetConversationView();
                        persistThreadIndex();
                        renderThreadList();
                        notifyResearchThreadChanged();
                        void restoreThreadView(nextThreadId);
                    } else {
                        startNewThread();
                    }
                } else {
                    persistThreadIndex();
                    renderThreadList();
                }

                status.textContent = "会话已删除";
            } catch {
                status.textContent = "删除失败，请稍后重试";
            }
        }

        function resetConversationView() {
            conversation.replaceChildren();

            conversationEmpty = document.createElement("div");
            conversationEmpty.id = "conversation-empty";
            conversationEmpty.className = "conversation-empty";
            conversationEmpty.textContent =
                "输入问题后，对话会显示在这里。";
            conversation.append(conversationEmpty);

            answer = null;
            resetActivityLog();
            tracePanel = null;
            traceStatus = null;
            traceList = null;
            planPanel = null;
            planStatus = null;
            planList = null;
            taskPanel = null;
            taskStatus = null;
            taskList = null;
            memoryNotice = null;
            activeTasks.clear();
        }
