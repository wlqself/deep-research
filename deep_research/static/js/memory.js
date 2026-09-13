        function setMemoryStatus(message, state = "") {
            if (!memoryStatus) {
                return;
            }

            memoryStatus.textContent = message || "";
            memoryStatus.dataset.state = state;
        }

        function formatMemoryDate(value) {
            if (typeof value !== "string" || !value) {
                return "更新时间未知";
            }

            const date = new Date(value);

            if (Number.isNaN(date.getTime())) {
                return "更新时间未知";
            }

            return new Intl.DateTimeFormat("zh-CN", {
                dateStyle: "medium",
                timeStyle: "short",
            }).format(date);
        }

        function toDateTimeLocal(value) {
            if (typeof value !== "string" || !value) {
                return "";
            }

            const date = new Date(value);

            if (Number.isNaN(date.getTime())) {
                return "";
            }

            const pad = (part) => String(part).padStart(2, "0");

            return [
                date.getFullYear(),
                pad(date.getMonth() + 1),
                pad(date.getDate()),
            ].join("-") + "T" + [
                pad(date.getHours()),
                pad(date.getMinutes()),
            ].join(":");
        }

        function openMemoryDialog() {
            if (!memoryDialog) {
                return;
            }

            if (typeof memoryDialog.showModal === "function") {
                memoryDialog.showModal();
            } else {
                memoryDialog.setAttribute("open", "");
            }

            memoryPage = 1;
            void loadMemories();
        }

        function closeMemoryDialog() {
            if (!memoryDialog) {
                return;
            }

            if (typeof memoryDialog.close === "function") {
                memoryDialog.close();
            } else {
                memoryDialog.removeAttribute("open");
            }

            closeMemoryEditor();
        }

        function closeMemoryEditor() {
            memoryEditingEntry = null;

            if (memoryForm) {
                memoryForm.hidden = true;
                memoryForm.reset();
            }
        }

        function createMemoryAction(label, className, handler) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = `memory-item-action ${className || ""}`;
            button.textContent = label;
            button.addEventListener("click", handler);
            return button;
        }

        function renderMemories(memories) {
            if (!memoryList) {
                return;
            }

            memoryList.replaceChildren();

            if (!Array.isArray(memories) || memories.length === 0) {
                const empty = document.createElement("li");
                empty.className = "memory-empty";
                empty.textContent = "当前没有符合条件的记忆";
                memoryList.append(empty);
                return;
            }

            for (const entry of memories) {
                if (!entry || typeof entry.id !== "string") {
                    continue;
                }

                const item = document.createElement("li");
                item.className = "memory-item";

                const heading = document.createElement("div");
                heading.className = "memory-item-heading";

                const title = document.createElement("div");
                title.className = "memory-item-title";
                title.textContent = entry.title || "未命名记忆";

                const kind = document.createElement("span");
                kind.className = "memory-item-kind";
                kind.textContent = memoryKindLabels[entry.kind] || "记忆";
                heading.append(title, kind);

                const summary = document.createElement("div");
                summary.className = "memory-item-summary";
                summary.textContent = entry.summary || "";

                const content = document.createElement("div");
                content.className = "memory-item-content";
                content.textContent = entry.content || "";

                const meta = document.createElement("div");
                meta.className = "memory-item-meta";
                meta.textContent = [
                    `来源：${entry.source_type || "未知"}`,
                    formatMemoryDate(entry.updated_at),
                ].join(" · ");

                const keywords = document.createElement("div");
                keywords.className = "memory-item-keywords";
                keywords.textContent = Array.isArray(entry.keywords)
                    ? `关键词：${entry.keywords.join("、")}`
                    : "";

                const actions = document.createElement("div");
                actions.className = "memory-item-actions";
                actions.append(
                    createMemoryAction(
                        "编辑",
                        "",
                        () => openMemoryEditor(entry),
                    ),
                    createMemoryAction(
                        "删除",
                        "is-danger",
                        () => void deleteMemory(entry),
                    ),
                );

                item.append(
                    heading,
                    summary,
                    content,
                    meta,
                    keywords,
                    actions,
                );
                memoryList.append(item);
            }
        }

        async function loadMemories() {
            if (!memoryList) {
                return;
            }

            setMemoryStatus("正在加载…");
            memoryList.replaceChildren();

            const params = new URLSearchParams({
                page: String(memoryPage),
            });

            if (memoryKind) {
                params.set("kind", memoryKind);
            }

            const keyword = memorySearchInput?.value.trim() || "";

            if (keyword) {
                params.set("keyword", keyword);
            }

            try {
                const response = await fetch(`/memory?${params.toString()}`);

                if (!response.ok) {
                    throw new Error("memory list failed");
                }

                const payload = await response.json();

                if (!payload || !Array.isArray(payload.memories)) {
                    throw new Error("memory list response is invalid");
                }

                renderMemories(payload.memories);

                if (memoryPageIndicator) {
                    memoryPageIndicator.textContent = `第 ${memoryPage} 页`;
                }

                if (memoryPrevButton) {
                    memoryPrevButton.disabled = memoryPage <= 1;
                }

                if (memoryNextButton) {
                    memoryNextButton.disabled = (
                        payload.memories.length < payload.page_size
                    );
                }

                setMemoryStatus(
                    payload.memories.length
                        ? `显示 ${payload.memories.length} 条`
                        : "没有匹配结果",
                );
            } catch (error) {
                renderMemories([]);
                setMemoryStatus("记忆加载失败，请重试", "error");
            }
        }

        function openMemoryEditor(entry) {
            if (
                !memoryForm ||
                !entry ||
                typeof entry.id !== "string"
            ) {
                return;
            }

            memoryEditingEntry = entry;
            memoryForm.hidden = false;
            memoryEditTitle.value = entry.title || "";
            memoryEditSummary.value = entry.summary || "";
            memoryEditContent.value = entry.content || "";
            memoryEditKeywords.value = Array.isArray(entry.keywords)
                ? entry.keywords.join(", ")
                : "";
            memoryEditUrl.value = entry.url || "";
            memoryEditVerifiedAt.value = toDateTimeLocal(entry.verified_at);
            memoryEditIncorrect.value = entry.incorrect || "";
            memoryEditCorrect.value = entry.correct || "";
            memoryEditAppliesWhen.value = entry.applies_when || "";

            const isReference = entry.kind === "reference";
            const isFeedback = entry.kind === "feedback";

            memoryReferenceFields.hidden = !isReference;
            memoryReferenceDateField.hidden = !isReference;
            memoryFeedbackFields.hidden = !isFeedback;
            memoryFormTitle.textContent = (
                `编辑${memoryKindLabels[entry.kind] || ""}记忆`
            );
            memoryEditTitle.focus();
        }

        async function submitMemoryEdit(event) {
            event.preventDefault();

            if (!memoryEditingEntry || !memoryFormSubmit) {
                return;
            }

            memoryFormSubmit.disabled = true;
            setMemoryStatus("正在保存…");

            const payload = {
                title: memoryEditTitle.value.trim(),
                summary: memoryEditSummary.value.trim(),
                content: memoryEditContent.value.trim(),
                keywords: memoryEditKeywords.value
                    .split(",")
                    .map((item) => item.trim())
                    .filter(Boolean),
                url: memoryEditUrl.value.trim() || null,
                verified_at: memoryEditVerifiedAt.value
                    ? new Date(memoryEditVerifiedAt.value).toISOString()
                    : null,
                incorrect: memoryEditIncorrect.value.trim() || null,
                correct: memoryEditCorrect.value.trim() || null,
                applies_when: memoryEditAppliesWhen.value.trim() || null,
            };

            try {
                const response = await fetch(
                    `/memory/${encodeURIComponent(memoryEditingEntry.kind)}/${encodeURIComponent(memoryEditingEntry.id)}`,
                    {
                        method: "PUT",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify(payload),
                    },
                );

                if (!response.ok) {
                    throw new Error("memory update failed");
                }

                closeMemoryEditor();
                setMemoryStatus("已保存");
                await loadMemories();
            } catch (error) {
                setMemoryStatus("保存失败，请检查内容后重试", "error");
            } finally {
                memoryFormSubmit.disabled = false;
            }
        }

        async function deleteMemory(entry) {
            if (!entry || typeof entry.id !== "string") {
                return;
            }

            if (!window.confirm(`确定删除“${entry.title || "这条记忆"}”吗？`)) {
                return;
            }

            setMemoryStatus("正在删除…");

            try {
                const response = await fetch(
                    `/memory/${encodeURIComponent(entry.kind)}/${encodeURIComponent(entry.id)}`,
                    { method: "DELETE" },
                );

                if (!response.ok) {
                    throw new Error("memory delete failed");
                }

                closeMemoryEditor();
                setMemoryStatus("已删除");
                await loadMemories();
            } catch (error) {
                setMemoryStatus("删除失败，请重试", "error");
            }
        }

        async function clearMemories() {
            const label = memoryKindLabels[memoryKind] || "全部";

            if (!window.confirm(`确定清空${label}记忆吗？此操作不可撤销。`)) {
                return;
            }

            setMemoryStatus("正在清空…");

            const params = memoryKind
                ? `?kind=${encodeURIComponent(memoryKind)}`
                : "";

            try {
                const response = await fetch(
                    `/memory${params}`,
                    { method: "DELETE" },
                );

                if (!response.ok) {
                    throw new Error("memory clear failed");
                }

                closeMemoryEditor();
                memoryPage = 1;
                setMemoryStatus("已清空");
                await loadMemories();
            } catch (error) {
                setMemoryStatus("清空失败，请重试", "error");
            }
        }

        const memoryKindLabels = {
            "": "全部",
            user: "用户",
            reference: "参考",
            project: "项目",
            feedback: "反馈",
        };
        let memoryKind = "";
        let memoryPage = 1;
        let memoryEditingEntry = null;

        let controller = null;
        let cancellationRequested = false;
        let answer = null;
        let tracePanel = null;
        let traceStatus = null;
        let traceList = null;
        let planPanel = null;
        let planStatus = null;
        let planList = null;
        let taskPanel = null;
        let taskStatus = null;
        let taskList = null;
        let memoryNotice = null;
        let currentThreadSummary = null;
        const activeTasks = new Map();
