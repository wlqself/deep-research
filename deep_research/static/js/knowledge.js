        const knowledgeStatusLabels = {
            pending: "待处理",
            processing: "处理中",
            indexed: "已索引",
            failed: "失败",
            deleting: "删除中",
            deleted: "已删除",
        };

        const knowledgeDocumentDetailDialog = document.querySelector(
            "#knowledge-document-detail-dialog",
        );
        const knowledgeDocumentDetail = document.querySelector(
            "#knowledge-document-detail",
        );
        const knowledgeDocumentDetailCloseButton = document.querySelector(
            "#knowledge-document-detail-close",
        );
        let knowledgeDocuments = [];
        let focusedKnowledgeDocumentId = null;
        let knowledgeContextMenu = null;

        function formatKnowledgeBytes(value) {
            if (!Number.isFinite(value) || value < 0) {
                return "未知大小";
            }

            if (value < 1024) {
                return `${value} B`;
            }

            if (value < 1024 * 1024) {
                return `${(value / 1024).toFixed(1)} KB`;
            }

            return `${(value / (1024 * 1024)).toFixed(1)} MB`;
        }

        function setKnowledgeStatus(message) {
            if (knowledgeStatus) {
                knowledgeStatus.textContent = message || "";
            }
        }

        function findKnowledgeDocument(documentId) {
            return knowledgeDocuments.find(
                (item) => item && item.document_id === documentId,
            ) || null;
        }

        function closeKnowledgeContextMenu() {
            if (!knowledgeContextMenu) {
                return;
            }
            knowledgeContextMenu.hidden = true;
            knowledgeContextMenu.replaceChildren();
        }

        function closeKnowledgeDocumentDetail() {
            focusedKnowledgeDocumentId = null;
            if (typeof knowledgeDocumentDetailDialog?.close === "function") {
                knowledgeDocumentDetailDialog.close();
            } else {
                knowledgeDocumentDetailDialog?.removeAttribute("open");
            }
        }

        function appendKnowledgeDetailField(parent, label, value, className = "") {
            if (value === null || value === undefined || value === "") {
                return;
            }
            const field = document.createElement("div");
            field.className = `knowledge-detail-field ${className}`.trim();
            const labelElement = document.createElement("span");
            labelElement.className = "knowledge-detail-label";
            labelElement.textContent = label;
            const valueElement = document.createElement("div");
            valueElement.className = "knowledge-detail-value";
            valueElement.textContent = String(value);
            field.append(labelElement, valueElement);
            parent.append(field);
        }

        function knowledgeFileExtension(filename) {
            const match = String(filename || "").match(/\.([a-z0-9]{1,8})$/i);
            return match ? match[1].toUpperCase() : "FILE";
        }

        function knowledgeFileIcon(filename) {
            const icon = document.createElement("div");
            icon.className = "knowledge-file-icon";
            icon.textContent = knowledgeFileExtension(filename);
            icon.setAttribute("aria-hidden", "true");
            return icon;
        }

        function isKnowledgeTextDocument(item) {
            const filename = String(item?.filename || "").toLowerCase();
            const mimeType = String(item?.mime_type || "").toLowerCase();
            return [".md", ".markdown", ".txt"].some((suffix) => filename.endsWith(suffix))
                || mimeType === "text/markdown"
                || mimeType === "text/plain";
        }

        async function loadKnowledgeDocumentPreview(item, previewElement) {
            try {
                const response = await fetch(
                    `/knowledge/documents/${encodeURIComponent(item.document_id)}/download`,
                    { headers: { Accept: "text/plain" } },
                );
                if (!response.ok) {
                    throw new Error("knowledge preview failed");
                }
                const content = await response.text();
                if (
                    focusedKnowledgeDocumentId === item.document_id
                    && previewElement.isConnected
                ) {
                    previewElement.textContent = content || "（文件内容为空）";
                }
            } catch {
                if (previewElement.isConnected) {
                    previewElement.textContent = "正文加载失败，请使用下载文件查看。";
                }
            }
        }

        function knowledgeDocumentActionButton(label, callback, className = "") {
            const button = document.createElement("button");
            button.type = "button";
            button.className = `knowledge-action ${className}`.trim();
            button.textContent = label;
            button.addEventListener("click", callback);
            return button;
        }

        function renderKnowledgeDocumentDetail(item) {
            if (!knowledgeDocumentDetail || !item) {
                return;
            }
            knowledgeDocumentDetail.replaceChildren();

            const preview = document.createElement("div");
            preview.className = "knowledge-document-detail-preview";
            preview.append(knowledgeFileIcon(item.filename));
            const previewLabel = document.createElement("span");
            previewLabel.textContent = knowledgeFileExtension(item.filename);
            preview.append(previewLabel);

            const info = document.createElement("div");
            info.className = "knowledge-document-detail-info";
            const title = document.createElement("h3");
            title.textContent = item.filename || "未命名文件";
            info.append(title);
            appendKnowledgeDetailField(info, "状态", knowledgeStatusLabels[item.status] || "未知");
            appendKnowledgeDetailField(info, "文件类型", item.mime_type);
            appendKnowledgeDetailField(info, "文件大小", formatKnowledgeBytes(item.size_bytes));
            appendKnowledgeDetailField(info, "页数", Number(item.page_count) || 0);
            appendKnowledgeDetailField(info, "Chunk 数量", Number(item.chunk_count) || 0);
            appendKnowledgeDetailField(info, "创建时间", item.created_at);
            appendKnowledgeDetailField(info, "更新时间", item.updated_at);
            appendKnowledgeDetailField(info, "文档 ID", item.document_id, "is-long");
            if (item.error) {
                appendKnowledgeDetailField(info, "错误信息", item.error, "is-error is-long");
            }

            const actions = document.createElement("div");
            actions.className = "knowledge-document-detail-actions";
            if (item.status !== "deleted") {
                const download = document.createElement("a");
                download.className = "knowledge-action";
                download.textContent = "下载文件";
                download.href = `/knowledge/documents/${encodeURIComponent(item.document_id)}/download`;
                actions.append(download);
            }
            if (item.status === "failed" || item.status === "indexed") {
                actions.append(knowledgeDocumentActionButton(
                    "重新索引",
                    () => void reindexKnowledgeDocument(item.document_id),
                ));
            }
            if (item.status !== "deleted" && item.status !== "deleting") {
                actions.append(knowledgeDocumentActionButton(
                    "删除文件",
                    () => {
                        closeKnowledgeDocumentDetail();
                        void deleteKnowledgeDocument(item.document_id);
                    },
                    "knowledge-danger-action",
                ));
            }
            knowledgeDocumentDetail.append(preview, info, actions);

            if (isKnowledgeTextDocument(item)) {
                const contentPreview = document.createElement("section");
                contentPreview.className = "knowledge-document-content-preview";
                const contentTitle = document.createElement("h3");
                contentTitle.textContent = item.filename?.toLowerCase().endsWith(".md")
                    || item.filename?.toLowerCase().endsWith(".markdown")
                    ? "Markdown 正文"
                    : "文本正文";
                const content = document.createElement("pre");
                content.className = "knowledge-document-content";
                content.textContent = "正在加载正文...";
                contentPreview.append(contentTitle, content);
                knowledgeDocumentDetail.append(contentPreview);
                void loadKnowledgeDocumentPreview(item, content);
            }
        }

        function openKnowledgeDocumentDetail(item) {
            if (!item) {
                return;
            }
            closeKnowledgeContextMenu();
            focusedKnowledgeDocumentId = item.document_id;
            renderKnowledgeDocumentDetail(item);
            if (typeof knowledgeDocumentDetailDialog?.showModal === "function") {
                knowledgeDocumentDetailDialog.showModal();
            } else {
                knowledgeDocumentDetailDialog?.setAttribute("open", "");
            }
        }

        function ensureKnowledgeContextMenu() {
            if (knowledgeContextMenu) {
                return knowledgeContextMenu;
            }
            knowledgeContextMenu = document.createElement("div");
            knowledgeContextMenu.className = "knowledge-context-menu";
            knowledgeContextMenu.setAttribute("role", "menu");
            knowledgeContextMenu.hidden = true;
            document.body.append(knowledgeContextMenu);
            return knowledgeContextMenu;
        }

        function showKnowledgeContextMenu(event, item) {
            event.preventDefault();
            const menu = ensureKnowledgeContextMenu();
            menu.replaceChildren();
            const addAction = (label, callback, disabled = false, className = "") => {
                const button = document.createElement("button");
                button.type = "button";
                button.setAttribute("role", "menuitem");
                button.className = className;
                button.textContent = label;
                button.disabled = disabled;
                button.addEventListener("click", () => {
                    closeKnowledgeContextMenu();
                    callback();
                });
                menu.append(button);
            };
            addAction("查看详情", () => openKnowledgeDocumentDetail(item));
            if (item.status !== "deleted") {
                addAction("下载文件", () => {
                    window.location.href = `/knowledge/documents/${encodeURIComponent(item.document_id)}/download`;
                });
            }
            if (item.status === "failed" || item.status === "indexed") {
                addAction("重新索引", () => void reindexKnowledgeDocument(item.document_id));
            }
            if (item.status !== "deleted" && item.status !== "deleting") {
                addAction("删除文件", () => void deleteKnowledgeDocument(item.document_id), false, "knowledge-danger-action");
            }
            const menuWidth = 180;
            const menuHeight = Math.min(260, menu.childElementCount * 42 + 12);
            menu.style.left = `${Math.max(8, Math.min(event.clientX, window.innerWidth - menuWidth - 8))}px`;
            menu.style.top = `${Math.max(8, Math.min(event.clientY, window.innerHeight - menuHeight - 8))}px`;
            menu.hidden = false;
        }

        function openKnowledgeDialog() {
            if (!knowledgeDialog) {
                return;
            }

            if (typeof knowledgeDialog.showModal === "function") {
                knowledgeDialog.showModal();
            } else {
                knowledgeDialog.setAttribute("open", "");
            }

            void loadKnowledgeDocuments();
        }

        function closeKnowledgeDialog() {
            if (!knowledgeDialog) {
                return;
            }

            if (typeof knowledgeDialog.close === "function") {
                knowledgeDialog.close();
            } else {
                knowledgeDialog.removeAttribute("open");
            }
        }

        function renderKnowledgeStats(stats) {
            if (!knowledgeSummary || !stats) {
                return;
            }

            const values = [
                stats.document_count,
                stats.chunk_count,
                stats.indexed_count,
                stats.processing_count,
                stats.failed_count,
                stats.pending_count,
            ];

            if (!values.every((value) => Number.isInteger(value) && value >= 0)) {
                knowledgeSummary.textContent = "统计数据异常";
                return;
            }

            for (const [key, element] of Object.entries(
                knowledgeOverviewFields,
            )) {
                if (element) {
                    element.textContent = String(stats[key]);
                }
            }

            knowledgeSummary.textContent = [
                `${stats.document_count} 文件`,
                `${stats.chunk_count} Chunk`,
                `${stats.indexed_count} 已索引`,
                `${stats.processing_count} 处理中`,
                `${stats.failed_count} 失败`,
                `${stats.pending_count} 待处理`,
            ].join(" · ");
        }

        function renderKnowledgeDocuments(documents) {
            if (!knowledgeList) {
                return;
            }

            knowledgeDocuments = Array.isArray(documents)
                ? documents.filter((item) => item && typeof item.document_id === "string")
                : [];
            knowledgeList.replaceChildren();

            if (knowledgeDocuments.length === 0) {
                const empty = document.createElement("li");
                empty.className = "knowledge-empty";
                empty.textContent = "暂无文件";
                knowledgeList.append(empty);
                if (focusedKnowledgeDocumentId) {
                    closeKnowledgeDocumentDetail();
                }
                return;
            }

            for (const item of knowledgeDocuments) {
                const row = document.createElement("li");
                row.className = "knowledge-item knowledge-file-row";
                row.tabIndex = 0;
                row.setAttribute("role", "button");
                row.title = "点击查看详情，或点击右侧操作按钮";
                row.addEventListener("click", () => openKnowledgeDocumentDetail(item));
                row.addEventListener("keydown", (event) => {
                    if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        openKnowledgeDocumentDetail(item);
                    }
                });
                row.addEventListener("contextmenu", (event) => showKnowledgeContextMenu(event, item));

                const icon = knowledgeFileIcon(item.filename);
                const status = document.createElement("span");
                status.className = `knowledge-file-status knowledge-status-${item.status || "unknown"}`;
                status.textContent = knowledgeStatusLabels[item.status] || "未知";
                icon.append(status);

                const name = document.createElement("strong");
                name.className = "knowledge-name";
                name.textContent = item.filename || "未命名文件";

                const meta = document.createElement("span");
                meta.className = "knowledge-file-meta";
                meta.textContent = `${formatKnowledgeBytes(item.size_bytes)} · ${Number(item.chunk_count) || 0} Chunks`;

                const hint = document.createElement("span");
                hint.className = "knowledge-file-hint";
                hint.textContent = `${Number(item.page_count) || 0} 页 · 点击查看详情`;

                const copy = document.createElement("div");
                copy.className = "knowledge-file-copy";
                copy.append(name, meta, hint);

                const actionButton = document.createElement("button");
                actionButton.type = "button";
                actionButton.className = "knowledge-row-action";
                actionButton.textContent = "操作";
                actionButton.title = "打开文件操作菜单";
                actionButton.addEventListener("click", (event) => {
                    event.stopPropagation();
                    const bounds = actionButton.getBoundingClientRect();
                    showKnowledgeContextMenu(
                        {
                            preventDefault: () => {},
                            clientX: bounds.right,
                            clientY: bounds.bottom,
                        },
                        item,
                    );
                });

                row.append(icon, copy, actionButton);
                knowledgeList.append(row);
            }
            if (focusedKnowledgeDocumentId) {
                const focused = findKnowledgeDocument(focusedKnowledgeDocumentId);
                if (focused && knowledgeDocumentDetailDialog?.open) {
                    renderKnowledgeDocumentDetail(focused);
                } else if (!focused) {
                    closeKnowledgeDocumentDetail();
                }
            }
        }

        async function loadKnowledgeDocuments() {
            try {
                const [response, statsResponse] = await Promise.all([
                    fetch("/knowledge/documents"),
                    fetch("/knowledge/stats"),
                ]);

                if (!response.ok) {
                    throw new Error("knowledge list failed");
                }

                const documents = await response.json();

                if (!Array.isArray(documents)) {
                    throw new Error("knowledge list response is invalid");
                }

                renderKnowledgeDocuments(documents);
                setKnowledgeStatus("");

                if (statsResponse.ok) {
                    renderKnowledgeStats(await statsResponse.json());
                } else if (knowledgeSummary) {
                    knowledgeSummary.textContent = "统计暂不可用";
                }
            } catch (error) {
                console.error("Knowledge list load failed", error);
                setKnowledgeStatus("知识库加载失败");
            }
        }

        async function uploadKnowledgeDocument(file) {
            if (!file || knowledgeFileInput.disabled) {
                return;
            }

            knowledgeFileInput.disabled = true;
            setKnowledgeStatus("正在上传并索引...");

            try {
                const formData = new FormData();
                formData.append("file", file);

                const response = await fetch(
                    "/knowledge/documents",
                    {
                        method: "POST",
                        body: formData,
                    },
                );

                if (!response.ok) {
                    if (response.status === 409) {
                        setKnowledgeStatus(
                            "文件已存在，请在列表中重新索引",
                        );
                        await loadKnowledgeDocuments();
                        return;
                    }

                    throw new Error("knowledge upload failed");
                }

                setKnowledgeStatus("已加入知识库");
                await loadKnowledgeDocuments();
            } catch {
                setKnowledgeStatus("上传或索引失败");
                await loadKnowledgeDocuments();
            } finally {
                knowledgeFileInput.disabled = false;
                knowledgeFileInput.value = "";
            }
        }

        async function reindexKnowledgeDocument(documentId) {
            setKnowledgeStatus("正在重新索引...");

            try {
                const response = await fetch(
                    `/knowledge/documents/${encodeURIComponent(documentId)}/reindex`,
                    { method: "POST" },
                );

                if (!response.ok) {
                    throw new Error("knowledge reindex failed");
                }

                setKnowledgeStatus("重新索引完成");
                await loadKnowledgeDocuments();
            } catch {
                setKnowledgeStatus("重新索引失败");
            }
        }

        async function deleteKnowledgeDocument(documentId) {
            if (!window.confirm("确定删除这个知识库文件吗？")) {
                return;
            }

            setKnowledgeStatus("正在删除...");

            try {
                const response = await fetch(
                    `/knowledge/documents/${encodeURIComponent(documentId)}`,
                    { method: "DELETE" },
                );

                if (!response.ok) {
                    throw new Error("knowledge delete failed");
                }

                setKnowledgeStatus("已删除");
                await loadKnowledgeDocuments();
            } catch {
                setKnowledgeStatus("删除失败");
            }
        }

        knowledgeDocumentDetailCloseButton?.addEventListener(
            "click",
            closeKnowledgeDocumentDetail,
        );
        knowledgeDocumentDetailDialog?.addEventListener("close", () => {
            focusedKnowledgeDocumentId = null;
        });
        document.addEventListener("click", (event) => {
            if (knowledgeContextMenu && !knowledgeContextMenu.contains(event.target)) {
                closeKnowledgeContextMenu();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") {
                closeKnowledgeContextMenu();
            }
        });
        window.addEventListener("scroll", closeKnowledgeContextMenu, true);
