        const imageUploadButton = document.querySelector("#image-upload-button");
        const imageLibraryOpenButton = document.querySelector("#image-library-open");
        const imageLibraryDialog = document.querySelector("#image-library-dialog");
        const imageLibraryCloseButton = document.querySelector("#image-library-close");
        const imageLibraryRefreshButton = document.querySelector("#image-library-refresh");
        const imageLibraryUploadButton = document.querySelector("#image-library-upload-button");
        const imageFileInput = document.querySelector("#image-file-input");
        const imageAttachments = document.querySelector("#image-attachments");
        const messageAttachments = document.querySelector("#message-attachments");
        const imageAttachmentDetailDialog = document.querySelector("#image-attachment-detail-dialog");
        const imageAttachmentDetail = document.querySelector("#image-attachment-detail");
        const imageAttachmentDetailCloseButton = document.querySelector("#image-attachment-detail-close");
        const composer = document.querySelector(".composer");

        const IMAGE_MAX_BYTES = 10 * 1024 * 1024;
        const IMAGE_TYPES = new Set([
            "image/jpeg",
            "image/png",
            "image/gif",
        ]);
        const attachmentErrorLabels = {
            image_attachment_type_unsupported: "只支持 JPG、JPEG、PNG、GIF 图片",
            image_attachment_empty: "图片内容为空",
            image_attachment_too_large: "图片不能超过 10 MB",
            image_attachment_not_configured: "图片上传服务暂不可用",
            wechat_cover_not_configured: "微信公众号封面服务暂不可用",
            wechat_cover_attachment_failed: "设置微信公众号封面失败",
        };

        let currentImageAttachments = [];
        const selectedPublicationAttachments = new Set();
        const pendingMessageAttachments = new Set();
        const knownAnalysisStatuses = new Map();
        let attachmentRequestNumber = 0;
        let attachmentBusy = false;
        let inlineImagePickerCallback = null;
        let focusedImageAttachmentId = null;
        let imageAttachmentContextMenu = null;

        function uniqueImageAttachments(attachments) {
            const seen = new Set();
            const unique = [];
            for (const attachment of Array.isArray(attachments) ? attachments : []) {
                if (!attachment || typeof attachment.attachment_id !== "string") {
                    continue;
                }
                const attachmentId = attachment.attachment_id.trim();
                if (!attachmentId) {
                    continue;
                }
                const contentHash = typeof attachment.content_sha256 === "string"
                    ? attachment.content_sha256.trim().toLowerCase()
                    : "";
                // Keep the UI safe when an older server/database returns
                // historical duplicate rows for the same image.
                const identity = contentHash
                    ? `content:${contentHash}`
                    : `id:${attachmentId}`;
                if (seen.has(identity)) {
                    continue;
                }
                seen.add(identity);
                unique.push({ ...attachment, attachment_id: attachmentId });
            }
            return unique;
        }

        function attachmentThreadId() {
            const getter = globalThis.getCurrentResearchThreadId;
            const value = typeof getter === "function" ? getter() : "";
            return typeof value === "string" ? value.trim() : "";
        }

        function attachmentErrorCode(payload) {
            const candidate = payload?.detail?.error_code;
            return typeof candidate === "string" && /^[a-z0-9_]{1,80}$/i.test(candidate)
                ? candidate
                : "publishing_operation_failed";
        }

        function attachmentErrorMessage(code) {
            const safeCode = typeof code === "string" && /^[a-z0-9_]{1,80}$/i.test(code)
                ? code
                : "publishing_operation_failed";
            return `${attachmentErrorLabels[safeCode] || "图片操作失败"}（错误代码：${safeCode}）`;
        }

        function formatAttachmentSize(bytes) {
            if (!Number.isFinite(bytes) || bytes < 1024) {
                return `${Math.max(0, Number(bytes) || 0)} B`;
            }
            const units = ["KB", "MB", "GB"];
            let value = bytes;
            let index = -1;
            do {
                value /= 1024;
                index += 1;
            } while (value >= 1024 && index < units.length - 1);
            return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[index]}`;
        }

        function analysisStatusLabel(status) {
            return {
                pending: "等待分析",
                analyzing: "正在分析",
                completed: "已完成分析",
                skipped: "未配置分析服务",
                failed: "分析失败",
            }[status] || "等待分析";
        }

        function attachmentPreviewUrl(attachment, threadId) {
            if (attachment.local_url) {
                return attachment.local_url;
            }
            if (!attachment.attachment_id || !threadId) {
                return "";
            }
            return `/publishing/attachments/images/${encodeURIComponent(attachment.attachment_id)}/content?thread_id=${encodeURIComponent(threadId)}`;
        }

        function findImageAttachment(attachmentId) {
            return uniqueImageAttachments(currentImageAttachments).find(
                (attachment) => attachment.attachment_id === attachmentId,
            ) || null;
        }

        function closeImageAttachmentContextMenu() {
            if (!imageAttachmentContextMenu) {
                return;
            }
            imageAttachmentContextMenu.hidden = true;
            imageAttachmentContextMenu.replaceChildren();
        }

        function closeImageAttachmentDetail() {
            focusedImageAttachmentId = null;
            if (typeof imageAttachmentDetailDialog?.close === "function") {
                imageAttachmentDetailDialog.close();
            } else {
                imageAttachmentDetailDialog?.removeAttribute("open");
            }
        }

        function appendAttachmentDetailField(parent, label, value, className = "") {
            if (!value) {
                return;
            }
            const field = document.createElement("div");
            field.className = `image-attachment-detail-field ${className}`.trim();
            const labelElement = document.createElement("span");
            labelElement.className = "image-attachment-detail-label";
            labelElement.textContent = label;
            const valueElement = document.createElement("div");
            valueElement.className = "image-attachment-detail-value";
            valueElement.textContent = value;
            field.append(labelElement, valueElement);
            parent.append(field);
        }

        function togglePendingMessageAttachment(attachmentId) {
            if (pendingMessageAttachments.has(attachmentId)) {
                pendingMessageAttachments.delete(attachmentId);
            } else {
                pendingMessageAttachments.add(attachmentId);
            }
            renderImageAttachments();
            renderMessageAttachments();
        }

        function toggleSelectedPublicationAttachment(attachmentId) {
            if (selectedPublicationAttachments.has(attachmentId)) {
                selectedPublicationAttachments.delete(attachmentId);
            } else {
                selectedPublicationAttachments.add(attachmentId);
            }
            renderImageAttachments();
        }

        function invokeInlineImagePicker(attachment) {
            if (typeof inlineImagePickerCallback !== "function") {
                return false;
            }
            const callback = inlineImagePickerCallback;
            inlineImagePickerCallback = null;
            callback({ ...attachment });
            if (typeof imageLibraryDialog?.close === "function") {
                imageLibraryDialog.close();
            } else {
                imageLibraryDialog?.removeAttribute("open");
            }
            closeImageAttachmentDetail();
            return true;
        }

        function renderImageAttachmentDetail(attachment) {
            if (!imageAttachmentDetail || !attachment) {
                return;
            }
            imageAttachmentDetail.replaceChildren();
            const currentThreadId = attachmentThreadId();
            const previewUrl = attachmentPreviewUrl(attachment, currentThreadId);
            if (previewUrl) {
                const image = document.createElement("img");
                image.className = "image-attachment-detail-preview";
                image.alt = attachment.filename || "图片附件";
                image.src = previewUrl;
                imageAttachmentDetail.append(image);
            }

            const info = document.createElement("div");
            info.className = "image-attachment-detail-info";
            const title = document.createElement("h3");
            title.textContent = attachment.filename || "未命名图片";
            info.append(title);
            appendAttachmentDetailField(info, "大小", formatAttachmentSize(attachment.size_bytes));
            appendAttachmentDetailField(info, "来源", attachment.thread_id && attachment.thread_id !== currentThreadId ? "其他对话" : "当前对话");
            appendAttachmentDetailField(info, "状态", attachment.status === "failed" ? "上传失败" : "已上传");
            appendAttachmentDetailField(info, "图片类型", attachment.analysis_type || "尚未识别");
            appendAttachmentDetailField(info, "分析结果", attachment.analysis_summary, "is-long");
            appendAttachmentDetailField(info, "OCR 文字", attachment.analysis_ocr_text || attachment.ocr_text, "is-long");
            appendAttachmentDetailField(info, "分析状态", analysisStatusLabel(attachment.analysis_status));
            if (attachment.analysis_error_code) {
                appendAttachmentDetailField(info, "分析错误", attachment.analysis_error_code, "is-error");
            }
            if (attachment.is_active) {
                appendAttachmentDetailField(info, "微信公众号封面", "当前封面");
            }
            imageAttachmentDetail.append(info);

            const actions = document.createElement("div");
            actions.className = "image-attachment-detail-actions";
            if (attachment.status === "uploaded") {
                const messageButton = document.createElement("button");
                messageButton.type = "button";
                messageButton.className = "secondary-action";
                messageButton.textContent = pendingMessageAttachments.has(attachment.attachment_id)
                    ? "取消附加到下一条消息"
                    : "附加到下一条消息";
                messageButton.addEventListener("click", () => togglePendingMessageAttachment(attachment.attachment_id));
                actions.append(messageButton);

                const publicationButton = document.createElement("button");
                publicationButton.type = "button";
                publicationButton.className = "secondary-action";
                publicationButton.textContent = selectedPublicationAttachments.has(attachment.attachment_id)
                    ? "取消选择配图"
                    : "选择为本次发布配图";
                publicationButton.addEventListener("click", () => toggleSelectedPublicationAttachment(attachment.attachment_id));
                actions.append(publicationButton);

                const coverButton = document.createElement("button");
                coverButton.type = "button";
                coverButton.className = "secondary-action";
                coverButton.textContent = attachment.is_active ? "当前微信公众号封面" : "设为微信公众号封面";
                coverButton.disabled = Boolean(attachment.is_active || attachmentBusy);
                coverButton.addEventListener("click", () => void setWechatCover(attachment.attachment_id));
                actions.append(coverButton);

                if (typeof inlineImagePickerCallback === "function") {
                    const insertButton = document.createElement("button");
                    insertButton.type = "button";
                    insertButton.className = "secondary-action";
                    insertButton.textContent = "插入正文";
                    insertButton.addEventListener("click", () => invokeInlineImagePicker(attachment));
                    actions.append(insertButton);
                }

                const removeButton = document.createElement("button");
                removeButton.type = "button";
                removeButton.className = "secondary-action danger-action";
                removeButton.textContent = "删除图片";
                removeButton.disabled = Boolean(attachmentBusy);
                removeButton.addEventListener("click", () => {
                    closeImageAttachmentDetail();
                    void removeImageAttachment(attachment.attachment_id);
                });
                actions.append(removeButton);
            }
            imageAttachmentDetail.append(actions);
        }

        function openImageAttachmentDetail(attachment) {
            if (!attachment) {
                return;
            }
            closeImageAttachmentContextMenu();
            focusedImageAttachmentId = attachment.attachment_id;
            renderImageAttachmentDetail(attachment);
            if (typeof imageAttachmentDetailDialog?.showModal === "function") {
                imageAttachmentDetailDialog.showModal();
            } else {
                imageAttachmentDetailDialog?.setAttribute("open", "");
            }
        }

        function ensureImageAttachmentContextMenu() {
            if (imageAttachmentContextMenu) {
                return imageAttachmentContextMenu;
            }
            imageAttachmentContextMenu = document.createElement("div");
            imageAttachmentContextMenu.className = "image-attachment-context-menu";
            imageAttachmentContextMenu.setAttribute("role", "menu");
            imageAttachmentContextMenu.hidden = true;
            document.body.append(imageAttachmentContextMenu);
            return imageAttachmentContextMenu;
        }

        function showImageAttachmentContextMenu(event, attachment) {
            event.preventDefault();
            const menu = ensureImageAttachmentContextMenu();
            menu.replaceChildren();
            const addAction = (label, callback, disabled = false) => {
                const button = document.createElement("button");
                button.type = "button";
                button.setAttribute("role", "menuitem");
                button.textContent = label;
                button.disabled = disabled;
                button.addEventListener("click", () => {
                    closeImageAttachmentContextMenu();
                    callback();
                });
                menu.append(button);
            };
            addAction("查看详情", () => openImageAttachmentDetail(attachment));
            if (attachment.status === "uploaded") {
                addAction(
                    pendingMessageAttachments.has(attachment.attachment_id) ? "取消附加到下一条消息" : "附加到下一条消息",
                    () => togglePendingMessageAttachment(attachment.attachment_id),
                );
                addAction(
                    selectedPublicationAttachments.has(attachment.attachment_id) ? "取消选择发布配图" : "选择为本次发布配图",
                    () => toggleSelectedPublicationAttachment(attachment.attachment_id),
                );
                addAction(
                    attachment.is_active ? "当前微信公众号封面" : "设为微信公众号封面",
                    () => void setWechatCover(attachment.attachment_id),
                    Boolean(attachment.is_active || attachmentBusy),
                );
                if (typeof inlineImagePickerCallback === "function") {
                    addAction("插入正文", () => invokeInlineImagePicker(attachment));
                }
                addAction("删除图片", () => void removeImageAttachment(attachment.attachment_id), Boolean(attachmentBusy));
            }
            const menuWidth = 220;
            const menuHeight = Math.min(320, menu.childElementCount * 42 + 12);
            menu.style.left = `${Math.max(8, Math.min(event.clientX, window.innerWidth - menuWidth - 8))}px`;
            menu.style.top = `${Math.max(8, Math.min(event.clientY, window.innerHeight - menuHeight - 8))}px`;
            menu.hidden = false;
        }

        function renderMessageAttachments() {
            if (!messageAttachments) {
                return;
            }
            messageAttachments.replaceChildren();
            const currentThreadId = attachmentThreadId();
            const attachments = uniqueImageAttachments(currentImageAttachments).filter((attachment) => (
                attachment.status === "uploaded" &&
                pendingMessageAttachments.has(attachment.attachment_id)
            ));
            messageAttachments.hidden = attachments.length === 0;
            for (const attachment of attachments) {
                const chip = document.createElement("div");
                chip.className = "message-attachment";

                const previewUrl = attachmentPreviewUrl(attachment, currentThreadId);
                if (previewUrl) {
                    const image = document.createElement("img");
                    image.className = "message-attachment-preview";
                    image.alt = attachment.filename || "已附加图片";
                    image.src = previewUrl;
                    chip.append(image);
                }

                const name = document.createElement("span");
                name.className = "message-attachment-name";
                name.textContent = attachment.filename || "图片附件";
                chip.append(name);

                const remove = document.createElement("button");
                remove.type = "button";
                remove.className = "message-attachment-remove";
                remove.textContent = "×";
                remove.title = "从本条消息移除（不会删除图片库中的图片）";
                remove.addEventListener("click", () => {
                    pendingMessageAttachments.delete(attachment.attachment_id);
                    renderMessageAttachments();
                });
                chip.append(remove);
                messageAttachments.append(chip);
            }
        }

        function renderImageAttachments() {
            if (!imageAttachments) {
                return;
            }
            imageAttachments.replaceChildren();
            if (currentImageAttachments.length === 0) {
                imageAttachments.hidden = true;
                if (focusedImageAttachmentId) {
                    closeImageAttachmentDetail();
                }
                return;
            }
            imageAttachments.hidden = false;

            const currentThreadId = attachmentThreadId();
            for (const attachment of uniqueImageAttachments(currentImageAttachments)) {
                const item = document.createElement("article");
                item.className = "image-attachment image-attachment-tile";
                item.dataset.active = attachment.is_active ? "true" : "false";
                item.dataset.pending = pendingMessageAttachments.has(attachment.attachment_id) ? "true" : "false";
                item.dataset.selected = selectedPublicationAttachments.has(attachment.attachment_id) ? "true" : "false";
                item.tabIndex = 0;
                item.setAttribute("role", "button");
                item.title = "点击查看详情，右键打开操作菜单";
                item.addEventListener("click", () => openImageAttachmentDetail(attachment));
                item.addEventListener("keydown", (event) => {
                    if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        openImageAttachmentDetail(attachment);
                    }
                });
                item.addEventListener("contextmenu", (event) => showImageAttachmentContextMenu(event, attachment));

                const previewUrl = attachmentPreviewUrl(attachment, currentThreadId);
                if (previewUrl) {
                    const image = document.createElement("img");
                    image.className = "image-attachment-preview";
                    image.alt = attachment.filename || "图片附件";
                    image.src = previewUrl;
                    image.addEventListener("error", () => {
                        image.hidden = true;
                    });
                    item.append(image);
                }

                const overlay = document.createElement("div");
                overlay.className = "image-attachment-overlay";
                const filename = document.createElement("strong");
                filename.textContent = attachment.filename || "未命名图片";
                const meta = document.createElement("span");
                meta.textContent = `${formatAttachmentSize(attachment.size_bytes)} · ${analysisStatusLabel(attachment.analysis_status)}`;
                overlay.append(filename, meta);

                const badges = document.createElement("div");
                badges.className = "image-attachment-badges";
                if (attachment.is_active) {
                    const badge = document.createElement("span");
                    badge.textContent = "微信封面";
                    badges.append(badge);
                }
                if (pendingMessageAttachments.has(attachment.attachment_id)) {
                    const badge = document.createElement("span");
                    badge.textContent = "待发送";
                    badges.append(badge);
                }
                if (selectedPublicationAttachments.has(attachment.attachment_id)) {
                    const badge = document.createElement("span");
                    badge.textContent = "已选配图";
                    badges.append(badge);
                }
                if (badges.childElementCount) {
                overlay.append(badges);
                }
                item.append(overlay);
                imageAttachments.append(item);
            }
            if (focusedImageAttachmentId && imageAttachmentDetailDialog?.open) {
                const focused = findImageAttachment(focusedImageAttachmentId);
                if (focused) {
                    renderImageAttachmentDetail(focused);
                } else {
                    closeImageAttachmentDetail();
                }
            }
        }

        async function removeImageAttachment(attachmentId) {
            const target = currentImageAttachments.find(
                (attachment) => attachment.attachment_id === attachmentId,
            );
            if (!target || attachmentBusy) {
                return;
            }
            const message = target.is_active
                ? "这张图片是当前微信公众号封面，删除后将取消当前封面。确定继续吗？"
                : "确定从共享图片库删除这张图片吗？";
            if (!window.confirm(message)) {
                return;
            }

            attachmentBusy = true;
            renderImageAttachments();
            try {
                const response = await fetch(
                    `/publishing/attachments/${encodeURIComponent(attachmentId)}`,
                    {
                        method: "DELETE",
                        headers: { Accept: "application/json" },
                    },
                );
                const payload = await response.json().catch(() => null);
                if (!response.ok) {
                    throw new Error(attachmentErrorCode(payload));
                }
                if (target.local_url && target.local_url.startsWith("blob:")) {
                    URL.revokeObjectURL(target.local_url);
                }
            currentImageAttachments = currentImageAttachments.filter(
                    (attachment) => attachment.attachment_id !== attachmentId,
                );
                pendingMessageAttachments.delete(attachmentId);
                selectedPublicationAttachments.delete(attachmentId);
                if (target.is_active) {
                    document.dispatchEvent(
                        new CustomEvent("wechat-cover-changed", {
                            detail: {
                                thread_id: attachmentThreadId(),
                                attachment: null,
                            },
                        }),
                    );
                }
            } catch (error) {
                renderAttachmentOperationError(error.message);
            } finally {
                attachmentBusy = false;
                renderImageAttachments();
                renderMessageAttachments();
            }
        }

        async function loadImageAttachments() {
            const requestNumber = ++attachmentRequestNumber;
            const currentThreadId = attachmentThreadId();
            currentImageAttachments = [];
            renderImageAttachments();
            if (!currentThreadId) {
                return;
            }
            try {
                const response = await fetch(
                    `/publishing/attachments/images?thread_id=${encodeURIComponent(currentThreadId)}`,
                    { headers: { Accept: "application/json" } },
                );
                const payload = await response.json().catch(() => null);
                if (!response.ok) {
                    throw new Error(attachmentErrorCode(payload));
                }
                if (requestNumber !== attachmentRequestNumber || currentThreadId !== attachmentThreadId()) {
                    return;
                }
                currentImageAttachments = uniqueImageAttachments(
                    Array.isArray(payload)
                        ? payload
                            .filter((attachment) => attachment && typeof attachment === "object")
                            .map((attachment) => ({ ...attachment, status: "uploaded" }))
                        : [],
                );
                for (const attachment of currentImageAttachments) {
                    const previousStatus = knownAnalysisStatuses.get(
                        attachment.attachment_id,
                    );
                    const currentStatus = attachment.analysis_status || "pending";
                    if (
                        currentStatus === "completed" &&
                        ["pending", "analyzing"].includes(previousStatus) &&
                        typeof globalThis.renderImageAnalysisCard === "function"
                    ) {
                        globalThis.renderImageAnalysisCard(attachment);
                    }
                    knownAnalysisStatuses.set(
                        attachment.attachment_id,
                        currentStatus,
                    );
                }
                const availableIds = new Set(
                    currentImageAttachments.map((attachment) => attachment.attachment_id),
                );
                for (const attachmentId of selectedPublicationAttachments) {
                    if (!availableIds.has(attachmentId)) {
                        selectedPublicationAttachments.delete(attachmentId);
                    }
                }
                for (const attachmentId of pendingMessageAttachments) {
                    if (!availableIds.has(attachmentId)) {
                        pendingMessageAttachments.delete(attachmentId);
                    }
                }
                renderImageAttachments();
                renderMessageAttachments();
            } catch (error) {
                if (requestNumber !== attachmentRequestNumber) {
                    return;
                }
                imageAttachments?.replaceChildren();
                if (imageAttachments) {
                    imageAttachments.hidden = false;
                    const message = document.createElement("span");
                    message.className = "image-attachment-error";
                    message.textContent = attachmentErrorMessage(error.message);
                    imageAttachments.append(message);
                }
            }
        }

        async function uploadImageFile(file) {
            const currentThreadId = attachmentThreadId();
            if (!file || !currentThreadId) {
                return;
            }
            if (!IMAGE_TYPES.has(file.type)) {
                renderAttachmentOperationError("image_attachment_type_unsupported");
                return;
            }
            if (file.size <= 0) {
                renderAttachmentOperationError("image_attachment_empty");
                return;
            }
            if (file.size > IMAGE_MAX_BYTES) {
                renderAttachmentOperationError("image_attachment_too_large");
                return;
            }

            const localUrl = URL.createObjectURL(file);
            const pending = {
                attachment_id: `pending-${Date.now()}-${Math.random().toString(16).slice(2)}`,
                thread_id: currentThreadId,
                filename: file.name || "image",
                content_type: file.type,
                size_bytes: file.size,
                status: "uploading",
                local_url: localUrl,
            };
            currentImageAttachments = uniqueImageAttachments([
                ...currentImageAttachments,
                pending,
            ]);
            renderImageAttachments();

            const formData = new FormData();
            formData.append("thread_id", currentThreadId);
            formData.append("file", file, file.name || "image");
            try {
                const response = await fetch("/publishing/attachments/images", {
                    method: "POST",
                    body: formData,
                    headers: { Accept: "application/json" },
                });
                const payload = await response.json().catch(() => null);
                if (!response.ok) {
                    throw new Error(attachmentErrorCode(payload));
                }
                if (currentThreadId !== attachmentThreadId()) {
                    URL.revokeObjectURL(localUrl);
                    return;
                }
                const saved = {
                    ...payload,
                    status: "uploaded",
                    local_url: localUrl,
                };
                knownAnalysisStatuses.set(
                    saved.attachment_id,
                    saved.analysis_status || "pending",
                );
                pendingMessageAttachments.add(saved.attachment_id);
                currentImageAttachments = uniqueImageAttachments(currentImageAttachments.map(
                    (attachment) => attachment.attachment_id === pending.attachment_id
                        ? saved
                        : attachment,
                ));
                renderImageAttachments();
                renderMessageAttachments();
            } catch (error) {
                const message = attachmentErrorMessage(error.message);
                currentImageAttachments = uniqueImageAttachments(currentImageAttachments.map(
                    (attachment) => attachment.attachment_id === pending.attachment_id
                        ? { ...attachment, status: "failed", status_message: message }
                        : attachment,
                ));
                renderImageAttachments();
                renderMessageAttachments();
            }
        }

        function renderAttachmentOperationError(code) {
            if (!imageAttachments) {
                return;
            }
            imageAttachments.hidden = false;
            const message = document.createElement("span");
            message.className = "image-attachment-error";
            message.textContent = attachmentErrorMessage(code);
            imageAttachments.append(message);
        }

        async function setWechatCover(attachmentId) {
            const currentThreadId = attachmentThreadId();
            if (!currentThreadId || attachmentBusy) {
                return;
            }
            attachmentBusy = true;
            renderImageAttachments();
            try {
                const response = await fetch(
                    `/publishing/attachments/${encodeURIComponent(attachmentId)}/wechat-cover`,
                    {
                        method: "POST",
                        headers: {
                            Accept: "application/json",
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({ thread_id: currentThreadId }),
                    },
                );
                const payload = await response.json().catch(() => null);
                if (!response.ok) {
                    throw new Error(attachmentErrorCode(payload));
                }
                if (currentThreadId !== attachmentThreadId()) {
                    return;
                }
                currentImageAttachments = uniqueImageAttachments(currentImageAttachments.map(
                    (attachment) => ({
                        ...attachment,
                        is_active: attachment.attachment_id === payload.attachment_id,
                        cover_asset_id: attachment.attachment_id === payload.attachment_id
                            ? payload.cover_asset_id
                            : null,
                    }),
                ));
                renderImageAttachments();
                document.dispatchEvent(
                    new CustomEvent("wechat-cover-changed", {
                        detail: {
                            thread_id: currentThreadId,
                            attachment: currentImageAttachments.find(
                                (attachment) => attachment.is_active,
                            ) || null,
                        },
                    }),
                );
            } catch (error) {
                renderAttachmentOperationError(error.message);
            } finally {
                attachmentBusy = false;
                renderImageAttachments();
            }
        }

        globalThis.refreshWechatAttachments = loadImageAttachments;
        globalThis.openInlineImagePicker = (callback) => {
            if (typeof callback !== "function") {
                return;
            }
            inlineImagePickerCallback = callback;
            if (typeof imageLibraryDialog?.showModal === "function") {
                imageLibraryDialog.showModal();
            } else {
                imageLibraryDialog?.setAttribute("open", "");
            }
            void loadImageAttachments();
        };
        globalThis.getWechatCoverPreview = () => {
            const active = currentImageAttachments.find(
                (attachment) => attachment.is_active,
            );
            return active ? { ...active, thread_id: attachmentThreadId() } : null;
        };
        globalThis.getSelectedPublicationAttachmentIds = () => (
            uniqueImageAttachments(currentImageAttachments)
                .filter((attachment) => (
                    attachment.status === "uploaded" &&
                    selectedPublicationAttachments.has(attachment.attachment_id)
                ))
                .map((attachment) => attachment.attachment_id)
        );
        globalThis.getPendingMessageAttachmentIds = () => (
            uniqueImageAttachments(currentImageAttachments)
                .filter((attachment) => (
                    attachment.status === "uploaded" &&
                    pendingMessageAttachments.has(attachment.attachment_id)
                ))
                .map((attachment) => attachment.attachment_id)
        );
        globalThis.hasPendingImageUploads = () => currentImageAttachments.some(
            (attachment) => attachment.status === "uploading",
        );
        globalThis.clearPendingMessageAttachmentIds = () => {
            pendingMessageAttachments.clear();
            renderMessageAttachments();
        };
        globalThis.restorePendingMessageAttachmentIds = (attachmentIds) => {
            if (!Array.isArray(attachmentIds)) {
                return;
            }
            for (const attachmentId of attachmentIds) {
                if (typeof attachmentId === "string" && attachmentId.trim()) {
                    pendingMessageAttachments.add(attachmentId.trim());
                }
            }
            renderImageAttachments();
            renderMessageAttachments();
        };
        globalThis.selectPublicationAttachmentIds = (attachmentIds) => {
            if (!Array.isArray(attachmentIds)) {
                return;
            }
            for (const attachmentId of attachmentIds) {
                if (typeof attachmentId === "string" && attachmentId.trim()) {
                    selectedPublicationAttachments.add(attachmentId.trim());
                }
            }
            renderImageAttachments();
        };
        globalThis.setPublicationAttachmentIds = (attachmentIds) => {
            selectedPublicationAttachments.clear();
            if (Array.isArray(attachmentIds)) {
                for (const attachmentId of attachmentIds) {
                    if (typeof attachmentId === "string" && attachmentId.trim()) {
                        selectedPublicationAttachments.add(attachmentId.trim());
                    }
                }
            }
            renderImageAttachments();
        };

        imageUploadButton?.addEventListener("click", () => imageFileInput?.click());
        imageLibraryOpenButton?.addEventListener("click", () => {
            if (typeof imageLibraryDialog?.showModal === "function") {
                imageLibraryDialog.showModal();
            } else {
                imageLibraryDialog?.setAttribute("open", "");
            }
            void loadImageAttachments();
        });
        imageLibraryCloseButton?.addEventListener("click", () => {
            inlineImagePickerCallback = null;
            closeImageAttachmentContextMenu();
            if (typeof imageLibraryDialog?.close === "function") {
                imageLibraryDialog.close();
            } else {
                imageLibraryDialog?.removeAttribute("open");
            }
        });
        imageAttachmentDetailCloseButton?.addEventListener("click", closeImageAttachmentDetail);
        imageAttachmentDetailDialog?.addEventListener("close", () => {
            focusedImageAttachmentId = null;
        });
        document.addEventListener("click", (event) => {
            if (imageAttachmentContextMenu && !imageAttachmentContextMenu.contains(event.target)) {
                closeImageAttachmentContextMenu();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") {
                closeImageAttachmentContextMenu();
            }
        });
        window.addEventListener("scroll", closeImageAttachmentContextMenu, true);
        imageLibraryRefreshButton?.addEventListener(
            "click",
            () => void loadImageAttachments(),
        );
        imageLibraryUploadButton?.addEventListener(
            "click",
            () => imageFileInput?.click(),
        );
        imageFileInput?.addEventListener("change", () => {
            const files = Array.from(imageFileInput.files || []);
            imageFileInput.value = "";
            for (const file of files) {
                void uploadImageFile(file);
            }
        });

        composer?.addEventListener("dragover", (event) => {
            if (Array.from(event.dataTransfer?.items || []).some((item) => item.kind === "file")) {
                event.preventDefault();
                composer.classList.add("is-image-dragover");
            }
        });
        composer?.addEventListener("dragleave", () => composer.classList.remove("is-image-dragover"));
        composer?.addEventListener("drop", (event) => {
            composer.classList.remove("is-image-dragover");
            const files = Array.from(event.dataTransfer?.files || []);
            if (files.length === 0) {
                return;
            }
            event.preventDefault();
            for (const file of files) {
                void uploadImageFile(file);
            }
        });
        document.addEventListener("research:thread-changed", () => {
            pendingMessageAttachments.clear();
            renderMessageAttachments();
            void loadImageAttachments();
        });

        window.setInterval(() => {
            // Pending is the normal state for an uploaded-but-unsent image;
            // do not poll it as if an analysis job had been requested.
            if (currentImageAttachments.some((attachment) => (
                attachment.analysis_status === "analyzing"
            ))) {
                void loadImageAttachments();
            }
        }, 3000);
        void loadImageAttachments();
