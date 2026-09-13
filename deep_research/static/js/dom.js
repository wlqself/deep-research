        const form = document.querySelector("#research-form");
        const questionInput = document.querySelector("#question");
        const submitButton = document.querySelector("#submit-button");
        const cancelButton = document.querySelector("#cancel-button");
        const status = document.querySelector("#status");
        const conversation = document.querySelector("#conversation");
        let conversationEmpty = document.querySelector(
            "#conversation-empty",
        );
        const newThreadButton = document.querySelector("#new-thread");
        const threadList = document.querySelector("#thread-list");
        const threadContextMenu = document.querySelector(
            "#thread-context-menu",
        );
        const threadRenameDialog = document.querySelector(
            "#thread-rename-dialog",
        );
        const threadRenameForm = document.querySelector(
            "#thread-rename-form",
        );
        const threadRenameInput = document.querySelector(
            "#thread-rename-input",
        );
        const threadRenameError = document.querySelector(
            "#thread-rename-error",
        );
        const threadRenameCancel = document.querySelector(
            "#thread-rename-cancel",
        );
        const threadRenameSubmit = document.querySelector(
            "#thread-rename-submit",
        );
        const threadSettingsButton = document.querySelector(
            "#thread-settings",
        );
        const settingsDialog = document.querySelector("#settings-dialog");
        const settingsCloseButton = document.querySelector("#settings-close");
        const settingsSummary = document.querySelector("#settings-summary");
        const activityOpenButton = document.querySelector(
            "#activity-open",
        );
        const activityDialog = document.querySelector(
            "#activity-dialog",
        );
        const activityCloseButton = document.querySelector(
            "#activity-close",
        );
        const activitySummary = document.querySelector(
            "#activity-summary",
        );
        const activityStatus = document.querySelector(
            "#activity-status",
        );
        const activityList = document.querySelector("#activity-list");
        const promptButtons = document.querySelectorAll("[data-prompt]");
        const knowledgeFileInput = document.querySelector(
            "#knowledge-file",
        );
        const knowledgeDialog = document.querySelector(
            "#knowledge-dialog",
        );
        const knowledgeOpenButton = document.querySelector(
            "#knowledge-open",
        );
        const memoryOpenButton = document.querySelector(
            "#memory-open",
        );
        const memoryDialog = document.querySelector(
            "#memory-dialog",
        );
        const memoryCloseButton = document.querySelector(
            "#memory-close",
        );
        const memorySearchInput = document.querySelector(
            "#memory-search",
        );
        const memorySearchButton = document.querySelector(
            "#memory-search-button",
        );
        const memoryClearButton = document.querySelector(
            "#memory-clear",
        );
        const memoryTabs = document.querySelector(
            "#memory-tabs",
        );
        const memoryStatus = document.querySelector(
            "#memory-status",
        );
        const memoryList = document.querySelector(
            "#memory-list",
        );
        const memoryPrevButton = document.querySelector(
            "#memory-prev",
        );
        const memoryNextButton = document.querySelector(
            "#memory-next",
        );
        const memoryPageIndicator = document.querySelector(
            "#memory-page-indicator",
        );
        const memoryForm = document.querySelector(
            "#memory-form",
        );
        const memoryFormTitle = document.querySelector(
            "#memory-form-title",
        );
        const memoryFormCancel = document.querySelector(
            "#memory-form-cancel",
        );
        const memoryFormSubmit = document.querySelector(
            "#memory-form-submit",
        );
        const memoryEditTitle = document.querySelector(
            "#memory-edit-title",
        );
        const memoryEditSummary = document.querySelector(
            "#memory-edit-summary",
        );
        const memoryEditContent = document.querySelector(
            "#memory-edit-content",
        );
        const memoryEditKeywords = document.querySelector(
            "#memory-edit-keywords",
        );
        const memoryReferenceFields = document.querySelector(
            "#memory-reference-fields",
        );
        const memoryReferenceDateField = document.querySelector(
            "#memory-reference-date-field",
        );
        const memoryEditUrl = document.querySelector(
            "#memory-edit-url",
        );
        const memoryEditVerifiedAt = document.querySelector(
            "#memory-edit-verified-at",
        );
        const memoryFeedbackFields = document.querySelector(
            "#memory-feedback-fields",
        );
        const memoryEditIncorrect = document.querySelector(
            "#memory-edit-incorrect",
        );
        const memoryEditCorrect = document.querySelector(
            "#memory-edit-correct",
        );
        const memoryEditAppliesWhen = document.querySelector(
            "#memory-edit-applies-when",
        );
        const knowledgeCloseButton = document.querySelector(
            "#knowledge-close",
        );
        const knowledgeRefreshButton = document.querySelector(
            "#knowledge-refresh",
        );
        const knowledgeStatus = document.querySelector(
            "#knowledge-status",
        );
        const knowledgeSummary = document.querySelector(
            "#knowledge-summary",
        );
        const knowledgeOverviewFields = {
            document_count: document.querySelector(
                "#knowledge-document-count",
            ),
            chunk_count: document.querySelector(
                "#knowledge-chunk-count",
            ),
            indexed_count: document.querySelector(
                "#knowledge-indexed-count",
            ),
            processing_count: document.querySelector(
                "#knowledge-processing-count",
            ),
            failed_count: document.querySelector(
                "#knowledge-failed-count",
            ),
            pending_count: document.querySelector(
                "#knowledge-pending-count",
            ),
        };
        const knowledgeList = document.querySelector(
            "#knowledge-list",
        );
