import subprocess
import unittest
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
STATIC_INDEX = STATIC_DIR / "index.html"
FRONTEND_SCRIPTS = (
    STATIC_DIR / "js" / "dom.js",
    STATIC_DIR / "js" / "memory.js",
    STATIC_DIR / "js" / "knowledge.js",
    STATIC_DIR / "js" / "threads.js",
    STATIC_DIR / "js" / "thread-view.js",
    STATIC_DIR / "js" / "activity.js",
    STATIC_DIR / "js" / "research-ui.js",
    STATIC_DIR / "js" / "research-stream.js",
    STATIC_DIR / "js" / "publishing.js",
)


def frontend_script() -> str:
    return "\n\n".join(
        path.read_text(encoding="utf-8")
        for path in FRONTEND_SCRIPTS
    )


class FrontendCancellationContractTests(unittest.TestCase):
    def test_index_references_all_frontend_assets(self):
        html = STATIC_INDEX.read_text(encoding="utf-8")

        self.assertIn('href="/static/styles.css"', html)
        for path in FRONTEND_SCRIPTS:
            self.assertIn(
                f'src="/static/{path.relative_to(STATIC_DIR).as_posix()}"',
                html,
            )

    def test_frontend_scripts_have_valid_javascript_syntax(self):
        result = subprocess.run(
            ["node", "--check", "-"],
        input=frontend_script(),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_recent_threads_navigation_is_removed(self):
        html = STATIC_INDEX.read_text(encoding="utf-8")
        script = frontend_script()
        styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
        self.assertNotIn('id="recent-threads-section"', html)
        self.assertNotIn('id="recent-threads-toggle"', html)
        self.assertNotIn("setRecentThreadsExpanded", script)
        self.assertNotIn("RECENT_THREADS_COLLAPSED_KEY", script)
        self.assertNotIn(".recent-threads-section", styles)
        self.assertIn('id="new-thread"', html)
        self.assertIn('id="conversation-toggle"', html)
        self.assertIn('id="conversation-panel"', html)
        self.assertIn('id="thread-list"', html)
        self.assertIn("setConversationPanelExpanded", script)
        self.assertIn("CONVERSATION_PANEL_KEY", script)
        self.assertIn(".conversation-section", styles)

    def test_cancelled_state_contract_is_present(self):
        script = frontend_script()

        self.assertIn("function markRunningTasksCancelled()", script)
        self.assertIn('"cancelled"', script)
        self.assertIn("let cancellationRequested = false", script)
        self.assertIn(
            'cancellationRequested && incomingStatus === "running"',
            script,
        )
        self.assertIn('status.textContent = "已取消"', script)
        self.assertIn('traceStatus.textContent = "已取消"', script)
        self.assertIn('"研究服务失败"', script)

    def test_success_persistence_requires_done_and_nonempty_answer(self):
        script = frontend_script()

        self.assertIn("researchSucceeded", script)
        self.assertIn("fullText.trim()", script)
        self.assertIn("persistSuccessfulThread(question)", script)
        self.assertIn(
            'if (cancellationRequested) {',
            script,
        )
        self.assertIn('event.goal_status || "completed"', script)
        self.assertIn('goalStatus === "completed"', script)
        self.assertIn("model_repetition_detected", script)
        self.assertIn("non_retryable_tool_loop_detected", script)
        self.assertIn("publication_repair_exhausted", script)
        self.assertIn("xiaohongshu_title_too_long", script)

    def test_memory_management_surface_and_api_contract_are_present(self):
        html = STATIC_INDEX.read_text(encoding="utf-8")
        script = frontend_script()

        for value in (
            'id="memory-open"',
            'id="memory-dialog"',
            'data-kind="user"',
            'data-kind="reference"',
            'data-kind="project"',
            'data-kind="feedback"',
            'id="memory-search"',
            'id="memory-form"',
        ):
            self.assertIn(value, html)

        for value in (
            "fetch(`/memory?",
            'method: "PUT"',
            'method: "DELETE"',
            "function openMemoryEditor(entry)",
            "function clearMemories()",
        ):
            self.assertIn(value, script)

        self.assertNotIn("source_thread_id", script)

    def test_activity_log_surface_and_api_contract_are_present(self):
        html = STATIC_INDEX.read_text(encoding="utf-8")
        script = frontend_script()

        for value in (
            'id="activity-open"',
            'id="activity-dialog"',
            'id="activity-list"',
            'id="publishing-open"',
            'id="publishing-dialog"',
            'id="publishing-article-list"',
            'id="publishing-form"',
            'id="publishing-publish-button"',
        ):
            self.assertIn(value, html)

        for value in (
            "function setActivityEvents(events)",
            "function recordQuestionActivity(question)",
            "function recordActivityFromStreamEvent(event)",
            "开始整理文章草稿",
            "文章草稿整理完成",
            "等待用户人工审批",
        ):
            self.assertIn(value, script)

    def test_publishing_frontend_uses_management_api_contract(self):
        html = STATIC_INDEX.read_text(encoding="utf-8")
        script = frontend_script()

        for value in (
            "/publishing/articles",
            "/publishing/articles/from-artifact",
            "/publishing/articles/${encodeURIComponent(article.article_id)}/approve",
            "/publishing/articles/${encodeURIComponent(article.article_id)}/publish",
            "local_static_site",
            "window.confirm",
            "DOMPurify",
        ):
            self.assertIn(value, html + script)


if __name__ == "__main__":
    unittest.main()
