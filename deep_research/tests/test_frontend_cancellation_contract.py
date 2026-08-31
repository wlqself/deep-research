import re
import subprocess
import unittest
from pathlib import Path


STATIC_INDEX = (
    Path(__file__).resolve().parents[1] / "static" / "index.html"
)


def inline_script() -> str:
    html = STATIC_INDEX.read_text(encoding="utf-8")
    match = re.search(
        r"<script>(?P<script>[\s\S]*)</script>",
        html,
    )
    if match is None:
        raise AssertionError("index.html has no inline script")
    return match.group("script")


class FrontendCancellationContractTests(unittest.TestCase):
    def test_inline_script_has_valid_javascript_syntax(self):
        result = subprocess.run(
            ["node", "--check", "-"],
            input=inline_script(),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cancelled_state_contract_is_present(self):
        script = inline_script()

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
        script = inline_script()

        self.assertIn("researchSucceeded", script)
        self.assertIn("fullText.trim()", script)
        self.assertIn("persistSuccessfulThread(question)", script)
        self.assertIn(
            'if (cancellationRequested) {',
            script,
        )

    def test_memory_management_surface_and_api_contract_are_present(self):
        html = STATIC_INDEX.read_text(encoding="utf-8")
        script = inline_script()

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


if __name__ == "__main__":
    unittest.main()
