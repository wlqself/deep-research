import unittest

from deep_research.activity import (
    activity_from_stream_event,
    build_activity_event,
    merge_activity_events,
)


class ActivityEventTests(unittest.TestCase):
    def test_stream_events_are_projected_to_user_facing_activity(self):
        activity = activity_from_stream_event(
            {
                "type": "subagent_end",
                "agent_name": "researcher",
                "status": "completed",
                "source_ids": ["S1", "S2"],
            },
            run_id="run-1",
        )

        self.assertIsNotNone(activity)
        self.assertEqual(activity["kind"], "agent")
        self.assertEqual(activity["status"], "completed")
        self.assertEqual(activity["summary"], "关联 2 个来源")
        self.assertEqual(activity["detail"], "关联 2 个来源")
        self.assertEqual(activity["run_id"], "run-1")

    def test_text_chunks_are_not_recorded_as_activity(self):
        activity = activity_from_stream_event(
            {"type": "text", "text": "answer"},
            run_id="run-1",
        )

        self.assertIsNone(activity)

    def test_publishing_tool_uses_safe_user_facing_activity(self):
        started = activity_from_stream_event(
            {
                "type": "tool_start",
                "name": "prepare_article_for_publication",
            },
            run_id="run-1",
        )
        completed = activity_from_stream_event(
            {
                "type": "tool_end",
                "name": "prepare_article_for_publication",
                "status": "completed",
            },
            run_id="run-1",
        )
        failed = activity_from_stream_event(
            {
                "type": "tool_end",
                "name": "prepare_article_for_publication",
                "status": "failed",
            },
            run_id="run-1",
        )

        self.assertEqual(started["kind"], "publishing")
        self.assertEqual(started["status"], "running")
        self.assertEqual(started["label"], "开始整理文章草稿")
        self.assertEqual(completed["label"], "文章草稿整理完成")
        self.assertEqual(
            completed["detail"],
            "草稿已准备完成，等待用户人工审批",
        )
        self.assertEqual(failed["label"], "文章草稿整理失败")
        self.assertNotIn("article_id", completed["detail"])

    def test_activity_merge_deduplicates_and_keeps_latest_events(self):
        first = build_activity_event(
            run_id="run-1",
            kind="research",
            actor="main",
            status="running",
            label="开始处理问题",
        )
        second = build_activity_event(
            run_id="run-1",
            kind="research",
            actor="main",
            status="completed",
            label="研究完成",
        )

        merged = merge_activity_events(
            [first],
            [first, second],
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            {event["id"] for event in merged},
            {first["id"], second["id"]},
        )

    def test_done_distinguishes_success_failure_and_waiting(self):
        completed = activity_from_stream_event(
            {"type": "done", "goal_status": "completed"},
            run_id="run-1",
        )
        failed = activity_from_stream_event(
            {
                "type": "done",
                "goal_status": "completed_with_failure",
                "error_codes": ["invalid_slug"],
            },
            run_id="run-2",
        )
        waiting = activity_from_stream_event(
            {"type": "done", "goal_status": "waiting_for_user"},
            run_id="run-3",
        )

        self.assertEqual(completed["label"], "处理完成")
        self.assertEqual(failed["label"], "处理结束，但目标未完成")
        self.assertEqual(failed["detail"], "invalid_slug")
        self.assertEqual(waiting["label"], "等待用户确认")


if __name__ == "__main__":
    unittest.main()
