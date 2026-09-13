import unittest
from types import SimpleNamespace

from deep_research.agent.hitl_resume import resume_agent_after_hitl


class FakeAgent:
    def __init__(self):
        self.resumed_input = None

    async def aget_state(self, config):
        return SimpleNamespace(
            values={},
            interrupts=(
                SimpleNamespace(
                    value={
                        "kind": "publication_approval",
                        "interaction_id": "interaction-1",
                    },
                ),
            ),
        )

    async def ainvoke(self, value, *, config, context):
        self.resumed_input = value
        return {
            "messages": [
                SimpleNamespace(content="审批后的 Graph 已继续执行。"),
            ],
        }


class HITLResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_uses_command_and_returns_safe_answer(self):
        agent = FakeAgent()

        result = await resume_agent_after_hitl(
            agent,
            thread_id="thread-1",
            interaction_id="interaction-1",
            decision="approved",
        )

        self.assertTrue(result["resumed"])
        self.assertEqual(
            result["answer"],
            "审批后的 Graph 已继续执行。",
        )
        self.assertEqual(
            agent.resumed_input.resume,
            {
                "interaction_id": "interaction-1",
                "decision": "approved",
            },
        )
        self.assertEqual(result["next_intent"], None)

    async def test_missing_interrupt_is_not_resumed(self):
        agent = FakeAgent()

        async def no_interrupt(config):
            return SimpleNamespace(values={}, interrupts=())

        agent.aget_state = no_interrupt
        result = await resume_agent_after_hitl(
            agent,
            thread_id="thread-1",
            interaction_id="missing",
            decision="approved",
        )

        self.assertFalse(result["resumed"])
        self.assertEqual(result["error_code"], "hitl_graph_not_interrupted")

    async def test_resume_finds_task_level_interrupts_and_non_last_answer(self):
        agent = FakeAgent()

        async def task_level_state(config):
            return SimpleNamespace(
                values={},
                interrupts=(),
                tasks=(
                    SimpleNamespace(
                        interrupts=(
                            SimpleNamespace(
                                value={
                                    "kind": "publication_approval",
                                    "interaction_id": "interaction-1",
                                },
                            ),
                        ),
                    ),
                ),
            )

        async def result_with_empty_last_message(value, *, config, context):
            return {
                "messages": [
                    SimpleNamespace(content="恢复后的模型反馈。"),
                    SimpleNamespace(content=""),
                ],
            }

        agent.aget_state = task_level_state
        agent.ainvoke = result_with_empty_last_message

        result = await resume_agent_after_hitl(
            agent,
            thread_id="thread-1",
            interaction_id="interaction-1",
            decision="approved",
        )

        self.assertTrue(result["resumed"])
        self.assertEqual(result["answer"], "恢复后的模型反馈。")

    async def test_approved_publication_resume_returns_publish_confirmation(self):
        agent = FakeAgent()

        async def publication_state(config):
            return SimpleNamespace(
                values={},
                interrupts=(
                    SimpleNamespace(
                        value={
                            "kind": "publication_approval",
                            "interaction_id": "interaction-1",
                            "attachment_ids": ["image-1"],
                            "target": {
                                "approval_id": "approval-1",
                                "article_id": "article-1",
                                "article_title": "示例文章",
                                "article_slug": "example",
                                "article_version": 1,
                                "channel": "xiaohongshu",
                                "approval_status": "pending",
                            },
                        },
                    ),
                ),
            )

        async def result_with_feedback(value, *, config, context):
            return {"messages": [SimpleNamespace(content="已继续处理。")]}

        agent.aget_state = publication_state
        agent.ainvoke = result_with_feedback
        result = await resume_agent_after_hitl(
            agent,
            thread_id="thread-1",
            interaction_id="interaction-1",
            decision="approved",
        )

        self.assertEqual(result["next_intent"]["action"], "resume")
        self.assertTrue(result["next_intent"]["requires_user_confirmation"])
        self.assertEqual(
            result["next_intent"]["target"]["approval_status"],
            "approved",
        )


if __name__ == "__main__":
    unittest.main()
