import json
import asyncio
import unittest

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deep_research.agent.streaming import (
    stream_research_events_with_agent,
    stream_research_with_agent,
)


class FakeSnapshot:
    values = {
        "sources": {},
    }


class FakeToken:
    def __init__(self, text: str):
        self.text = text


class FakeAgent:
    async def aget_state(self, config):
        return FakeSnapshot()

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        yield {
            "event": "on_chat_model_stream",
            "name": "BindableFakeChatModel",
            "metadata": {
                "lc_agent_name": "researcher",
                "langgraph_node": "model",
            },
            "data": {
                "chunk": FakeToken("RESEARCHER_INTERNAL_TOKEN"),
            },
        }

        yield {
            "event": "on_chat_model_stream",
            "name": "BindableFakeChatModel",
            "metadata": {
                "langgraph_node": "model",
            },
            "data": {
                "chunk": FakeToken("MAIN_FINAL_TOKEN"),
            },
        }


class ChildToolEventAgent:
    async def aget_state(self, config):
        return FakeSnapshot()

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        yield {
            "event": "on_tool_start",
            "name": "web_search",
            "metadata": {
                "lc_agent_name": "researcher",
            },
            "data": {
                "input": {
                    "query": "internal query",
                }
            },
        }

        yield {
            "event": "on_tool_end",
            "name": "web_search",
            "metadata": {
                "lc_agent_name": "researcher",
            },
            "data": {
                "output": {
                    "ok": True,
                }
            },
        }

        yield {
            "event": "on_chat_model_stream",
            "name": "BindableFakeChatModel",
            "metadata": {
                "lc_agent_name": "researcher",
            },
            "data": {
                "chunk": FakeToken("RESEARCHER_TOOL_RESULT"),
            },
        }

        yield {
            "event": "on_chat_model_stream",
            "name": "BindableFakeChatModel",
            "metadata": {
                "langgraph_node": "model",
            },
            "data": {
                "chunk": FakeToken("MAIN_RESULT"),
            },
        }


class TaskLifecycleAgent:
    def __init__(self):
        self.state_updates = []

    async def aget_state(self, config):
        return FakeSnapshot()

    async def aupdate_state(self, config, values):
        self.state_updates.append((config, values))

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        yield {
            "event": "on_tool_start",
            "name": "task",
            "run_id": "task-run-1",
            "metadata": {},
            "data": {
                "input": {
                    "description": "研究指定主题。",
                    "subagent_type": "researcher",
                }
            },
        }

        yield {
            "event": "on_tool_end",
            "name": "task",
            "run_id": "task-run-1",
            "metadata": {},
            "data": {
                "output": Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=json.dumps(
                                    {
                                        "summary": "Research completed.",
                                        "finding_ids": ["F1"],
                                        "source_ids": ["S1"],
                                        "knowledge_gaps": [],
                                        "conflicts": [],
                                        "recommended_action": "answer",
                                    }
                                ),
                                tool_call_id="task-call-1",
                            )
                        ]
                    }
                ),
            },
        }

        yield {
            "event": "on_chat_model_stream",
            "name": "BindableFakeChatModel",
            "metadata": {
                "langgraph_node": "model",
            },
            "data": {
                "chunk": FakeToken("MAIN_RESULT"),
            },
        }


class FailedTaskLifecycleAgent:
    def __init__(self):
        self.state_updates = []

    async def aget_state(self, config):
        return FakeSnapshot()

    async def aupdate_state(self, config, values):
        self.state_updates.append((config, values))

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        yield {
            "event": "on_tool_start",
            "name": "task",
            "run_id": "task-run-failed",
            "metadata": {},
            "data": {
                "input": {
                    "description": "Research a failing task.",
                    "subagent_type": "researcher",
                }
            },
        }

        yield {
            "event": "on_tool_error",
            "name": "task",
            "run_id": "task-run-failed",
            "metadata": {},
            "data": {
                "error": "api_key=secret-value must not leak",
            },
        }


class InvalidTaskResultAgent:
    def __init__(self):
        self.state_updates = []

    async def aget_state(self, config):
        return FakeSnapshot()

    async def aupdate_state(self, config, values):
        self.state_updates.append((config, values))

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        yield {
            "event": "on_tool_start",
            "name": "task",
            "run_id": "task-run-invalid",
            "metadata": {},
            "data": {
                "input": {
                    "description": "Research an invalid result.",
                    "subagent_type": "researcher",
                }
            },
        }

        yield {
            "event": "on_tool_end",
            "name": "task",
            "run_id": "task-run-invalid",
            "metadata": {},
            "data": {
                "output": Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content="not valid ResearcherResult JSON",
                                tool_call_id="task-call-invalid",
                            )
                        ]
                    }
                ),
            },
        }


class ParallelLimitTaskAgent:
    def __init__(self):
        self.state_updates = []

    async def aget_state(self, config):
        return FakeSnapshot()

    async def aupdate_state(self, config, values):
        self.state_updates.append((config, values))

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        for task_number in range(1, 4):
            task_id = f"task-run-{task_number}"
            yield {
                "event": "on_tool_start",
                "name": "task",
                "run_id": task_id,
                "metadata": {},
                "data": {
                    "input": {
                        "description": f"Research task {task_number}.",
                        "subagent_type": "researcher",
                    }
                },
            }

        for task_number in (1, 2):
            task_id = f"task-run-{task_number}"
            yield {
                "event": "on_tool_end",
                "name": "task",
                "run_id": task_id,
                "metadata": {},
                "data": {
                    "output": Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    content=json.dumps(
                                        {
                                            "summary": (
                                                f"Research task {task_number} completed."
                                            ),
                                            "finding_ids": [],
                                            "source_ids": [],
                                            "knowledge_gaps": [],
                                            "conflicts": [],
                                            "recommended_action": "answer",
                                        }
                                    ),
                                    tool_call_id=f"task-call-{task_number}",
                                )
                            ]
                        }
                    )
                },
            }

        yield {
            "event": "on_tool_end",
            "name": "task",
            "run_id": "task-run-3",
            "metadata": {},
            "data": {
                "output": ToolMessage(
                    content=json.dumps(
                        {
                            "ok": False,
                            "error": "parallel_task_limit_reached",
                            "message": "provider-secret-must-not-leak",
                        }
                    ),
                    name="task",
                    tool_call_id="task-call-3",
                    status="error",
                )
            },
        }

        yield {
            "event": "on_chat_model_stream",
            "name": "BindableFakeChatModel",
            "metadata": {
                "langgraph_node": "model",
            },
            "data": {
                "chunk": FakeToken("MAIN_RESULT"),
            },
        }


class CancellableTaskAgent:
    def __init__(self, task_count=1, complete_first=False):
        self.task_count = task_count
        self.complete_first = complete_first
        self.started = asyncio.Event()
        self.first_completed = asyncio.Event()
        self.release = asyncio.Event()
        self.state_updates = []
        self.research_context = None

    async def aget_state(self, config):
        return FakeSnapshot()

    async def aupdate_state(self, config, values):
        self.state_updates.append((config, values))

    def _task_start(self, task_number):
        return {
            "event": "on_tool_start",
            "name": "task",
            "run_id": f"task-run-{task_number}",
            "metadata": {},
            "data": {
                "input": {
                    "description": f"Research task {task_number}.",
                    "subagent_type": "researcher",
                }
            },
        }

    def _task_completion(self, task_number):
        return {
            "event": "on_tool_end",
            "name": "task",
            "run_id": f"task-run-{task_number}",
            "metadata": {},
            "data": {
                "output": Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=json.dumps(
                                    {
                                        "summary": "Research completed.",
                                        "finding_ids": [],
                                        "source_ids": [],
                                        "knowledge_gaps": [],
                                        "conflicts": [],
                                        "recommended_action": "answer",
                                    }
                                ),
                                tool_call_id=f"task-call-{task_number}",
                            )
                        ]
                    }
                )
            },
        }

    async def astream_events(
        self,
        input_data,
        config,
        context,
        version,
    ):
        self.research_context = context
        context.search_count += 1
        context.page_read_count += 1
        context.register_source(
            "Committed before cancellation",
            "https://example.com/committed",
            "committed before cancellation",
        )

        for task_number in range(1, self.task_count + 1):
            yield self._task_start(task_number)

        self.started.set()

        if self.complete_first:
            yield self._task_completion(1)
            self.first_completed.set()

        await self.release.wait()


class MessageStreamAgent:
    async def aget_state(self, config):
        return FakeSnapshot()

    async def astream(
        self,
        input_data,
        config,
        context,
        stream_mode,
    ):
        yield (
            FakeToken("RESEARCHER_MESSAGE_TOKEN"),
            {
                "langgraph_node": "model",
                "lc_agent_name": "researcher",
            },
        )
        yield (
            FakeToken("MAIN_MESSAGE_TOKEN"),
            {
                "langgraph_node": "model",
            },
        )


class StreamSubAgentIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_researcher_model_tokens_are_not_frontend_text(self):
        events = [
            event
            async for event in stream_research_events_with_agent(
                FakeAgent(),
                "test question",
                "stream-isolation-thread",
            )
        ]

        text = "".join(
            event["text"]
            for event in events
            if event["type"] == "text"
        )

        self.assertEqual(text, "MAIN_FINAL_TOKEN")
        self.assertNotIn(
            "RESEARCHER_INTERNAL_TOKEN",
            text,
        )

    async def test_researcher_tool_events_are_not_frontend_events(self):
        events = [
            event
            async for event in stream_research_events_with_agent(
                ChildToolEventAgent(),
                "test question",
                "stream-tool-isolation-thread",
            )
        ]

        self.assertEqual(
            [event["type"] for event in events],
            ["text", "done"],
        )
        self.assertEqual(
            events[0]["text"],
            "MAIN_RESULT",
        )

    async def test_task_lifecycle_is_exposed_as_subagent_events(self):
        agent = TaskLifecycleAgent()
        events = [
            event
            async for event in stream_research_events_with_agent(
                agent,
                "test question",
                "task-lifecycle-thread",
            )
        ]

        self.assertEqual(
            [event["type"] for event in events],
            [
                "subagent_start",
                "subagent_end",
                "text",
                "done",
            ],
        )

        self.assertEqual(
            events[0]["agent_name"],
            "researcher",
        )
        self.assertEqual(events[0]["status"], "running")
        self.assertEqual(
            events[0]["task_id"],
            "task-run-1",
        )
        self.assertEqual(
            events[1]["task_id"],
            "task-run-1",
        )
        self.assertEqual(events[1]["status"], "completed")
        self.assertEqual(events[1]["finding_ids"], ["F1"])
        self.assertEqual(events[1]["source_ids"], ["S1"])
        self.assertEqual(events[1]["error"], "")

        self.assertEqual(len(agent.state_updates), 1)
        task = agent.state_updates[0][1]["tasks"]["task-run-1"]
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["finding_ids"], ["F1"])
        self.assertEqual(task["source_ids"], ["S1"])
        self.assertEqual(task["error"], "")

    async def test_failed_task_emits_safe_terminal_event(self):
        agent = FailedTaskLifecycleAgent()
        events = [
            event
            async for event in stream_research_events_with_agent(
                agent,
                "test question",
                "failed-task-thread",
            )
        ]

        self.assertEqual(
            [event["type"] for event in events],
            ["subagent_start", "subagent_end", "done"],
        )
        self.assertEqual(events[1]["task_id"], "task-run-failed")
        self.assertEqual(events[1]["status"], "failed")
        self.assertEqual(events[1]["finding_ids"], [])
        self.assertEqual(events[1]["source_ids"], [])
        self.assertEqual(events[1]["error"], "task_execution_failed")
        self.assertNotIn("secret-value", events[1]["error"])

        self.assertEqual(len(agent.state_updates), 1)
        task = agent.state_updates[0][1]["tasks"]["task-run-failed"]
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["finding_ids"], [])
        self.assertEqual(task["source_ids"], [])
        self.assertEqual(task["error"], "task_execution_failed")

    async def test_invalid_task_result_emits_safe_failed_event(self):
        agent = InvalidTaskResultAgent()
        events = [
            event
            async for event in stream_research_events_with_agent(
                agent,
                "test question",
                "invalid-task-result-thread",
            )
        ]

        self.assertEqual(events[1]["task_id"], "task-run-invalid")
        self.assertEqual(events[1]["status"], "failed")
        self.assertEqual(events[1]["finding_ids"], [])
        self.assertEqual(events[1]["source_ids"], [])
        self.assertEqual(
            events[1]["error"],
            "invalid_researcher_result",
        )

        self.assertEqual(len(agent.state_updates), 1)
        task = agent.state_updates[0][1]["tasks"]["task-run-invalid"]
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["error"], "invalid_researcher_result")

    async def test_parallel_limit_error_is_preserved_end_to_end(self):
        agent = ParallelLimitTaskAgent()
        events = [
            event
            async for event in stream_research_events_with_agent(
                agent,
                "run three parallel tasks",
                "parallel-limit-thread",
            )
        ]

        terminal_events = {
            event["task_id"]: event
            for event in events
            if event["type"] == "subagent_end"
        }

        self.assertEqual(
            terminal_events["task-run-1"]["status"],
            "completed",
        )
        self.assertEqual(
            terminal_events["task-run-2"]["status"],
            "completed",
        )
        self.assertEqual(
            terminal_events["task-run-3"]["status"],
            "failed",
        )
        self.assertEqual(
            terminal_events["task-run-3"]["error"],
            "parallel_task_limit_reached",
        )
        self.assertNotIn(
            "invalid_researcher_result",
            repr(terminal_events["task-run-3"]),
        )
        self.assertNotIn("provider-secret-must-not-leak", repr(events))

        self.assertEqual(len(agent.state_updates), 1)
        tasks = agent.state_updates[0][1]["tasks"]
        self.assertEqual(tasks["task-run-1"]["status"], "completed")
        self.assertEqual(tasks["task-run-2"]["status"], "completed")
        self.assertEqual(tasks["task-run-3"]["status"], "failed")
        self.assertEqual(
            tasks["task-run-3"]["error"],
            "parallel_task_limit_reached",
        )

    async def test_active_task_is_persisted_as_cancelled(self):
        agent = CancellableTaskAgent()
        seen_events = []

        async def consume():
            async for event in stream_research_events_with_agent(
                agent,
                "cancel one task",
                "cancel-one-task-thread",
            ):
                seen_events.append(event)

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(agent.started.wait(), timeout=1)
        consumer.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await consumer

        self.assertFalse(
            any(event["type"] == "done" for event in seen_events)
        )
        self.assertEqual(len(agent.state_updates), 1)
        task = agent.state_updates[0][1]["tasks"]["task-run-1"]
        self.assertEqual(task["status"], "cancelled")
        self.assertEqual(task["error"], "user_cancelled")
        self.assertEqual(agent.research_context.search_count, 1)
        self.assertEqual(agent.research_context.page_read_count, 1)
        self.assertEqual(
            set(agent.research_context.new_sources),
            {"S1"},
        )

    async def test_two_active_tasks_are_cancelled_together(self):
        agent = CancellableTaskAgent(task_count=2)

        async def consume():
            async for _ in stream_research_events_with_agent(
                agent,
                "cancel two tasks",
                "cancel-two-task-thread",
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(agent.started.wait(), timeout=1)
        consumer.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await consumer

        tasks = agent.state_updates[0][1]["tasks"]
        self.assertEqual(
            {tasks["task-run-1"]["status"], tasks["task-run-2"]["status"]},
            {"cancelled"},
        )
        self.assertEqual(tasks["task-run-1"]["error"], "user_cancelled")
        self.assertEqual(tasks["task-run-2"]["error"], "user_cancelled")

    async def test_completed_task_stays_completed_when_another_is_cancelled(self):
        agent = CancellableTaskAgent(task_count=2, complete_first=True)

        async def consume():
            async for _ in stream_research_events_with_agent(
                agent,
                "complete one and cancel one",
                "complete-cancel-thread",
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(agent.first_completed.wait(), timeout=1)
        consumer.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await consumer

        tasks = agent.state_updates[0][1]["tasks"]
        self.assertEqual(tasks["task-run-1"]["status"], "completed")
        self.assertEqual(tasks["task-run-1"]["error"], "")
        self.assertEqual(tasks["task-run-2"]["status"], "cancelled")
        self.assertEqual(tasks["task-run-2"]["error"], "user_cancelled")

    async def test_message_stream_hides_researcher_tokens(self):
        chunks = [
            chunk
            async for chunk in stream_research_with_agent(
                MessageStreamAgent(),
                "test question",
                "message-stream-isolation-thread",
            )
        ]

        self.assertEqual(
            chunks,
            ["MAIN_MESSAGE_TOKEN"],
        )


if __name__ == "__main__":
    unittest.main()
