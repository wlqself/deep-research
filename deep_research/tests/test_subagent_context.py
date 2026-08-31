from typing import Any

import unittest
from typing_extensions import NotRequired
from unittest.mock import patch

import deep_research.agent.factory as factory_module
import deep_research.agent.sub_agent as sub_agent_module
from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_research.context import ResearchContext
from deep_research.state import ResearchState


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ProbeState(ResearchState, total=False):
    todos: list[dict[str, str]]
    files: dict[str, Any]


class SubAgentContextTests(unittest.TestCase):
    def test_task_propagates_state_and_runtime_context(self):
        observed: dict[str, Any] = {}

        @tool
        def probe_research_runtime(
            runtime: ToolRuntime[ResearchContext],
        ) -> Command:
            """Inspect the Researcher runtime and return test state updates."""

            observed["state"] = runtime.state
            observed["context"] = runtime.context

            runtime.context.search_count += 1
            runtime.context.page_read_count += 1

            source = runtime.context.register_source(
                title="Child source",
                url="https://example.com/child",
                snippet="source created by researcher",
            )

            child_finding = {
                "finding_id": "F1",
                "claim": "child claim",
                "evidence_summary": "child evidence",
                "source_ids": [source["source_id"]],
                "status": "supported",
                "uncertainty": "",
                "conflicts": "",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }

            return Command(
                update={
                    "sources": {
                        source["source_id"]: source,
                    },
                    "next_source_number": runtime.context.next_source_number,
                    "findings": {
                        "F1": child_finding,
                    },
                    # 这两个字段应该被 private_state_keys 过滤掉
                    "files": {
                        "child-secret.txt": "must not return",
                    },
                    "artifacts": {
                        "child-artifact": {
                            "artifact_id": "child-artifact",
                        },
                    },
                    "messages": [
                        ToolMessage(
                            content="probe completed",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        supervisor_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": (
                                        "研究 child context propagation，"
                                        "只执行 probe 工具并返回结果。"
                                    ),
                                    "subagent_type": "researcher",
                                },
                                "id": "task-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="supervisor final answer",
                    ),
                ]
            )
        )

        researcher_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "probe_research_runtime",
                                "args": {},
                                "id": "probe-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="researcher internal final",
                    ),
                ]
            )
        )

        researcher_graph = create_agent(
            model=researcher_model,
            tools=[probe_research_runtime],
            system_prompt=(
                "只执行委派任务。调用 probe_research_runtime，"
                "然后返回简短结果。"
            ),
            state_schema=ProbeState,
            context_schema=ResearchContext,
        )
        researcher = {
            "name": "researcher",
            "description": "Runs the delegated research task.",
            "runnable": researcher_graph,
        }

        subagent_middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[researcher],
            private_state_keys=frozenset(
                {
                    "files",
                    "artifacts",
                }
            ),
            state_schema=ProbeState,
        )

        graph = create_agent(
            model=supervisor_model,
            tools=[],
            middleware=[subagent_middleware],
            state_schema=ProbeState,
            context_schema=ResearchContext,
            checkpointer=InMemorySaver(),
        )

        context = ResearchContext.from_settings(
            next_source_number=2,
        )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": "PARENT_SECRET",
                }
            ],
            "todos": [
                {
                    "content": "parent todo",
                    "status": "in_progress",
                }
            ],
            "files": {
                "parent.txt": "parent file",
            },
            "artifacts": {
                "parent-artifact": {
                    "artifact_id": "parent-artifact",
                }
            },
            "sources": {
                "S1": {
                    "source_id": "S1",
                    "title": "Parent source",
                    "url": "https://example.com/parent",
                    "snippet": "parent source",
                }
            },
            "next_source_number": 2,
            "findings": {
                "F0": {
                    "finding_id": "F0",
                    "claim": "parent claim",
                    "evidence_summary": "parent evidence",
                    "source_ids": ["S1"],
                    "status": "supported",
                    "uncertainty": "",
                    "conflicts": "",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            },
        }

        result = graph.invoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": "subagent-context-test",
                }
            },
            context=context,
        )

        child_state = observed["state"]
        child_context = observed["context"]

        # 1. Researcher 没有收到父 messages，只收到 task description
        child_messages_text = repr(child_state["messages"])

        self.assertIn("研究 child context propagation", child_messages_text)
        self.assertNotIn("PARENT_SECRET", child_messages_text)

        # 2. todos、files、artifacts 不进入 Researcher
        self.assertNotIn("todos", child_state)
        child_files = child_state.get("files") or {}
        child_artifacts = child_state.get("artifacts") or {}

        self.assertNotIn("parent.txt", child_files)
        self.assertNotIn("parent-artifact", child_artifacts)

        self.assertNotIn(
            "child-secret.txt",
            child_files,
        )
        self.assertNotIn(
            "child-artifact",
            child_artifacts,
        )
        # 3. 普通结构化 state 会进入 Researcher
        self.assertIn("sources", child_state)
        self.assertIn("next_source_number", child_state)
        self.assertIn("findings", child_state)

        self.assertEqual(child_state["next_source_number"], 2)
        self.assertIn("S1", child_state["sources"])
        self.assertIn("F0", child_state["findings"])

        # 4. Researcher runtime.context 是当前 ResearchContext
        self.assertIs(child_context, context)
        self.assertEqual(context.search_count, 1)
        self.assertEqual(context.page_read_count, 1)

        # 5. 子 Agent 新增 source 和 finding 成功返回 Main
        self.assertIn("S2", result["sources"])
        self.assertIn("F1", result["findings"])
        self.assertEqual(result["next_source_number"], 3)

        # 6. private state 不会被子 Agent 返回结果覆盖
        self.assertEqual(
            result["files"],
            initial_state["files"],
        )
        self.assertEqual(
            result["artifacts"],
            initial_state["artifacts"],
        )

        # 7. Main 只收到一个 task ToolMessage
        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]

        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(
            tool_messages[0].content,
            "researcher internal final",
        )

        # 8. Researcher 内部 probe ToolMessage 没有进入 Main
        self.assertNotIn(
            "probe completed",
            repr(result["messages"]),
        )

        # 9. Main 最终回答来自 Supervisor
        self.assertEqual(
            result["messages"][-1].content,
            "supervisor final answer",
        )

    def test_factory_hides_tasks_and_blocks_child_task_updates(self):
        observed: dict[str, Any] = {}

        @tool
        def probe_tasks(runtime: ToolRuntime[ResearchContext]) -> Command:
            """Inspect child state and attempt to modify the parent task ledger."""

            observed["state"] = dict(runtime.state)

            return Command(
                update={
                    "tasks": {
                        "parent-task": {
                            "task_id": "parent-task",
                            "agent_name": "researcher",
                            "description": "spoofed by researcher",
                            "status": "completed",
                            "created_at": "2026-01-01T00:00:00Z",
                            "started_at": "2026-01-01T00:00:01Z",
                            "completed_at": "2099-01-01T00:00:00Z",
                            "updated_at": "2099-01-01T00:00:00Z",
                            "finding_ids": [],
                            "source_ids": [],
                            "error": "",
                        }
                    },
                    "messages": [
                        ToolMessage(
                            content="probe completed",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        parent_task = {
            "task_id": "parent-task",
            "agent_name": "researcher",
            "description": "original parent task",
            "status": "completed",
            "created_at": "2026-01-01T00:00:00Z",
            "started_at": "2026-01-01T00:00:01Z",
            "completed_at": "2026-01-01T00:01:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
            "finding_ids": [],
            "source_ids": [],
            "error": "",
        }

        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "执行状态隔离 probe。",
                                    "subagent_type": "researcher",
                                },
                                "id": "factory-task-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "probe_tasks",
                                "args": {},
                                "id": "probe-tasks-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ResearcherResult",
                                "args": {
                                    "summary": "Research completed.",
                                    "finding_ids": [],
                                    "source_ids": [],
                                    "knowledge_gaps": [],
                                    "conflicts": [],
                                    "recommended_action": "answer",
                                },
                                "id": "researcher-result-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Supervisor final answer"),
                ]
            )
        )

        with patch.object(factory_module, "model", fake_model):
            with patch.object(
                sub_agent_module,
                "build_researcher_tools",
                return_value=[probe_tasks],
            ):
                graph = factory_module.build_agent(InMemorySaver())

        result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "验证 Researcher state 隔离。",
                    }
                ],
                "tasks": {
                    "parent-task": parent_task,
                },
            },
            config={
                "configurable": {
                    "thread_id": "factory-subagent-tasks-private",
                }
            },
            context=ResearchContext.from_settings(),
        )

        child_tasks = observed["state"].get("tasks") or {}
        self.assertNotIn("parent-task", child_tasks)
        self.assertEqual(result["tasks"], {"parent-task": parent_task})


if __name__ == "__main__":
    unittest.main()
