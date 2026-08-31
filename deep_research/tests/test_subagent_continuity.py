import json
import unittest

from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_research.agent.results import ResearcherResult
from deep_research.context import ResearchContext
from deep_research.state import ResearchState


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def task_call(task_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {
                    "description": (
                        f"执行 {task_id}，研究指定问题并返回结果。"
                    ),
                    "subagent_type": "researcher",
                },
                "id": f"task-call-{task_id}",
                "type": "tool_call",
            }
        ],
    )


def probe_call(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "probe_research_context",
                "args": {},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def result_call(task_id: str, source_ids: list[str]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ResearcherResult",
                "args": {
                    "summary": f"完成 {task_id}。",
                    "finding_ids": [],
                    "source_ids": source_ids,
                    "knowledge_gaps": [],
                    "conflicts": [],
                    "recommended_action": "answer",
                },
                "id": f"result-call-{task_id}",
                "type": "tool_call",
            }
        ],
    )


def build_probe_graph(model, observations):
    @tool
    def probe_research_context(
        runtime: ToolRuntime[ResearchContext],
    ) -> Command:
        """Consume one search budget unit and register one source."""

        context = runtime.context
        observations.append(
            {
                "search_count_before": context.search_count,
                "next_source_number_before": context.next_source_number,
            }
        )

        if context.search_count >= context.max_search_calls:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=json.dumps({"ok": False}),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        context.search_count += 1
        source = context.register_source(
            title=f"Source {context.next_source_number}",
            url=f"https://example.com/{context.next_source_number}",
            snippet="continuity source",
        )
        finding_id = f"F{source['source_id'][1:]}"

        finding = {
            "finding_id": finding_id,
            "claim": f"Claim from {source['source_id']}",
            "evidence_summary": "continuity evidence",
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
                "next_source_number": context.next_source_number,
                "findings": {
                    finding_id: finding,
                },
                "messages": [
                    ToolMessage(
                        content=json.dumps({"ok": True}),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    researcher = {
        "name": "researcher",
        "description": "Runs a delegated research task.",
        "system_prompt": "Use the probe tool and return ResearcherResult.",
        "model": model,
        "tools": [probe_research_context],
        "response_format": ToolStrategy(ResearcherResult),
    }

    return create_agent(
        model=model,
        tools=[],
        middleware=[
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[researcher],
                state_schema=ResearchState,
            )
        ],
        context_schema=ResearchContext,
        state_schema=ResearchState,
        checkpointer=InMemorySaver(),
    )


class SubAgentContinuityTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequential_delegation_continues_source_ids_and_budget(self):
        model = BindableFakeChatModel(
            messages=iter(
                [
                    task_call("task-1"),
                    probe_call("probe-1"),
                    result_call("task-1", ["S1"]),
                    task_call("task-2"),
                    probe_call("probe-2"),
                    result_call("task-2", ["S1"]),
                    AIMessage(content="supervisor final"),
                ]
            )
        )
        observations: list[dict[str, int]] = []
        graph = build_probe_graph(model, observations)

        result = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "执行连续研究任务。",
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "continuity-same-turn-thread",
                }
            },
            context=ResearchContext(
                max_search_calls=1,
                max_page_reads=2,
                max_page_chars=1000,
            ),
        )

        self.assertEqual(observations[0]["next_source_number_before"], 1)
        self.assertEqual(observations[1]["next_source_number_before"], 2)
        self.assertEqual(observations[0]["search_count_before"], 0)
        self.assertEqual(observations[1]["search_count_before"], 1)

        self.assertEqual(set(result["sources"]), {"S1"})
        self.assertEqual(set(result["findings"]), {"F1"})
        self.assertEqual(result["next_source_number"], 2)

    async def test_checkpoint_source_number_continues_on_next_turn(self):
        model = BindableFakeChatModel(
            messages=iter(
                [
                    task_call("turn-1"),
                    probe_call("probe-turn-1"),
                    result_call("turn-1", ["S1"]),
                    AIMessage(content="turn one done"),
                    task_call("turn-2"),
                    probe_call("probe-turn-2"),
                    result_call("turn-2", ["S2"]),
                    AIMessage(content="turn two done"),
                ]
            )
        )
        observations: list[dict[str, int]] = []
        graph = build_probe_graph(model, observations)
        config = {
            "configurable": {
                "thread_id": "continuity-thread",
            }
        }

        first_context = ResearchContext.from_settings(
            next_source_number=1,
        )
        first = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "完成第一轮研究。",
                    }
                ]
            },
            config=config,
            context=first_context,
        )

        second_context = ResearchContext.from_settings(
            next_source_number=first["next_source_number"],
        )
        second = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "继续第二轮研究。",
                    }
                ]
            },
            config=config,
            context=second_context,
        )

        self.assertEqual(set(first["sources"]), {"S1"})
        self.assertEqual(set(second["sources"]), {"S1", "S2"})
        self.assertEqual(set(second["findings"]), {"F1", "F2"})
        self.assertEqual(second["next_source_number"], 3)
        self.assertEqual(
            observations[1]["next_source_number_before"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
