import asyncio
import json
import unittest

from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from deep_research.middleware import (
    ResearchTaskConcurrencyMiddleware,
)
from deep_research.context import ResearchContext
from deep_research.state import ResearchState


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def task_call(task_id: str) -> dict[str, object]:
    return {
        "name": "task",
        "args": {
            "description": f"Research {task_id} independently.",
            "subagent_type": "researcher",
        },
        "id": task_id,
        "type": "tool_call",
    }


def build_task_graph(model, async_researcher):
    researcher = {
        "name": "researcher",
        "description": "Runs one bounded research task.",
        "runnable": RunnableLambda(
            lambda state: {
                "messages": [AIMessage(content="sync researcher result")]
            },
            afunc=async_researcher,
        ),
    }

    return create_agent(
        model=model,
        tools=[],
        middleware=[
            ResearchTaskConcurrencyMiddleware(),
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[researcher],
                state_schema=ResearchState,
            ),
        ],
        context_schema=ResearchContext,
        state_schema=ResearchState,
    )


def make_context() -> ResearchContext:
    return ResearchContext(
        max_search_calls=4,
        max_page_reads=6,
        max_page_chars=1000,
        max_parallel_research_tasks=2,
    )


@tool
def echo_tool(value: str) -> str:
    """Return the supplied value."""
    return value


class TaskConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_tasks_overlap_and_third_is_rejected(self):
        active_count = 0
        max_active_count = 0
        two_tasks_started = asyncio.Event()
        release_tasks = asyncio.Event()

        async def run_researcher(state):
            nonlocal active_count, max_active_count

            active_count += 1
            max_active_count = max(max_active_count, active_count)

            if active_count == 2:
                two_tasks_started.set()

            try:
                await release_tasks.wait()
                return {
                    "messages": [
                        AIMessage(content="research completed")
                    ]
                }
            finally:
                active_count -= 1

        researcher = {
            "name": "researcher",
            "description": "Runs one bounded research task.",
            "runnable": RunnableLambda(
                lambda state: {
                    "messages": [
                        AIMessage(content="sync researcher result")
                    ]
                },
                afunc=run_researcher,
            ),
        }
        model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            task_call("task-call-1"),
                            task_call("task-call-2"),
                            task_call("task-call-3"),
                        ],
                    ),
                    AIMessage(content="supervisor final answer"),
                ]
            )
        )
        graph = create_agent(
            model=model,
            tools=[],
            middleware=[
                ResearchTaskConcurrencyMiddleware(),
                SubAgentMiddleware(
                    backend=StateBackend(),
                    subagents=[researcher],
                    state_schema=ResearchState,
                ),
            ],
            context_schema=ResearchContext,
            state_schema=ResearchState,
        )
        context = ResearchContext(
            max_search_calls=4,
            max_page_reads=6,
            max_page_chars=1000,
            max_parallel_research_tasks=2,
        )

        invoke_task = asyncio.create_task(
            graph.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Run three independent tasks.",
                        }
                    ]
                },
                context=context,
            )
        )

        await asyncio.wait_for(two_tasks_started.wait(), timeout=1)
        self.assertEqual(max_active_count, 2)

        release_tasks.set()
        result = await asyncio.wait_for(invoke_task, timeout=1)

        task_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        rejected_messages = [
            message
            for message in task_messages
            if message.status == "error"
        ]

        self.assertEqual(len(rejected_messages), 1)
        self.assertEqual(
            json.loads(rejected_messages[0].content)["error"],
            "parallel_task_limit_reached",
        )
        self.assertEqual(max_active_count, 2)
        self.assertEqual(active_count, 0)

        self.assertTrue(context.try_acquire_research_task_slot())
        self.assertTrue(context.try_acquire_research_task_slot())
        self.assertFalse(context.try_acquire_research_task_slot())
        context.release_research_task_slot()
        context.release_research_task_slot()

    async def test_task_exception_releases_permit(self):
        async def failing_researcher(state):
            raise RuntimeError("researcher failed")

        model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[task_call("task-call-failing")],
                    )
                ]
            )
        )
        context = make_context()

        with self.assertRaisesRegex(RuntimeError, "researcher failed"):
            await build_task_graph(model, failing_researcher).ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Run a failing task.",
                        }
                    ]
                },
                context=context,
            )

        self.assertTrue(context.try_acquire_research_task_slot())
        self.assertTrue(context.try_acquire_research_task_slot())
        context.release_research_task_slot()
        context.release_research_task_slot()

    async def test_task_cancellation_releases_permit(self):
        researcher_started = asyncio.Event()
        never_release = asyncio.Event()

        async def blocking_researcher(state):
            researcher_started.set()
            await never_release.wait()
            return {
                "messages": [AIMessage(content="unreachable")]
            }

        model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[task_call("task-call-cancelled")],
                    )
                ]
            )
        )
        context = make_context()
        invoke_task = asyncio.create_task(
            build_task_graph(model, blocking_researcher).ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Run a cancellable task.",
                        }
                    ]
                },
                context=context,
            )
        )

        await asyncio.wait_for(researcher_started.wait(), timeout=1)
        invoke_task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await invoke_task

        self.assertTrue(context.try_acquire_research_task_slot())
        self.assertTrue(context.try_acquire_research_task_slot())
        context.release_research_task_slot()
        context.release_research_task_slot()

    async def test_non_task_tool_runs_when_task_slots_are_exhausted(self):
        model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "echo_tool",
                                "args": {"value": "echo result"},
                                "id": "echo-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="supervisor final answer"),
                ]
            )
        )
        context = make_context()
        self.assertTrue(context.try_acquire_research_task_slot())
        self.assertTrue(context.try_acquire_research_task_slot())

        try:
            result = await create_agent(
                model=model,
                tools=[echo_tool],
                middleware=[ResearchTaskConcurrencyMiddleware()],
                context_schema=ResearchContext,
            ).ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Use the echo tool.",
                        }
                    ]
                },
                context=context,
            )
        finally:
            context.release_research_task_slot()
            context.release_research_task_slot()

        tool_message = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        )
        self.assertEqual(tool_message.content, "echo result")


if __name__ == "__main__":
    unittest.main()
