import unittest

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.prompts.research import (
    TODO_SYSTEM_PROMPT,
    TODO_TOOL_DESCRIPTION,
)
from deep_research.state import ResearchState

class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self
    
class TodoMiddlewareTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_write_todos_updates_thread_state(self):
        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "核对核心概念",
                                            "status": "in_progress",
                                        },
                                        {
                                            "content": "整理研究结论",
                                            "status": "pending",
                                        },
                                    ]
                                },
                                "id": "todo-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="研究计划已创建。",
                    ),
                ]
            )
        )

        graph = create_agent(
            model=fake_model,
            tools=[],
            state_schema=ResearchState,
            middleware=[
                TodoListMiddleware(
                    system_prompt=TODO_SYSTEM_PROMPT,
                    tool_description=TODO_TOOL_DESCRIPTION,
                )
            ],
            checkpointer=InMemorySaver(),
        )

        thread_a = {
            "configurable": {
                "thread_id": "todo-thread-a",
            }
        }

        thread_b = {
            "configurable": {
                "thread_id": "todo-thread-b",
            }
        }

        result = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "研究 LangChain 和 LangGraph 的职责",
                    }
                ]
            },
            config=thread_a,
        )

        state_a = await graph.aget_state(thread_a)
        state_b = await graph.aget_state(thread_b)

        self.assertEqual(
            result["todos"],
            [
                {
                    "content": "核对核心概念",
                    "status": "in_progress",
                },
                {
                    "content": "整理研究结论",
                    "status": "pending",
                },
            ],
        )

        self.assertEqual(
            state_a.values["todos"],
            result["todos"],
        )

        self.assertNotIn(
            "todos",
            state_b.values,
        )
    async def test_todo_status_moves_to_completed(self):
        fake_model = BindableFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "核对核心概念",
                                            "status": "in_progress",
                                        },
                                        {
                                            "content": "整理研究结论",
                                            "status": "pending",
                                        },
                                    ]
                                },
                                "id": "todo-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "核对核心概念",
                                            "status": "completed",
                                        },
                                        {
                                            "content": "整理研究结论",
                                            "status": "in_progress",
                                        },
                                    ]
                                },
                                "id": "todo-call-2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "核对核心概念",
                                            "status": "completed",
                                        },
                                        {
                                            "content": "整理研究结论",
                                            "status": "completed",
                                        },
                                    ]
                                },
                                "id": "todo-call-3",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="研究计划已完成。",
                    ),
                ]
            )
        )

        graph = create_agent(
            model=fake_model,
            tools=[],
            state_schema=ResearchState,
            middleware=[
                TodoListMiddleware(
                    system_prompt=TODO_SYSTEM_PROMPT,
                    tool_description=TODO_TOOL_DESCRIPTION,
                )
            ],
            checkpointer=InMemorySaver(),
        )

        config = {
            "configurable": {
                "thread_id": "todo-progress-thread",
            }
        }

        result = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "完成一个两步研究任务",
                    }
                ]
            },
            config=config,
        )

        state = await graph.aget_state(config)

        expected_todos = [
            {
                "content": "核对核心概念",
                "status": "completed",
            },
            {
                "content": "整理研究结论",
                "status": "completed",
            },
        ]

        self.assertEqual(
            result["todos"],
            expected_todos,
        )
        self.assertEqual(
            state.values["todos"],
            expected_todos,
        )
        self.assertEqual(
            result["messages"][-1].content,
            "研究计划已完成。",
        )
        
if __name__ == "__main__":
    unittest.main()