import unittest

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)

from deep_research.citations import append_verified_sources
from deep_research.prompts.research import SUMMARY_PROMPT
from deep_research.state import ResearchState
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage

from deep_research.handlers.threads import (
    _extract_summary,
    _is_summary_message,
    _serialize_message,
)

class SummarizationStateTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_summary_model_end_event_has_summarization_metadata(self):
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    "answer 1",
                    "answer 2",
                    "summary text",
                    "answer 3",
                ]
            )
        )
        agent = create_agent(
            model=fake_model,
            tools=[],
            state_schema=ResearchState,
            middleware=[
                SummarizationMiddleware(
                    model=fake_model,
                    trigger=("messages", 4),
                    keep=("messages", 2),
                    summary_prompt=SUMMARY_PROMPT,
                    trim_tokens_to_summarize=200,
                )
            ],
            checkpointer=InMemorySaver(),
        )
        config = {
            "configurable": {
                "thread_id": "summary-event-metadata-thread",
            }
        }

        for question in ("question 1", "question 2"):
            await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                },
                config=config,
            )

        captured = []

        async for event in agent.astream_events(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question 3",
                    }
                ]
            },
            config=config,
            version="v2",
        ):
            metadata = event.get("metadata", {})

            if (
                event.get("event") == "on_chat_model_end"
                and isinstance(metadata, dict)
                and metadata.get("lc_source") == "summarization"
            ):
                captured.append(
                    {
                        "run_id": str(event.get("run_id", "")),
                        "metadata": metadata,
                        "output": (
                            event.get("data", {}).get("output")
                            if isinstance(event.get("data"), dict)
                            else None
                        ),
                    }
                )

        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0]["run_id"])
        self.assertEqual(
            captured[0]["metadata"]["lc_source"],
            "summarization",
        )

    async def test_summary_preserves_sources_and_artifacts(self):
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    "answer 1",
                    "answer 2",
                    "summary text",
                    "answer 3",
                ]
            )
        )

        checkpointer = InMemorySaver()

        agent = create_agent(
            model=fake_model,
            tools=[],
            state_schema=ResearchState,
            middleware=[
                SummarizationMiddleware(
                    model=fake_model,
                    trigger=("messages", 4),
                    keep=("messages", 2),
                    summary_prompt=SUMMARY_PROMPT,
                    trim_tokens_to_summarize=200,
                )
            ],
            checkpointer=checkpointer,
        )

        config = {
            "configurable": {
                "thread_id": "summary-state-thread",
            }
        }

        await agent.aupdate_state(
            config,
            {
                "sources": {
                    "S1": {
                        "source_id": "S1",
                        "title": "Test source",
                        "url": "https://example.com",
                        "snippet": "source content",
                    }
                },
                "artifacts": {
                    "artifact-1": {
                        "artifact_id": "artifact-1",
                        "filename": "report.md",
                        "workspace_path": "/final/report.md",
                        "created_at": "2026-08-21T00:00:00+00:00",
                        "size_bytes": 10,
                        "sha256": "a" * 64,
                    }
                },
                "findings": {
                    "F1": {
                        "finding_id": "F1",
                        "claim": "A persisted claim",
                        "evidence_summary": "Test evidence",
                        "source_ids": ["S1"],
                        "status": "supported",
                        "uncertainty": "",
                        "conflicts": "",
                        "created_at": "2026-08-21T00:00:00+00:00",
                        "updated_at": "2026-08-21T00:00:00+00:00",
                    }
                },
            },
        )

        before = await agent.aget_state(config)

        self.assertIn(
            "S1",
            before.values["sources"],
        )

        self.assertIn(
            "artifact-1",
            before.values["artifacts"],
        )

        self.assertEqual(
            before.values["findings"]["F1"]["claim"],
            "A persisted claim",
        )
        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question 1",
                    }
                ]
            },
            config=config,
        )

        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question 2",
                    }
                ]
            },
            config=config,
        )

        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question 3",
                    }
                ]
            },
            config=config,
        )

        after = await agent.aget_state(config)

        summary_messages = [
            message
            for message in after.values["messages"]
            if message.additional_kwargs.get("lc_source")
            == "summarization"
        ]


        self.assertEqual(
            len(summary_messages),
            1,
        )

        contents = [
            message.content
            for message in after.values["messages"]
        ]

        self.assertIn("question 3", contents)
        self.assertIn("answer 3", contents)

        self.assertEqual(
            after.values["sources"]["S1"]["title"],
            "Test source",
        )

        self.assertEqual(
            after.values["artifacts"]["artifact-1"]["filename"],
            "report.md",
        )

        self.assertEqual(
            after.values["findings"]["F1"]["claim"],
            "A persisted claim",
        )


        answer = "真实引用 [S1]，伪造引用 [S99]"

        verified = append_verified_sources(
            answer,
            after.values["sources"],
        )

        self.assertIn(
            "[S1] Test source - https://example.com",
            verified,
        )

        self.assertNotIn(
            "\n[S99] ",
            verified,
        )

    def test_summary_is_hidden_but_extractable(self):
        summary_message = HumanMessage(
            content="summary text",
            additional_kwargs={
                "lc_source": "summarization",
            },
        )

        self.assertTrue(
            _is_summary_message(summary_message)
        )

        self.assertIsNone(
            _serialize_message(summary_message)
        )

        self.assertEqual(
            _extract_summary(
                {
                    "messages": [
                        summary_message,
                    ],
                }
            ),
            "summary text",
        )

        wrapped_summary = HumanMessage(
            content=(
                "Here is a summary of the conversation to date:\n\n"
                "compressed summary"
            ),
            additional_kwargs={
                "lc_source": "summarization",
            },
        )

        self.assertEqual(
            _extract_summary({"messages": [wrapped_summary]}),
            "compressed summary",
        )

        normal_user_message = HumanMessage(
            content="真实用户问题",
        )

        self.assertEqual(
            _serialize_message(normal_user_message),
            {
                "role": "user",
                "content": "真实用户问题",
            },
        )

        image_user_message = HumanMessage(
            content="[系统图片附件上下文：图片附件属于用户上传的共享资源。]",
            additional_kwargs={
                "display_content": "请看看这张图片",
                "attachment_ids": ["image-1", "image-1", "image-2"],
            },
        )

        self.assertEqual(
            _serialize_message(image_user_message),
            {
                "role": "user",
                "content": "请看看这张图片",
                "attachment_ids": ["image-1", "image-2"],
            },
        )

    async def test_second_summary_replaces_first_summary(self):
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    "answer 1",
                    "answer 2",
                    "summary A",
                    "answer 3",
                    "summary B",
                    "answer 4",
                ]
            )
        )

        agent = create_agent(
            model=fake_model,
            tools=[],
            state_schema=ResearchState,
            middleware=[
                SummarizationMiddleware(
                    model=fake_model,
                    trigger=("messages", 4),
                    keep=("messages", 2),
                    summary_prompt=SUMMARY_PROMPT,
                    trim_tokens_to_summarize=200,
                )
            ],
            checkpointer=InMemorySaver(),
        )

        config = {
            "configurable": {
                "thread_id": "second-summary-thread",
            }
        }

        await agent.aupdate_state(
            config,
            {
                "findings": {
                    "F1": {
                        "finding_id": "F1",
                        "claim": "Finding survives repeated summaries",
                        "evidence_summary": "Test evidence",
                        "source_ids": ["S1"],
                        "status": "supported",
                        "uncertainty": "",
                        "conflicts": "",
                        "created_at": "2026-08-21T00:00:00+00:00",
                        "updated_at": "2026-08-21T00:00:00+00:00",
                    }
                }
            },
        )

        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question 1",
                    }
                ]
            },
            config=config,
        )

        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question 2",
                    }
                ]
            },
            config=config,
        )

        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question 3",
                    }
                ]
            },
            config=config,
        )

        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "question 4",
                    }
                ]
            },
            config=config,
        )

        state = await agent.aget_state(config)
        messages = state.values["messages"]

        summary_messages = [
            message
            for message in messages
            if message.additional_kwargs.get("lc_source")
            == "summarization"
        ]

        self.assertEqual(
            len(summary_messages),
            1,
        )

        self.assertIn(
            "summary B",
            summary_messages[0].content,
        )

        contents = [
            message.content
            for message in messages
        ]

        self.assertIn(
            "answer 4",
            contents,
        )

        self.assertEqual(
            state.values["findings"]["F1"]["claim"],
            "Finding survives repeated summaries",
        )

        visible_messages = [
            _serialize_message(message)
            for message in messages
        ]

        visible_messages = [
            message
            for message in visible_messages
            if message is not None
        ]

        self.assertFalse(
            any(
                message["content"]
                .startswith(
                    "Here is a summary of the conversation"
                )
                for message in visible_messages
            )
        )

        # 测试是否会保护标准工具调用配对
    def test_cutoff_keeps_ai_tool_pair(self):
        model = GenericFakeChatModel(
            messages=iter([])
        )

        middleware = SummarizationMiddleware(
            model=model,
            trigger=("messages", 100),
            keep=("messages", 2),
            trim_tokens_to_summarize=200,
        )

        messages = [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {
                            "query": "test",
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="search result",
                tool_call_id="call-1",
            ),
            AIMessage(content="answer"),
        ]

        cutoff = middleware._find_safe_cutoff_point(
            messages,
            2,
        )

        preserved = messages[cutoff:]

        self.assertIsInstance(
            preserved[0],
            AIMessage,
        )

        self.assertIsInstance(
            preserved[1],
            ToolMessage,
        )

        self.assertEqual(
            preserved[1].tool_call_id,
            preserved[0].tool_calls[0]["id"],
        )
