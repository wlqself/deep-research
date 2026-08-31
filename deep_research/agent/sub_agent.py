from typing import Any

from deepagents.middleware import CompiledSubAgent
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from ..context import ResearchContext
from ..prompts.researcher import RESEARCHER_SYSTEM_PROMPT
from ..state import ResearchState
from ..tools import (
    RESEARCHER_TOOLS,
    build_researcher_tools,
)
from .results import ResearcherResult


def build_researcher(
    model: Any,
    rag_service=None,
) -> CompiledSubAgent:
    researcher_tools = build_researcher_tools(
        rag_service
    )

    return {
        "name": "researcher",
        "description": (
            "负责复杂、多步骤网页研究，搜索和阅读来源，"
            "记录结构化 findings，但不直接回答用户或保存正式报告。"
        ),
        "runnable": create_agent(
            model=model,
            tools=researcher_tools,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            context_schema=ResearchContext,
            state_schema=ResearchState,
            response_format=ToolStrategy(ResearcherResult),
        ),
    }