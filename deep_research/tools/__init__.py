from .assess_research import assess_research
from .read_page import read_page
from .web_search import web_search
from .save_report import save_report
from .record_finding import record_research_finding
from .list_findings import list_research_findings
from .search_knowledge_base import (
    build_search_knowledge_base_tool,
)
from .memory import (
    build_forget_memory_tool,
    build_list_memories_tool,
    build_recall_memories_tool,
    build_remember_memory_tool,
)
RESEARCHER_TOOLS = [
    web_search,
    read_page,
    assess_research,
    record_research_finding,
    list_research_findings,
]

SUPERVISOR_TOOLS = [
    list_research_findings,
    save_report,
]

# 兼容旧代码和旧测试
RESEARCH_TOOLS = [
    *RESEARCHER_TOOLS,
    save_report,
]

def build_researcher_tools(
    rag_service=None,
):
    if rag_service is None:
        return list(RESEARCHER_TOOLS)

    return [
        *RESEARCHER_TOOLS,
        build_search_knowledge_base_tool(
            rag_service
        ),
    ]

def build_supervisor_tools(
    memory_service=None,
):
    tools = list(SUPERVISOR_TOOLS)

    if memory_service is not None:
        tools.extend(
            [
                build_remember_memory_tool(
                    memory_service
                ),
                build_recall_memories_tool(
                    memory_service
                ),
                build_list_memories_tool(
                    memory_service
                ),
                build_forget_memory_tool(
                    memory_service
                ),
            ]
        )

    return tools

__all__ = [
    "web_search",
    "read_page",
    "assess_research",
    "RESEARCH_TOOLS",
    "save_report",
    "record_research_finding",
    "list_research_findings",
    "RESEARCHER_TOOLS",
    "SUPERVISOR_TOOLS",
    "build_researcher_tools",
    "build_remember_memory_tool",
    "build_recall_memories_tool",
    "build_list_memories_tool",
    "build_supervisor_tools",
    "build_forget_memory_tool",
]
