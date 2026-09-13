from .assess_research import assess_research
from .read_page import read_page
from .web_search import web_search
from .save_report import save_report
from .publishing import (
    build_prepare_article_tool,
    build_read_article_for_revision_tool,
    build_revise_article_tool,
    build_publication_approval_status_tool,
    build_resolve_publication_intent_tool,
    build_request_publication_approval_tool,
    build_set_wechat_cover_tool,
    build_analyze_uploaded_image_tool,
    build_read_image_analysis_tool,
    build_generate_image_tool,
)
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
    publishing_service=None,
    hitl_service=None,
    image_attachment_service=None,
    image_analysis_service=None,
    wechat_cover_service=None,
    image_generation_service=None,
    enable_native_interrupt=False,
):
    tools = list(SUPERVISOR_TOOLS)

    if publishing_service is not None:
        tools.append(
            build_prepare_article_tool(
                publishing_service,
            )
        )
        tools.append(
            build_read_article_for_revision_tool(
                publishing_service,
            )
        )
        tools.append(
            build_revise_article_tool(
                publishing_service,
            )
        )
        tools.append(
            build_publication_approval_status_tool(
                publishing_service,
                hitl_service,
            )
        )
        tools.append(
            build_resolve_publication_intent_tool(
                publishing_service,
                hitl_service,
                enable_native_interrupt,
            )
        )
        tools.append(
            build_request_publication_approval_tool(
                publishing_service,
                hitl_service,
                enable_native_interrupt,
                image_attachment_service,
            )
        )
        if image_attachment_service is not None and wechat_cover_service is not None:
            tools.append(
                build_set_wechat_cover_tool(
                    image_attachment_service,
                    wechat_cover_service,
                )
            )
        if (
            image_attachment_service is not None
            and image_analysis_service is not None
        ):
            tools.append(
                build_analyze_uploaded_image_tool(
                    image_attachment_service,
                    image_analysis_service,
                )
            )
        elif image_attachment_service is not None:
            tools.append(
                build_read_image_analysis_tool(image_attachment_service)
            )
        if image_generation_service is not None:
            tools.append(build_generate_image_tool(image_generation_service))

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
    "build_prepare_article_tool",
    "build_read_article_for_revision_tool",
    "build_revise_article_tool",
    "build_publication_approval_status_tool",
    "build_resolve_publication_intent_tool",
    "build_request_publication_approval_tool",
    "build_set_wechat_cover_tool",
    "build_analyze_uploaded_image_tool",
    "build_read_image_analysis_tool",
    "build_generate_image_tool",
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
