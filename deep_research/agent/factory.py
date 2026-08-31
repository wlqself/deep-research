from langchain.agents import create_agent
from langchain.agents.middleware import (
    TodoListMiddleware,
    SummarizationMiddleware,
)
from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware

from ..context import ResearchContext
from deep_research.middleware import (
    ResearchTaskConcurrencyMiddleware,
)
from ..state import ResearchState
from ..tools import (
    build_supervisor_tools,
    RESEARCH_TOOLS,
    SUPERVISOR_TOOLS,
)

from ..prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from ..prompts.summary import SUMMARY_PROMPT
from ..prompts.todo import (
    TODO_SYSTEM_PROMPT,
    TODO_TOOL_DESCRIPTION,
)

from ..prompts.workspace import WORKSPACE_SYSTEM_PROMPT
from .sub_agent import build_researcher
from .model import model
from ..config import settings
from ..middleware.recall_middleware import (
    MainMemoryRecallMiddleware,
)

def build_agent(checkpointer, rag_service=None,memory_service=None,):
    filesystem_middleware = FilesystemMiddleware(
        backend=StateBackend(),
        tools=[
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
        ],
    )

    researcher = build_researcher(model, rag_service)

    main_memory_middleware = []

    if memory_service is not None:
        main_memory_middleware.append(
            MainMemoryRecallMiddleware(
                memory_service
            )
        )
        
    return create_agent(
        model=model,
        tools=build_supervisor_tools(
            memory_service,
        ),
        system_prompt=(
            f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
            f"{WORKSPACE_SYSTEM_PROMPT}"
        ),
        context_schema=ResearchContext,
        state_schema=ResearchState,
        checkpointer=checkpointer,
        middleware=[
            filesystem_middleware,
            ResearchTaskConcurrencyMiddleware(),
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[researcher],
                private_state_keys=frozenset(
                    {
                        "files",
                        "artifacts",
                        "tasks",
                        "memory_review_turn_count",
                        "last_reviewed_message_id",
                        "last_archived_summary_hash",
                        "memory_review_backlog_pending",
                    }
                ),
                state_schema=ResearchState,
            ),
            TodoListMiddleware(
                system_prompt=TODO_SYSTEM_PROMPT,
                tool_description=TODO_TOOL_DESCRIPTION,
            ),
            *main_memory_middleware,
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", settings.memory_trigger_tokens),
                keep=("messages", settings.memory_keep_messages),
                summary_prompt=SUMMARY_PROMPT,
                trim_tokens_to_summarize=settings.memory_trim_tokens,
            ),
        ],
    )
