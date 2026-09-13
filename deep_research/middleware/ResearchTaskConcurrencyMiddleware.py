import json
import logging

from ..log.logging_utils import log_event
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..context import ResearchContext

logger = logging.getLogger(
    "deep_research.subagent"
)

class ResearchTaskConcurrencyMiddleware(AgentMiddleware):
    @staticmethod
    def _is_task(request: ToolCallRequest) -> bool:
        return request.tool_call["name"] == "task"

    @staticmethod
    def _rejected(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=json.dumps(
                {
                    "ok": False,
                    "error": "parallel_task_limit_reached",
                    "message": (
                        "Maximum parallel research tasks reached."
                    ),
                },
                ensure_ascii=False,
            ),
            name="task",
            tool_call_id=request.tool_call["id"],
            status="error",
        )
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if not self._is_task(request):
            return handler(request)

        context = request.runtime.context

        if not isinstance(context, ResearchContext):
            return handler(request)

        if not context.try_acquire_research_task_slot():
            log_event(
                logger,
                logging.WARNING,
                "subagent.task.rejected",
                task_id=str(
                    request.tool_call.get(
                        "id",
                        "",
                    )
                ),
                status="rejected",
                error_code=(
                    "parallel_task_limit_reached"
                ),
            )
            return self._rejected(request)

        try:
            return handler(request)
        finally:
            context.release_research_task_slot()

    async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[
                [ToolCallRequest],
                Awaitable[ToolMessage | Command],
            ],
        ) -> ToolMessage | Command:
            if not self._is_task(request):
                return await handler(request)

            context = request.runtime.context

            if not isinstance(context, ResearchContext):
                return await handler(request)

            if not context.try_acquire_research_task_slot():
                log_event(
                    logger,
                    logging.WARNING,
                    "subagent.task.rejected",
                    task_id=str(
                        request.tool_call.get(
                            "id",
                            "",
                        )
                    ),
                    status="rejected",
                    error_code=(
                        "parallel_task_limit_reached"
                    ),
                )
                return self._rejected(request)

            try:
                return await handler(request)
            finally:
                context.release_research_task_slot()