import logging
import sys
from typing import Any

from .log_safety import _EVENT_PATTERN

# 
def _report_logging_failure() -> None:
    try:
        sys.stderr.write(
            "CRITICAL business logging failed\n"
        )
        sys.stderr.flush()
    except Exception:
        pass

def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    request_id: str | None = None,
    thread_id: str | None = None,
    task_id: str | None = None,
    agent_name: str | None = None,
    tool_name: str | None = None,
    document_id: str | None = None,
    memory_id: str | None = None,
    artifact_id: str | None = None,
    status: str | None = None,
    elapsed_ms: float | None = None,
    error_code: str | None = None,
    exception_type: str | None = None,
    method: str | None = None,
    route_template: str | None = None,
    status_code: int | None = None,
    exc_info: bool | tuple[Any, ...] = False,
) -> None:
    try:
        if (
            not isinstance(event, str)
            or not _EVENT_PATTERN.fullmatch(event)
        ):
            _report_logging_failure()
            return

        fields = {
            "event": event,
            "request_id": request_id,
            "thread_id": thread_id,
            "task_id": task_id,
            "agent_name": agent_name,
            "tool_name": tool_name,
            "document_id": document_id,
            "memory_id": memory_id,
            "artifact_id": artifact_id,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "error_code": error_code,
            "exception_type": exception_type,
            "method": method,
            "route_template": route_template,
            "status_code": status_code,
        }

        logger.log(
            level,
            event,
            extra=fields,
            exc_info=exc_info,
        )
    except Exception:
        _report_logging_failure()