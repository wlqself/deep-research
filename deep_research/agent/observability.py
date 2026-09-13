"""Safe, request-scoped lifecycle events for the research SSE stream.

This module deliberately contains no model payloads.  It is the boundary
between internal execution diagnostics and the public event stream.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import uuid4

from ..log.log_context import get_request_id


def monotonic_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


class LifecycleTracker:
    def __init__(self, *, thread_id: str, run_id: str | None = None) -> None:
        self.thread_id = thread_id
        self.request_id = get_request_id() or uuid4().hex
        self.run_id = run_id or uuid4().hex
        self.started_at = time.perf_counter()
        self.first_token_received = False
        self.model_attempts = 0
        self.heartbeat_count = 0

    def event(self, event_type: str, **fields: object) -> dict[str, object]:
        event: dict[str, object] = {
            "type": event_type,
            "event_id": uuid4().hex,
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "elapsed_ms": monotonic_ms(self.started_at),
            "timestamp": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        }
        event.update(fields)
        return event

    def model_started(self, *, agent_name: str = "main") -> dict[str, object]:
        self.model_attempts += 1
        return self.event(
            "model_started",
            agent_name=agent_name,
            model_role="main" if agent_name == "main" else "researcher",
            phase="model",
            status="started",
        )

    def first_token(self) -> dict[str, object] | None:
        if self.first_token_received:
            return None
        self.first_token_received = True
        return self.event(
            "first_token",
            agent_name="main",
            model_role="main",
            phase="model",
            status="received",
        )

    def heartbeat(self, *, phase: str = "processing") -> dict[str, object]:
        self.heartbeat_count += 1
        return self.event(
            "heartbeat",
            phase=phase,
            agent_name="main",
            status="running",
        )

    def done(self, *, goal_status: str, error_codes: list[str]) -> dict[str, object]:
        return self.event(
            "done",
            goal_status=goal_status,
            error_codes=list(dict.fromkeys(error_codes)),
            first_token_received=self.first_token_received,
            model_attempts=self.model_attempts,
            heartbeat_count=self.heartbeat_count,
        )


def safe_model_error_code(error: BaseException | object) -> str:
    """Map arbitrary failures to a stable public code without exposing text."""
    name = type(error).__name__.lower()
    if "timeout" in name:
        return "model_timeout"
    if "connect" in name or "network" in name:
        return "model_connection_failed"
    if "invalid" in name or "output" in name:
        return "model_output_invalid"
    return "model_failed"
