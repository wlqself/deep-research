"""Resume a LangGraph run after an application-level HITL decision."""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

from ..state.access import context_for_thread
from ..state.runtime import agent_config


def _interrupt_value(record: Any) -> Any:
    if isinstance(record, dict):
        return record.get("value")
    return getattr(record, "value", None)


def _snapshot_interrupts(snapshot: Any) -> list[Any]:
    """Collect interrupts from both StateSnapshot and task-level layouts."""

    records: list[Any] = []
    top_level = getattr(snapshot, "interrupts", ())
    if isinstance(top_level, (list, tuple)):
        records.extend(top_level)

    tasks = getattr(snapshot, "tasks", ())
    if isinstance(tasks, (list, tuple)):
        for task in tasks:
            task_interrupts = (
                task.get("interrupts", ())
                if isinstance(task, dict)
                else getattr(task, "interrupts", ())
            )
            if isinstance(task_interrupts, (list, tuple)):
                records.extend(task_interrupts)
    return records


def interrupt_id(record: Any) -> str | None:
    """Read the stable interrupt id across LangGraph record versions."""

    value = record.get("id") if isinstance(record, dict) else getattr(record, "id", None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def checkpoint_id(snapshot: Any) -> str | None:
    """Read the checkpoint cursor used to resume the current snapshot."""

    config = getattr(snapshot, "config", None)
    if isinstance(config, dict):
        configurable = config.get("configurable", {})
        value = configurable.get("checkpoint_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def native_hitl_snapshot(
    agent: Any,
    *,
    thread_id: str,
) -> tuple[Any, list[Any]]:
    """Load the authoritative graph snapshot and its pending interrupts."""

    snapshot = await agent.aget_state(agent_config(thread_id))
    return snapshot, _snapshot_interrupts(snapshot)


async def pending_native_hitl_interrupts(
    agent: Any,
    *,
    thread_id: str,
) -> list[Any]:
    """Return native graph interrupts currently blocking a thread."""

    if agent is None or not hasattr(agent, "aget_state"):
        return []
    _, interrupts = await native_hitl_snapshot(agent, thread_id=thread_id)
    return interrupts


def _next_publication_intent(
    interrupt_value: Any,
    *,
    decision: str,
) -> dict[str, object] | None:
    """Build the second, explicit publish-confirmation card after approval."""

    if decision != "approved" or not isinstance(interrupt_value, dict):
        return None
    target = interrupt_value.get("target")
    if not isinstance(target, dict):
        return None
    required = (
        "approval_id",
        "article_id",
        "article_title",
        "article_slug",
        "article_version",
        "channel",
    )
    if not all(target.get(field) for field in required):
        return None

    return {
        "action": "resume",
        "resolution_status": "resolved",
        "target": {
            **target,
            "approval_status": "approved",
        },
        "candidates": [],
        "requires_user_confirmation": True,
        "interaction_id": None,
        "interaction_status": None,
        "attachment_ids": interrupt_value.get("attachment_ids", []),
    }


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return ""


async def resume_agent_after_hitl(
    agent: Any,
    *,
    thread_id: str,
    interaction_id: str,
    decision: str,
    expected_interrupt_id: str | None = None,
) -> dict[str, object]:
    """Resume the matching native interrupt, if this thread has one."""

    config = agent_config(thread_id)
    snapshot, snapshot_interrupts = await native_hitl_snapshot(
        agent,
        thread_id=thread_id,
    )
    matches = []
    for interrupt_record in snapshot_interrupts:
        value = _interrupt_value(interrupt_record)
        if (
            isinstance(value, dict)
            and value.get("kind") == "publication_approval"
            and value.get("interaction_id") == interaction_id
            and (
                expected_interrupt_id is None
                or interrupt_id(interrupt_record) == expected_interrupt_id
            )
        ):
            matches.append(interrupt_record)
    if not matches:
        return {
            "resumed": False,
            "error_code": "hitl_graph_not_interrupted",
        }

    interrupt_value = _interrupt_value(matches[0])
    matched_interrupt_id = interrupt_id(matches[0])
    matched_checkpoint_id = checkpoint_id(snapshot)
    resume_value = {
        "interaction_id": interaction_id,
        "decision": decision,
    }
    # LangGraph supports an interrupt-id keyed resume map. Using it when the
    # runtime supplied an id prevents a stale approval from being applied to
    # a newer interrupt in the same thread.
    resume_payload: Any = (
        {matched_interrupt_id: resume_value}
        if matched_interrupt_id
        else resume_value
    )
    result = await agent.ainvoke(
        Command(
            resume=resume_payload,
        ),
        config=config,
        context=await context_for_thread(agent, thread_id),
    )
    messages = []
    if isinstance(result, dict):
        raw_messages = result.get("messages", [])
        if isinstance(raw_messages, list):
            messages = raw_messages
        elif isinstance(result.get("values"), dict):
            raw_messages = result["values"].get("messages", [])
            if isinstance(raw_messages, list):
                messages = raw_messages

    # The last state item can be an empty tool message. Walk backwards to the
    # latest non-empty assistant response so approval does not appear silent.
    answer = ""
    for message in reversed(messages):
        answer = _message_text(message).strip()
        if answer:
            break
    return {
        "resumed": True,
        "interrupt_id": matched_interrupt_id,
        "checkpoint_id": matched_checkpoint_id,
        "answer": answer or "审批已确认，Agent 已继续执行发布流程。",
        "next_intent": _next_publication_intent(
            interrupt_value,
            decision=decision,
        ),
    }


__all__ = [
    "checkpoint_id",
    "interrupt_id",
    "native_hitl_snapshot",
    "pending_native_hitl_interrupts",
    "resume_agent_after_hitl",
]
