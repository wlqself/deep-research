import logging
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import ValidationError
from ..log.logging_utils import log_event
from ..context import ResearchContext
from ..memory.decision import decide_candidate
from ..memory.service import MemoryService
from ..memory.type import (
    MemoryCandidate,
    MemoryEntryKind,
    MemoryScope,
    MemoryEntry,
)
from ..config import settings
logger = logging.getLogger(__name__)


def _error(
    code: str,
    message: str,
) -> dict[str, object]:
    return {
        "ok": False,
        "error": code,
        "message": message,
    }


def build_remember_memory_tool(
    memory_service: MemoryService,
):
    @tool
    async def remember_memory(
        kind: MemoryEntryKind,
        memory_key: str,
        scope: MemoryScope,
        title: str,
        summary: str,
        content: str,
        keywords: list[str],
        mode: Literal[
            "remember",
            "update",
        ] = "remember",
        url: str | None = None,
        verified_at: str | None = None,
        incorrect: str | None = None,
        correct: str | None = None,
        applies_when: str | None = None,
        runtime: ToolRuntime[ResearchContext] = None,
    ) -> dict[str, object]:
        """Explicitly remember or update a user-approved long-term memory."""
        configurable = (
            runtime.config.get("configurable", {})
            if runtime is not None
            else {}
        )

        thread_id = configurable.get("thread_id")

        if not isinstance(thread_id, str) or not thread_id.strip():
            return _error(
                "missing_thread_id",
                "A thread_id is required.",
            )

        try:
            candidate = MemoryCandidate.model_validate(
                {
                    "kind": kind,
                    "memory_key": memory_key,
                    "scope": scope,
                    "title": title,
                    "summary": summary,
                    "content": content,
                    "keywords": keywords,
                    "source_type": "explicit_user", # 区分这条记忆是「用户主动让记住的」还是「系统自动从对话里提取的」
                    "source_thread_id": thread_id,
                    "url": url,
                    "verified_at": verified_at,
                    "incorrect": incorrect,
                    "correct": correct,
                    "applies_when": applies_when,
                }
            )
        except ValidationError:
            return _error(
                "invalid_memory",
                "Memory fields are invalid.",
            )

        try:
            # 精确查找旧记忆
            existing_entries = (
                await memory_service.find_active_by_key(
                    candidate.kind,
                    candidate.memory_key,
                    candidate.scope,
                )
            )

            if len(existing_entries) > 1:
                return _error(
                    "duplicate_active_memory_key",
                    "Multiple active memories share this key.",
                )

            existing = (
                existing_entries[0]
                if existing_entries
                else None
            )
  
            if mode == "update" and existing is None:
                return _error(
                    "memory_not_found",
                    "No active memory exists for this key.",
                )

            intent = (
                "explicit_update"
                if mode == "update"
                else "explicit_remember"
            )
            # 根据 intent + existing + candidate 决定：新建 / 更新 / 忽略
            decision = decide_candidate(
                candidate,
                existing_entries,
                intent=intent,
            )
            # 写 SQLite、标记旧记忆为 superseded、重建投影
            saved = await memory_service.apply_decision(
                candidate,
                decision,
                existing=existing,
                clear_suppression=True,
            )

        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "memory.write.failed",
                thread_id=thread_id,
                status="failed",
                error_code="memory_write_failed",
                exception_type=type(error).__name__,
                exc_info=True,
            )
            return _error(
                "memory_write_failed",
                "The memory operation failed.",
            )

        return {
            "ok": True,
            "action": decision.action,
            "memory_id": (
                saved.id
                if saved is not None
                else decision.existing_id
            ),
        }

    return remember_memory
"""
Main 查询文本
  -> MemoryService.find_review_memories()
  -> 四类 active 记忆
  -> 限制 settings.memory_recall_limit
  -> 去除 source_thread_id 等内部字段
  -> 返回 Main
"""
def build_recall_memories_tool(
    memory_service: MemoryService,
):
    @tool
    async def recall_memories(
        query: str,
    ) -> dict[str, object]:
        """Read relevant active long-term memories for the Main agent."""

        normalized_query = query.strip()

        if not normalized_query:
            return _error(
                "invalid_query",
                "Memory query must not be empty.",
            )

        try:
            entries = (
                await memory_service.find_review_memories(
                    normalized_query,
                    limit=settings.memory_recall_limit,
                )
            )
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "memory.recall.failed",
                status="failed",
                error_code="memory_recall_failed",
                exception_type=type(error).__name__,
                exc_info=True,
            )
            return _error(
                "memory_recall_failed",
                "Memory recall failed.",
            )

        memories = [
            _public_memory(entry)
            for entry in entries
        ]
        return {
            "ok": True,
            "memories": memories,
        }

    return recall_memories

def _public_memory(
    entry: MemoryEntry,
) -> dict[str, object]:
    return entry.model_dump(
        mode="json",
        include={
            "id",
            "memory_key",
            "scope",
            "kind",
            "title",
            "summary",
            "content",
            "keywords",
            "status",
            "source_type",
            "url",
            "verified_at",
            "incorrect",
            "correct",
            "applies_when",
            "updated_at",
        },
    )
# 让 Agent 在对话中主动回看「我自己目前记住了哪些东西」，并以结构化 JSON 返回，供 Agent 决定要不要基于已有记忆继续回答（而不是重复问用户已知信息）。
def build_list_memories_tool(
    memory_service: MemoryService,
):
    @tool
    async def list_memories(
        kind: MemoryEntryKind | None = None,
        keyword: str | None = None,
        page: int = 1,
    ) -> dict[str, object]:
        """List active long-term memories for the Main agent."""

        try:
            entries = (
                await memory_service.list_active_memories(
                    kind=kind,
                    keyword=keyword,
                    page=page,
                    page_size=settings.memory_page_size,
                )
            )
        except ValueError:
            return _error(
                "invalid_memory_list_request",
                "Memory list parameters are invalid.",
            )
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "memory.list.failed",
                status="failed",
                error_code="memory_list_failed",
                exception_type=type(error).__name__,
                exc_info=True,
            )
            return _error(
                "memory_list_failed",
                "Memory listing failed.",
            )

        return {
            "ok": True,
            "page": page,
            "page_size": settings.memory_page_size,
            "memories": [
                _public_memory(entry)
                for entry in entries
            ],
        }

    return list_memories

"""
memory_id
  -> get_entry()
  -> 确认 active
  -> delete_entry()

memory_key + scope
  -> find_active_by_key()
  -> 确认唯一 active
  -> delete_entry()

memory_key
  -> 稳定的“事实槽位”
  -> 用来判断是不是同一件事

memory_id
  -> 某一条具体记录的唯一 ID
  -> 通常由系统生成

forget_memory
  -> 定位 active entry
  -> MemoryService.forget_entry()
       -> abatch:
            suppression put
            active delete
       -> recall_revision + 1
       -> rebuild_projections()
  -> 成功才返回 ok=True

forget_entry() 抛异常
-> 工具进入现有 except
-> 返回 memory_forget_failed
-> 不返回 ok=True
"""
def build_forget_memory_tool(
    memory_service: MemoryService,
):
    @tool
    async def forget_memory(
        kind: MemoryEntryKind,
        memory_id: str | None = None,
        memory_key: str | None = None,
        scope: MemoryScope | None = None,
        runtime: ToolRuntime[ResearchContext] = None,
    ) -> dict[str, object]:
        """Forget one active long-term memory."""
        # 获取线程 ID
        configurable = (
            runtime.config.get("configurable", {})
            if runtime is not None
            else {}
        )

        thread_id = configurable.get("thread_id")

        if not isinstance(thread_id, str) or not thread_id.strip():
            thread_id = None

        has_id = (
            isinstance(memory_id, str)
            and bool(memory_id.strip())
        )
        has_key = (
            isinstance(memory_key, str)
            and bool(memory_key.strip())
        )

        if has_id == has_key:
            return _error(
                "invalid_forget_request",
                "Provide exactly one of memory_id or memory_key.",
            )

        try:
            if has_id:
                entry = await memory_service.get_entry(
                    kind,
                    memory_id.strip(),
                )

                if (
                    entry is None
                    or entry.status != "active"
                ):
                    return _error(
                        "memory_not_found",
                        "Active memory was not found.",
                    )

            else:
                if scope is None:
                    return _error(
                        "missing_scope",
                        "scope is required when using memory_key.",
                    )

                entries = (
                    await memory_service.find_active_by_key(
                        kind,
                        memory_key.strip(),
                        scope,
                    )
                )

                if len(entries) > 1:
                    return _error(
                        "duplicate_active_memory_key",
                        "Multiple active memories share this key.",
                    )

                if not entries:
                    return _error(
                        "memory_not_found",
                        "Active memory was not found.",
                    )

                entry = entries[0]

            await memory_service.forget_entry(
                entry.kind,
                entry.id,
                source_thread_id=thread_id,
                reason="explicit_user_forget",
            )

        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "memory.forget.failed",
                thread_id=thread_id,
                memory_id=(
                    memory_id.strip()
                    if has_id
                    else None
                ),
                status="failed",
                error_code="memory_forget_failed",
                exception_type=type(error).__name__,
                exc_info=True,
            )
            return _error(
                "memory_forget_failed",
                "The memory could not be forgotten.",
            )

        return {
            "ok": True,
            "deleted_memory_id": entry.id,
        }

    return forget_memory