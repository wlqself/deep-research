from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from .. import agent as agent_module
from ..agent.messages import content_to_text
from ..citations import append_verified_sources
from ..state.access import sources_for_thread, thread_values
from ..state.runtime import agent_config

router = APIRouter()


class ThreadSnapshot(BaseModel):
    thread_id: UUID
    title: str
    messages: list[dict[str, str]]
    summary: str | None
    todos: list[dict[str, str]]
    artifacts: list[dict[str, Any]]

class ThreadRenameRequest(BaseModel):
    title: str = Field(
        min_length=1,
        max_length=80,
    )

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("title must not be empty")

        return normalized


class ThreadRenameResponse(BaseModel):
    thread_id: UUID
    title: str

class ThreadDeleteResponse(BaseModel):
    thread_id: UUID
    deleted: bool

# 判断一条 Message 是不是 SummarizationMiddleware 生成的内部摘要消息。
def _is_summary_message(message: Any) -> bool:
    return(
        isinstance(message, HumanMessage)
        and message.additional_kwargs.get("lc_source")
        == "summarization"
    )
# 逻辑顺序
# 摘要消息 -> 过滤
# ToolMessage -> 过滤
# 带 tool_calls 的 AIMessage -> 过滤
# 普通 HumanMessage -> user
# 普通 AIMessage -> assistant

def _serialize_message(message: Any) -> dict[str, str] | None:

    if _is_summary_message(message):
        return None   
    # 前端不会看到搜索 API 返回的 JSON
    if isinstance(message, ToolMessage):
        return None

    if isinstance(message, HumanMessage):
        role = "user"

    # LLM：我要调用 web_search。 这种话用户不应看到
    elif isinstance(message, AIMessage):
        if message.tool_calls:
            return None

        role = "assistant"
    else:
        return None
    #  尝试读取 message.content；如果这个对象根本没有 content 属性，就返回 ""，而不是报错。
    content = content_to_text(
        getattr(message, "content", "")
    )

    if not content:
        return None

    return {
        "role": role,
        "content": content,
    }


def _serialize_todos(values: dict[str, Any]) -> list[dict[str, str]]:
    raw_todos = values.get("todos", [])

    if not isinstance(raw_todos, list):
        return []

    todos: list[dict[str, str]] = []

    for item in raw_todos:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        status = item.get("status")

        if not isinstance(content, str):
            continue

        if status not in {
            "pending",
            "in_progress",
            "completed",
        }:
            continue

        todos.append(
            {
                "content": content,
                "status": status,
            }
        )

    return todos



def _serialize_artifacts(
    values: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_artifacts = values.get("artifacts", {})

    if not isinstance(raw_artifacts, dict):
        return []

    artifacts: list[dict[str, Any]] = []

    for artifact_id, metadata in raw_artifacts.items():
        if not isinstance(artifact_id, str):
            continue

        if not isinstance(metadata, dict):
            continue

        artifact = dict(metadata)
        artifact.setdefault("artifact_id", artifact_id)
        artifacts.append(artifact)

    return artifacts

def _extract_summary(
    values: dict[str, Any],
) -> str | None:
    raw_messages = values.get("messages", [])

    if not isinstance(raw_messages, list):
        return None

    for message in reversed(raw_messages):
        if not _is_summary_message(message):
            continue

        content = content_to_text(
            getattr(message, "content", "")
        )

        return content or None

    return None
"""
PATCH /threads/{thread_id}
  -> 校验标题
  -> 读取 Checkpointer
  -> 确认线程存在
  -> aupdate_state(thread_title)
  -> 返回新标题

标题为空或超过 80 字
  -> 422

线程不存在
  -> 404

Checkpointer 写入失败
  -> 503
  -> 不返回成功响应

写入成功
  -> 返回 200 + 新标题
"""
@router.patch(
    "/threads/{thread_id}",
    response_model=ThreadRenameResponse,
)
async def rename_thread(
    thread_id: UUID,
    payload: ThreadRenameRequest,
) -> ThreadRenameResponse:
    normalized_thread_id = str(thread_id)

    values = await thread_values(
        agent_module.agent,
        normalized_thread_id,
    )

    raw_messages = values.get("messages", [])

    if (
        not isinstance(raw_messages, list)
        or not any(
            _serialize_message(message)
            for message in raw_messages
        )
    ):
        raise HTTPException(
            status_code=404,
            detail="Thread not found.",
        )

    try:
        await agent_module.agent.aupdate_state(
            agent_config(normalized_thread_id),
            {
                "thread_title": payload.title,
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Thread rename failed.",
        ) from exc

    return ThreadRenameResponse(
        thread_id=thread_id,
        title=payload.title,
    )

#从当前线程恢复出来的 State 里拿消息，然后过滤成前端能展示的聊天记录。
@router.get(
    "/threads/{thread_id}",
    response_model=ThreadSnapshot,
)
async def get_thread_snapshot(
    thread_id: UUID,
) -> ThreadSnapshot:
    normalized_thread_id = str(thread_id)

    values = await thread_values(
        agent_module.agent,
        normalized_thread_id,
    )

    raw_messages = values.get("messages", [])

    if not isinstance(raw_messages, list):
        raise HTTPException(
            status_code=404,
            detail="Thread not found.",
        )

    messages: list[dict[str, str]] = []

    for message in raw_messages:
        serialized = _serialize_message(message)

        if serialized is not None:
            messages.append(serialized)

    if not messages:
        raise HTTPException(
            status_code=404,
            detail="Thread not found.",
        )

    summary = _extract_summary(values)

    # 把 [S1] 这类引用重新验证并补上来源链接。
    sources = await sources_for_thread(
        agent_module.agent,
        normalized_thread_id,
    )

    for message in messages:
        if message["role"] != "assistant":
            continue

        message["content"] = append_verified_sources(
            message["content"],
            sources,
        )
    raw_title = values.get("thread_title")

    thread_title = (
        raw_title.strip()[:80]
        if isinstance(raw_title, str)
        and raw_title.strip()
        else messages[0]["content"][:80]
    )
    return ThreadSnapshot(
        thread_id=thread_id,
        title=thread_title,
        messages=messages,
        summary=summary,
        todos=_serialize_todos(values),
        artifacts=_serialize_artifacts(values),
    )

@router.delete(
    "/threads/{thread_id}",
    response_model=ThreadDeleteResponse,
)
async def delete_thread(
    thread_id: UUID,
) -> ThreadDeleteResponse:
    normalized_thread_id = str(thread_id)

    values = await thread_values(
        agent_module.agent,
        normalized_thread_id,
    )

    raw_messages = values.get("messages", [])

    if (
        not isinstance(raw_messages, list)
        or not any(
            _serialize_message(message)
            for message in raw_messages
        )
    ):
        raise HTTPException(
            status_code=404,
            detail="Thread not found.",
        )

    checkpointer = getattr(
        agent_module.agent,
        "checkpointer",
        None,
    )

    if (
        checkpointer is None
        or not hasattr(checkpointer, "adelete_thread")
    ):
        raise HTTPException(
            status_code=503,
            detail="Thread deletion is unavailable.",
        )

    try:
        await checkpointer.adelete_thread(
            normalized_thread_id
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Thread deletion failed.",
        ) from exc

    return ThreadDeleteResponse(
        thread_id=thread_id,
        deleted=True,
    )