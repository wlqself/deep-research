from typing import Annotated, Literal, TypedDict

from langchain.agents import AgentState
from typing_extensions import NotRequired

from ..findings.types import ResearchFinding
from ..sources import PersistentSource
from ..artifacts.types import ArtifactMetadata

TaskStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
]

class TaskRecord(TypedDict):
    task_id: str
    agent_name: str
    description: str
    status: TaskStatus
    created_at: str
    started_at: str
    completed_at: str
    updated_at: str
    finding_ids: list[str]
    source_ids: list[str]
    error: str

_TERMINAL_TASK_STATUSES = frozenset(
    {"completed", "failed", "cancelled"}
)

def _copy_task_record(record: TaskRecord) -> TaskRecord:
    return {
        **record,
        "finding_ids": list(record["finding_ids"]),
        "source_ids": list(record["source_ids"]),
    }

def merge_tasks(
    current: dict[str, TaskRecord] | None,
    update: dict[str, TaskRecord] | None,
) -> dict[str, TaskRecord]:
    merged = {
        task_id: _copy_task_record(record)
        for task_id, record in (current or {}).items()
    }

    for task_id, incoming in (update or {}).items():
        existing = merged.get(task_id) # 遍历这次传入的更新，并检查这个任务以前是否已经存在。

        if existing is None:
            merged[task_id] = _copy_task_record(incoming)
            continue
        # 判断新旧状态是否为终态
        existing_terminal = (
            existing["status"] in _TERMINAL_TASK_STATUSES
        )
        incoming_terminal = (
            incoming["status"] in _TERMINAL_TASK_STATUSES
        )
        # 禁止任务状态倒退
        if existing_terminal and not incoming_terminal:
            continue

        if incoming["updated_at"] >= existing["updated_at"]:
            merged[task_id] = _copy_task_record(incoming)

    return merged

def merge_sources(
    current: dict[str, PersistentSource] | None,
    update: dict[str, PersistentSource] | None,
) -> dict[str, PersistentSource]:
    merged = dict(current or {})
    merged.update(update or {})
    return merged

def merge_artifacts(
    current: dict[str, ArtifactMetadata] | None,
    update: dict[str, ArtifactMetadata] | None,
) -> dict[str, ArtifactMetadata]:
    merged = dict(current or {})
    merged.update(update or {})
    return merged

def max_source_number(
    current: int | None,
    update: int | None,
) -> int:
    return max(current or 1, update or 1)

def merge_findings(
    current: dict[str, ResearchFinding] | None,
    update: dict[str, ResearchFinding] | None,
) -> dict[str, ResearchFinding]:
    merged = dict(current or {})
    merged.update(update or {})
    return merged

# State 的职责是
# sources   -> 当前线程来源注册表
# artifacts -> 当前线程正式报告元数据
# findings  -> 当前线程结构化研究结论
# messages  -> 对话与工具调用历史
# files     -> FilesystemMiddleware 管理的线程文件

class ResearchState(AgentState):
    sources: NotRequired[
        Annotated[
            dict[str, PersistentSource],
            merge_sources,
        ]
    ]

    next_source_number: NotRequired[
        Annotated[
            int,
            max_source_number,
        ]

    ]

    artifacts: NotRequired[
        Annotated[
            dict[str, ArtifactMetadata],
            merge_artifacts,
        ]
    ]

    findings : NotRequired[
        Annotated[
            dict[str, ResearchFinding],
            merge_findings,
        ]
    ]

    tasks: NotRequired[
        Annotated[
            dict[str, TaskRecord],
            merge_tasks,
        ]
    ]

    memory_review_turn_count: NotRequired[int]
    last_reviewed_message_id: NotRequired[str]
    last_archived_summary_hash: NotRequired[str]
    memory_review_backlog_pending: NotRequired[bool] # True   -> 上一次成功轮次之后仍有未复查内容 ；False 或缺失   -> 没有待处理 backlog
    thread_title: NotRequired[str]