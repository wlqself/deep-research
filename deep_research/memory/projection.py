import os
import tempfile
import hashlib

from pathlib import Path

from .type import SummaryArchive
from ..config import settings
from .type import MemoryEntry
from collections.abc import Sequence

_READ_WHEN: dict[str, str] = {
    "user": "生成回答或调整交互方式时读取。",
    "reference": "需要使用或介绍该外部参考资料时读取。",
    "project": "处理相关项目任务、约束或设计决策时读取。",
    "feedback": "准备采用类似做法时读取，避免重复已知错误。",
}

def _single_line(value: str) -> str:
    return " ".join(value.split())

def render_entry_markdown(entry: MemoryEntry) -> str:
    # Keywords 规范化
    keywords = ", ".join(
        _single_line(keyword)
        for keyword in entry.keywords
    )

    lines = [
        f"# {_single_line(entry.title)}",# H1 标题，单行化
        "",                              # 空行（Markdown 规范）
        "## Summary",
        entry.summary.strip(),           # 摘要（去除首尾空白）
        "",
        "## Content",
        entry.content.strip(),           # 完整内容
        "",
        "## When to read",
        _READ_WHEN[entry.kind],          # ← 关键设计：按 kind 给出阅读指引
        "",
        "## Metadata",                   # 机器可读的元数据区
        f"- ID: {entry.id}",
        f"- Memory key: {entry.memory_key}",
        f"- Kind: {entry.kind}",
        f"- Scope: {entry.scope}",
        f"- Status: {entry.status}",
        f"- Source type: {entry.source_type}",
        f"- Updated: {entry.updated_at.isoformat()}",
        f"- Keywords: {keywords}",
    ]
    # 条件字段：URL 类记忆，只对 reference 类且带 URL 的记忆才会追加这两行。比如网页抓取的知识，需要记录来源链接和验证时间。
    if entry.kind == "reference":
        lines.extend(
            [
                f"- URL: {entry.url}",
                (
                    "- Verified at: "
                    f"{entry.verified_at.isoformat()}"
                ),
            ]
        )
    # Feedback 类型有额外的纠错字段：
    if entry.kind == "feedback":
        lines.extend(
            [
                "",
                "## Feedback",
                f"- Incorrect: {entry.incorrect}",
                f"- Correct: {entry.correct}",
                f"- Applies when: {entry.applies_when}",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"

def projection_root() -> Path:
    return Path(settings.memory_projection_path)

def ensure_projection_directories() -> None:
    root = projection_root()

    for relative_path in (
        "memories/reference",
        "memories/project",
        "memories/feedback",
        "memory-internal/summary-archive",
    ):
        (root / relative_path).mkdir(
            parents=True,
            exist_ok=True,
        )

def write_text_atomic(
    path: Path,
    content: str,
) -> None:
    # 确保目录存在
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        # 在同目录下创建隐藏临时文件
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",  # ← 隐藏文件（以 . 开头）
            suffix=".tmp",
            delete=False,  # ← 退出 with 块后不自动删除
        ) as temporary_file: # 写入 + 强制刷盘，确保数据真正落到磁盘上，而不是停留在操作系统缓存里
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        #  原子替换，POSIX：rename() 系统调用 → 原子操作
        os.replace(
            temporary_path,
            path,
        )
        temporary_path = None
    # 异常清理
    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )
# 排序，最新的记忆排最前面
def _ordered_entries(
    entries: Sequence[MemoryEntry],
) -> list[MemoryEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            entry.updated_at,
            entry.id,
        ),
        reverse=True,
    )

# Markdown 表格用 | 做列分隔符。如果 title 或 summary 里本身含有 |（比如"输入|输出"），不转义就会破坏表格结构。
def _index_cell(value: str) -> str:
    return _single_line(value).replace("|", "\\|")

# 零填充保证按文件名字母排序 = 按页码数字排序，LLM 或 ls 列出来天然有序。
def _page_filename(page_number: int) -> str:
    return f"page-{page_number:03d}.md"

# 把"摘要"和"什么时候该读"拼在一起，让 LLM 在 index 表格里就能判断是否需要深入阅读该条记忆。
def _index_summary(entry: MemoryEntry) -> str:
    return (
        f"{_single_line(entry.summary)} "
        f"读取：{_READ_WHEN[entry.kind]}"
    )

# 给 LLM 看的目录页——用表格列出所有记忆的标题、摘要、所在页码
def render_index_markdown(
    entries: Sequence[MemoryEntry],
    *,
    page_size: int,
) -> str:
    if page_size < 1:
        raise ValueError("page_size must be positive")

    ordered = _ordered_entries(entries)

    lines = [
        "# Memory Index",
        "",
        "| ID | Page | Title | Summary | Keywords | Updated | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    """
    外层循环：按 page_size 切页
    内层循环：把当前页的所有 entry 追加到表格里，都指向同一个 page_name
    结果：LLM 看到表格后，知道"我想读第 3 条记忆，它在 page-001.md 里"
    """
    for start in range(0, len(ordered), page_size):
        page_number = start // page_size + 1
        page_name = _page_filename(page_number)

        for entry in ordered[start:start + page_size]:
            keywords = ", ".join(entry.keywords)

            lines.append(
                "| "
                f"{_index_cell(entry.id)} | "
                f"{page_name} | "
                f"{_index_cell(entry.title)} | "
                f"{_index_cell(_index_summary(entry))} | "
                f"{_index_cell(keywords)} | "
                f"{entry.updated_at.isoformat()} | "
                f"{entry.status} |"
            )

    return "\n".join(lines).rstrip() + "\n"

# 给 LLM 看的内容页——包含该页所有记忆的完整 Markdown
def render_page_markdown(
    entries: Sequence[MemoryEntry],
) -> str:
    ordered = _ordered_entries(entries)

    if not ordered:
        return "# Memory Page\n\nNo memories.\n"

    rendered_entries = [
        render_entry_markdown(entry).rstrip()
        for entry in ordered
    ]

    return (
        "# Memory Page\n\n"
        + "\n\n---\n\n".join(rendered_entries)
        + "\n"
    )

# 给 LLM 看的用户偏好总览——只取 active 的 user 类记忆，有长度限制
def render_user_markdown(
    entries: Sequence[MemoryEntry],
    *,
    max_entries: int,
    max_chars: int,
) -> str:
    
    if max_entries < 1:
        raise ValueError("max_entries must be positive")

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    # 投影是给 LLM 实时用的，superseded 的记忆不应该出现在 USER.md 里。
    active_entries = [
        entry
        for entry in _ordered_entries(entries)
        if (
            entry.kind == "user"
            and entry.status == "active"
        )
    ][:max_entries]

    blocks = [
        "# User Memories",
        "",
    ]

    for entry in active_entries:
        keywords = ", ".join(entry.keywords)

        block = "\n".join(
            [
                f"## {_single_line(entry.title)}",
                f"- Memory key: {entry.memory_key}",
                f"- Summary: {_single_line(entry.summary)}",
                f"- When to read: {_READ_WHEN[entry.kind]}",
                f"- Keywords: {_single_line(keywords)}",
                "",
            ]
        )
        # 逐条追加，一旦加上这条就超长了，就停止。保证已追加的内容一定在限制内。
        candidate = "\n".join(blocks) + block

        if len(candidate) > max_chars:
            break

        blocks.append(block)

    rendered = "\n".join(blocks).rstrip() + "\n"

    if len(rendered) <= max_chars:
        return rendered

    return rendered[:max_chars].rstrip()
"""
Summary Archive
  -> Retrieval Summary
  -> Original Summary
  -> Topics
  -> Archive ID
  -> Thread ID
  -> Summary hash
  -> Version
  -> Archived at
"""

def summary_archive_filename(
    archive: SummaryArchive,
) -> str:
    identity = (
        f"{archive.thread_id}\0"
        f"{archive.summary_hash}"
    )

    safe_hash = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    return f"summary-{safe_hash}.md"


def summary_archive_path(
    archive: SummaryArchive,
) -> Path:
    return (
        projection_root()
        / "memory-internal"
        / "summary-archive"
        / summary_archive_filename(archive)
    )


def render_summary_archive_markdown(
    archive: SummaryArchive,
) -> str:
    topics = ", ".join(
        _single_line(topic)
        for topic in archive.topics
    )

    lines = [
        "# Summary Archive",
        "",
        "## Retrieval Summary",
        archive.retrieval_summary.strip(),
        "",
        "## Original Summary",
        archive.original_summary.strip(),
        "",
        "## Topics",
        topics,
        "",
        "## Metadata",
        f"- Archive ID: {archive.id}",
        f"- Thread ID: {archive.thread_id}",
        f"- Summary hash: {archive.summary_hash}",
        f"- Version: {archive.version}",
        f"- Archived at: {archive.archived_at.isoformat()}",
    ]

    return "\n".join(lines).rstrip() + "\n"

def render_summary_archive_recall_markdown(
    archive: SummaryArchive,
) -> str:
    retrieval_summary = archive.retrieval_summary.strip()

    if not retrieval_summary:
        return ""

    return retrieval_summary