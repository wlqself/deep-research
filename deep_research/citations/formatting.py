import re
from collections.abc import Mapping

from ..sources import PersistentSource


_SOURCE_PATTERN = re.compile(r"\[S(\d+)\]")

# 提取所有引用ID（去重） 
def referenced_source_ids(answer: str) -> list[str]:
    referenced_ids: list[str] = []

    for match in _SOURCE_PATTERN.finditer(answer):
        source_id = f"S{match.group(1)}"

        if source_id not in referenced_ids:
            referenced_ids.append(source_id)

    return referenced_ids

# 所有在答案中被引用，但不存在于来源字典中的ID（即幻觉引用）
def invalid_source_ids(
    answer: str,
    sources: Mapping[str, PersistentSource],
) -> list[str]:
    return [
        source_id
        for source_id in referenced_source_ids(answer)
        if source_id not in sources
    ]

# 检查答案中是否已经存在“来源”标题（如## 来源、来源:等）
def _has_sources_heading(answer: str) -> bool:
    return re.search(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?来源[ \t]*:?[\t ]*$",
        answer,
    ) is not None

# 增加引用
def append_verified_sources(
    answer: str,
    sources: Mapping[str, PersistentSource],
) -> str:
    # 1. 过滤出“既被引用、又真实存在”的ID
    referenced_ids = [
        source_id
        for source_id in referenced_source_ids(answer)
        if source_id in sources
    ]
    # 2. 如果没有有效引用，或已经存在来源标题，直接返回原答案
    if not referenced_ids or _has_sources_heading(answer):
        return answer.rstrip()
    # 3. 构建来源列表
    source_lines = ["", "来源"]

    for source_id in referenced_ids:
        source = sources[source_id]

        source_lines.append(
            f"[{source_id}] "
            f"{source['title']} - "
            f"{source['url']}"
        )
    # 4. 拼接最终答案
    return (
        f"{answer.rstrip()}\n"
        + "\n".join(source_lines)
    )