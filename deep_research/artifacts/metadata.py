import re
from uuid import uuid4

# 把标题转成安全的文件名前缀
def _safe_title_stem(title: str) -> str:
    stem = re.sub(
        r"[^0-9A-Za-z\u4e00-\u9fff]+",
        "-",
        title.strip()
    ).strip("-")

    return stem[:60] or "research_report"

# 构建产物的完整元数据
def build_artifact_metadata(
        title: str,
        artifact_id:str | None = None,
) -> dict[str, str]:
    normalized_title = title.strip()
    
    artifact_id = artifact_id or uuid4().hex
    if not normalized_title:
        raise ValueError("title must not be empty")

    filename = (
        f"{_safe_title_stem(normalized_title)}"
        f"-{artifact_id[:12]}.md"
    )

    return {
        "artifact_id": artifact_id,
        "filename" : filename,
        "workspace_path": f"/final/{filename}",
    }