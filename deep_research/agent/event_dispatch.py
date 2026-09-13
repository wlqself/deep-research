from typing import Any

# 提取event中的元数据
def unpack_stream_event(
    event: dict[str, object],
) -> tuple[dict[str, object], bool, object, str, object]:
    metadata = event.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}
    # 判断这个事件是否来自 researcher 子 agent
    is_researcher_event = metadata.get("lc_agent_name") == "researcher"
    event_name = event.get("event")
    tool_name = str(event.get("name", ""))
    event_data = event.get("data", {})

    return (
        metadata,
        is_researcher_event,
        event_name,
        tool_name,
        event_data,
    )
