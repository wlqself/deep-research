# 判断是否来自researcher并且事件名是否正确
def is_summarization_end(
    event_name: object,
    event_metadata: dict[str, object],
) -> bool:
    return (
        event_name == "on_chat_model_end"
        and event_metadata.get("lc_source") == "summarization"
    )

# 过滤 researcher 子 agent 和非 model 节点的事件，从合法事件的 chunk 中提取 .text 属性并返回
def model_text_from_stream(
    event_data: object,
    *,
    is_researcher_event: bool,
    event_metadata: dict[str, object],
) -> str:
    if not isinstance(event_data, dict):
        return ""

    if (
        is_researcher_event
        or event_metadata.get("langgraph_node") != "model"
    ):
        return ""

    chunk = event_data.get("chunk")
    text = getattr(chunk, "text", "")

    return text if isinstance(text, str) else ""
