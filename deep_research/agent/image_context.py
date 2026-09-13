"""Build hidden, deterministic image instructions for the Main Agent."""

from ..context import ResearchContext


def user_message_with_image_ids(
    content: str,
    attachment_ids: tuple[str, ...] = (),
    *,
    display_content: str | None = None,
) -> dict[str, object]:
    """Build the persisted user message and keep image refs with that message."""

    message: dict[str, object] = {
        "role": "user",
        "content": content,
    }
    normalized_ids = tuple(
        dict.fromkeys(
            attachment_id.strip()
            for attachment_id in attachment_ids
            if isinstance(attachment_id, str) and attachment_id.strip()
        )
    )
    if normalized_ids:
        message["additional_kwargs"] = {
            "attachment_ids": list(normalized_ids),
        }
        if isinstance(display_content, str) and display_content.strip():
            message["additional_kwargs"]["display_content"] = display_content
    return message


def image_context_instruction(context: ResearchContext) -> str:
    """Return an instruction only when this conversation has image refs."""

    current = tuple(dict.fromkeys(context.selected_attachment_ids))
    historical = tuple(
        attachment_id
        for attachment_id in dict.fromkeys(context.conversation_attachment_ids)
        if attachment_id not in current
    )
    if not current and not historical:
        return ""

    sections: list[str] = [
        "[系统图片附件上下文：图片附件属于用户上传的共享资源，不是工作区文件。",
    ]
    if current:
        sections.append("本轮新附加图片 ID：" + ", ".join(current) + "。")
    if historical:
        sections.append(
            "本对话历史图片 ID（即使旧消息已被摘要，也仍可引用）："
            + ", ".join(historical)
            + "。"
        )
    sections.extend(
        [
            "当用户要求理解、提取、解题、描述或核对图片，且当前可见上下文没有足够图片内容时，必须调用 analyze_uploaded_image，"
            "并只传入上述确切 ID；普通追问使用 mode=\"cached\"，优先读取该图片已绑定的识别结果。",
            "当用户质疑之前的识别、说图片描述不对、要求重新看或核对局部细节时，使用 mode=\"fresh\" 强制重新调用视觉模型。",
            "如果本轮和历史都没有图片 ID，不要调用图片分析工具。不要向用户展示附件 ID。]",
        ]
    )
    return "\n".join(sections)


__all__ = ["image_context_instruction", "user_message_with_image_ids"]
