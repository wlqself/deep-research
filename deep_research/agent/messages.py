from typing import Any

# ========== 消息内容解析工具 ==========
# 兼容不同模型返回的 content 格式（字符串 / 多段文本）

def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text")

                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts).strip()

    return ""