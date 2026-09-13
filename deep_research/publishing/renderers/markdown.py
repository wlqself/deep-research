"""Small, dependency-free Markdown renderer for published articles.

The renderer intentionally supports a conservative Markdown subset.  Input is
HTML-escaped before any markup is generated, so raw HTML is displayed as text
and cannot execute in the generated site.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from urllib.parse import urlsplit

from ..inline_images import ATTACHMENT_IMAGE_PATTERN


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_UNORDERED_ITEM_PATTERN = re.compile(r"^\s*[-*+]\s+(.+)$")
_ORDERED_ITEM_PATTERN = re.compile(r"^\s*\d+[.)]\s+(.+)$")
_BLOCKQUOTE_PATTERN = re.compile(r"^\s*>\s?(.*)$")
_FENCE_PATTERN = re.compile(r"^\s*```(?:[^`]*)$")
_HORIZONTAL_RULE_PATTERN = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)"
)
_CODE_SPAN_PATTERN = re.compile(r"`([^`]+)`")
_STRONG_PATTERN = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_EMPHASIS_PATTERN = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)")

_ALLOWED_EXTERNAL_SCHEMES = frozenset({"http", "https", "mailto"})


def _safe_href(value: str) -> tuple[str, bool] | None:
    candidate = html.unescape(value).strip()
    if not candidate or any(
        character in candidate for character in "\r\n\x00"
    ):
        return None

    parsed = urlsplit(candidate)
    if parsed.scheme:
        if parsed.scheme.casefold() not in _ALLOWED_EXTERNAL_SCHEMES:
            return None
        return candidate, True

    if candidate.startswith("//") or candidate.startswith("\\"):
        return None

    return candidate, False


def _inline_html(
    text: str,
    *,
    image_renderer: Callable[[str, str], str] | None = None,
) -> str:
    image_values: list[str] = []

    def stash_image(match: re.Match[str]) -> str:
        attachment_id = match.group("attachment_id")
        alt = match.group("alt").strip()
        if image_renderer is not None:
            rendered = image_renderer(attachment_id, alt)
        else:
            rendered = (
                '<img src="attachment://'
                + html.escape(attachment_id, quote=True)
                + '" alt="'
                + html.escape(alt, quote=True)
                + '" data-attachment-id="'
                + html.escape(attachment_id, quote=True)
                + '">'
            )
        image_values.append(rendered)
        return f"\x00IMAGE{len(image_values) - 1}\x00"

    # Extract before escaping so the attachment URI remains a typed reference,
    # not a user-controlled HTML attribute.
    text = ATTACHMENT_IMAGE_PATTERN.sub(stash_image, text)
    escaped = html.escape(text, quote=False)

    code_values: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code_values.append(f"<code>{html.escape(match.group(1), quote=False)}</code>")
        return f"\x00CODE{len(code_values) - 1}\x00"

    escaped = _CODE_SPAN_PATTERN.sub(stash_code, escaped)

    def replace_link(match: re.Match[str]) -> str:
        safe_target = _safe_href(match.group(2))
        if safe_target is None:
            return match.group(1)

        href, external = safe_target
        attributes = f' href="{html.escape(href, quote=True)}"'
        if external:
            attributes += ' target="_blank" rel="noopener noreferrer"'
        return f"<a{attributes}>{match.group(1)}</a>"

    escaped = _LINK_PATTERN.sub(replace_link, escaped)
    escaped = _STRONG_PATTERN.sub(
        lambda match: f"<strong>{match.group(1) or match.group(2)}</strong>",
        escaped,
    )
    escaped = _EMPHASIS_PATTERN.sub(
        lambda match: f"<em>{match.group(1) or match.group(2)}</em>",
        escaped,
    )

    for index, value in enumerate(code_values):
        escaped = escaped.replace(f"\x00CODE{index}\x00", value)
    for index, value in enumerate(image_values):
        escaped = escaped.replace(f"\x00IMAGE{index}\x00", value)

    return escaped


def _flush_paragraph(
    output: list[str],
    paragraph: list[str],
    *,
    image_renderer: Callable[[str, str], str] | None = None,
) -> None:
    if not paragraph:
        return
    output.append(
        f"<p>{_inline_html(' '.join(paragraph), image_renderer=image_renderer)}</p>"
    )
    paragraph.clear()


def _close_list(output: list[str], list_kind: str | None) -> str | None:
    if list_kind is not None:
        output.append(f"</{list_kind}>")
    return None


def render_markdown(
    markdown: str,
    *,
    image_renderer: Callable[[str, str], str] | None = None,
) -> str:
    """Render safe HTML from a conservative Markdown subset."""

    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")

    output: list[str] = []
    paragraph: list[str] = []
    list_kind: str | None = None
    in_code_block = False
    code_lines: list[str] = []

    def flush_paragraph_and_list() -> None:
        nonlocal list_kind
        _flush_paragraph(
            output,
            paragraph,
            image_renderer=image_renderer,
        )
        list_kind = _close_list(output, list_kind)

    for line in markdown.splitlines():
        if in_code_block:
            if _FENCE_PATTERN.fullmatch(line):
                output.append(
                    "<pre><code>"
                    f"{html.escape(chr(10).join(code_lines), quote=False)}"
                    "</code></pre>"
                )
                code_lines.clear()
                in_code_block = False
            else:
                code_lines.append(line)
            continue

        if _FENCE_PATTERN.fullmatch(line):
            flush_paragraph_and_list()
            in_code_block = True
            continue

        if not line.strip():
            flush_paragraph_and_list()
            continue

        heading = _HEADING_PATTERN.fullmatch(line)
        if heading:
            flush_paragraph_and_list()
            level = len(heading.group(1))
            output.append(
                f"<h{level}>{_inline_html(heading.group(2), image_renderer=image_renderer)}</h{level}>"
            )
            continue

        if _HORIZONTAL_RULE_PATTERN.fullmatch(line):
            flush_paragraph_and_list()
            output.append("<hr>")
            continue

        unordered = _UNORDERED_ITEM_PATTERN.fullmatch(line)
        if unordered:
            _flush_paragraph(
                output,
                paragraph,
                image_renderer=image_renderer,
            )
            if list_kind != "ul":
                list_kind = _close_list(output, list_kind)
                output.append("<ul>")
                list_kind = "ul"
            output.append(
                f"<li>{_inline_html(unordered.group(1), image_renderer=image_renderer)}</li>"
            )
            continue

        ordered = _ORDERED_ITEM_PATTERN.fullmatch(line)
        if ordered:
            _flush_paragraph(
                output,
                paragraph,
                image_renderer=image_renderer,
            )
            if list_kind != "ol":
                list_kind = _close_list(output, list_kind)
                output.append("<ol>")
                list_kind = "ol"
            output.append(
                f"<li>{_inline_html(ordered.group(1), image_renderer=image_renderer)}</li>"
            )
            continue

        blockquote = _BLOCKQUOTE_PATTERN.fullmatch(line)
        if blockquote:
            flush_paragraph_and_list()
            output.append(
                "<blockquote><p>"
                f"{_inline_html(blockquote.group(1), image_renderer=image_renderer)}"
                "</p></blockquote>"
            )
            continue

        if list_kind is not None:
            list_kind = _close_list(output, list_kind)
        paragraph.append(line.strip())

    if in_code_block:
        output.append(
            "<pre><code>"
            f"{html.escape(chr(10).join(code_lines), quote=False)}"
            "</code></pre>"
        )

    _flush_paragraph(output, paragraph, image_renderer=image_renderer)
    _close_list(output, list_kind)
    return "\n".join(output)


__all__ = ["render_markdown"]
