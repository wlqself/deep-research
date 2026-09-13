"""Deterministic Markdown formatting and pre-publication checks.

This module does not render HTML or call an LLM. Source Artifacts remain
untouched; normalized Markdown belongs only to Article drafts and revisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_HEADING_PATTERN = re.compile(r"^\s{0,3}(#{1,6})\s*(.*?)\s*#*\s*$")
_UNORDERED_LIST_PATTERN = re.compile(r"^\s*[-*+]\s+(.+?)\s*$")
_ORDERED_LIST_PATTERN = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$")
_BLOCKQUOTE_PATTERN = re.compile(r"^\s*>\s?(.*?)\s*$")
_FENCE_PATTERN = re.compile(r"^\s*```([^`]*)\s*$")
_HORIZONTAL_RULE_PATTERN = re.compile(
    r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$"
)


@dataclass(frozen=True)
class MarkdownFormatIssue:
    """One stable, content-free Markdown preflight finding."""

    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class MarkdownFormatReport:
    """Preflight findings for one Markdown snapshot."""

    issues: tuple[MarkdownFormatIssue, ...] = ()

    @property
    def is_publishable(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_code(self) -> str | None:
        return next(
            (issue.code for issue in self.issues if issue.severity == "error"),
            None,
        )


def normalize_article_markdown(markdown: str) -> str:
    """Return canonical Markdown while preserving code-block content verbatim."""

    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")

    source = unicodedata.normalize("NFC", markdown).lstrip("\ufeff")
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    output: list[tuple[str, bool]] = []
    in_code_block = False

    for raw_line in source.split("\n"):
        fence = _FENCE_PATTERN.fullmatch(raw_line)
        if in_code_block:
            output.append((raw_line, True))
            if fence:
                in_code_block = False
            continue
        if fence:
            output.append((f"```{fence.group(1).strip()}", False))
            in_code_block = True
            continue

        line = raw_line.replace("\t", "    ").rstrip()
        if not line.strip():
            output.append(("", False))
            continue
        heading = _HEADING_PATTERN.fullmatch(line)
        if heading and heading.group(2).strip():
            output.append((f"{heading.group(1)} {heading.group(2).strip()}", False))
            continue
        unordered_item = _UNORDERED_LIST_PATTERN.fullmatch(line)
        if unordered_item:
            output.append((f"- {unordered_item.group(1)}", False))
            continue
        ordered_item = _ORDERED_LIST_PATTERN.fullmatch(line)
        if ordered_item:
            output.append((f"{ordered_item.group(1)}. {ordered_item.group(2)}", False))
            continue
        blockquote = _BLOCKQUOTE_PATTERN.fullmatch(line)
        if blockquote:
            output.append((f"> {blockquote.group(1)}".rstrip(), False))
            continue
        if _HORIZONTAL_RULE_PATTERN.fullmatch(line):
            output.append(("---", False))
            continue
        output.append((line, False))

    normalized: list[str] = []
    previous_blank = True
    for line, is_code_content in output:
        if is_code_content:
            normalized.append(line)
            previous_blank = False
            continue
        is_blank = not line
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank
    while normalized and not normalized[-1] and not in_code_block:
        normalized.pop()
    return "\n".join(normalized) + ("\n" if normalized else "")


def check_article_markdown(markdown: str) -> MarkdownFormatReport:
    """Check publication invariants without returning article content."""

    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")

    normalized = normalize_article_markdown(markdown)
    issues: list[MarkdownFormatIssue] = []
    if not normalized.strip():
        return MarkdownFormatReport((MarkdownFormatIssue(
            code="article_markdown_empty",
            severity="error",
            message="Article Markdown must contain publishable content.",
        ),))

    in_code_block = False
    heading_levels: list[int] = []
    for line in normalized.splitlines():
        if _FENCE_PATTERN.fullmatch(line):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        heading = _HEADING_PATTERN.fullmatch(line)
        if heading:
            heading_levels.append(len(heading.group(1)))

    if in_code_block:
        issues.append(MarkdownFormatIssue(
            code="article_markdown_unclosed_code_fence",
            severity="error",
            message="Article Markdown contains an unclosed code fence.",
        ))
    if not heading_levels:
        issues.append(MarkdownFormatIssue(
            code="article_markdown_missing_heading",
            severity="warning",
            message="Article Markdown has no heading.",
        ))
    elif heading_levels[0] != 1:
        issues.append(MarkdownFormatIssue(
            code="article_markdown_missing_h1",
            severity="warning",
            message="Article Markdown does not begin with a level-one heading.",
        ))
    for previous, current in zip(heading_levels, heading_levels[1:]):
        if current > previous + 1:
            issues.append(MarkdownFormatIssue(
                code="article_markdown_heading_level_jump",
                severity="warning",
                message="Article Markdown skips a heading level.",
            ))
            break
    return MarkdownFormatReport(tuple(issues))


__all__ = [
    "MarkdownFormatIssue",
    "MarkdownFormatReport",
    "check_article_markdown",
    "normalize_article_markdown",
]
