"""Bound visible Main-Agent output and stop deterministic repetition loops."""

from __future__ import annotations

import re


_WHITESPACE = re.compile(r"\s+")


class StreamOutputGuard:
    def __init__(
        self,
        *,
        max_chars: int,
        repetition_min_chars: int,
        repetition_count: int,
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        if repetition_min_chars < 1:
            raise ValueError("repetition_min_chars must be positive")
        if repetition_count < 2:
            raise ValueError("repetition_count must be at least two")

        self.max_chars = max_chars
        self.repetition_min_chars = repetition_min_chars
        self.repetition_count = repetition_count
        self._text = ""

    def add(self, chunk: str) -> str | None:
        if not isinstance(chunk, str) or not chunk:
            return None

        self._text += chunk
        if len(self._text) > self.max_chars:
            return "model_output_limit_exceeded"

        normalized = _WHITESPACE.sub(" ", self._text).strip()
        repeat_count = self.repetition_count
        maximum_pattern = min(
            2_000,
            len(normalized) // repeat_count,
        )
        if maximum_pattern < self.repetition_min_chars:
            return None

        for pattern_size in range(
            maximum_pattern,
            self.repetition_min_chars - 1,
            -1,
        ):
            repeated_size = pattern_size * repeat_count
            suffix = normalized[-repeated_size:]
            pattern = suffix[-pattern_size:]
            if suffix == pattern * repeat_count:
                return "model_repetition_detected"

        return None


__all__ = ["StreamOutputGuard"]
