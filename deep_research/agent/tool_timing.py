import time


def mark_tool_started(
    started_at: dict[str, float],
    run_id: str,
) -> None:
    started_at[run_id] = time.perf_counter()


def take_tool_elapsed(
    started_at: dict[str, float],
    run_id: str,
) -> float | None:
    started = started_at.pop(run_id, None)

    if started is None:
        return None

    return round((time.perf_counter() - started) * 1000, 3)
