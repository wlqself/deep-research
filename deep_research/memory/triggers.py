from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MemoryTriggerReason = Literal[
    "rule",
    "summary_changed",
    "report_saved",
    "project_decision_confirmed",
    "explicit_correction",
    "interval",
    "backlog",
]


class MemoryTriggerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool # 本轮对话是否成功
    answer: str # 助手回答内容

    rule_triggered: bool = False
    summary_changed: bool = False
    report_saved: bool = False
    project_decision_confirmed: bool = False
    explicit_correction: bool = False

    successful_turn_count: int = Field(
        ge=0,
    ) # 累计成功轮数
    review_interval: int = Field(
        gt=0,
    ) # 审查间隔（必须 > 0）

class MemoryTriggerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_review: bool
    reasons: list[MemoryTriggerReason]


def decide_memory_trigger(
    trigger_input: MemoryTriggerInput,
) -> MemoryTriggerDecision:
    # 短路守卫
    if (
        not trigger_input.success
        or not trigger_input.answer.strip()
    ):
        return MemoryTriggerDecision(
            should_review=False,
            reasons=[],
        )

    reasons: list[MemoryTriggerReason] = []
    # 收集显式原因
    checks = (
        ("rule", trigger_input.rule_triggered),
        ("summary_changed", trigger_input.summary_changed),
        ("report_saved", trigger_input.report_saved),
        (
            "project_decision_confirmed",
            trigger_input.project_decision_confirmed,
        ),
        (
            "explicit_correction",
            trigger_input.explicit_correction,
        ),
    )

    for reason, enabled in checks:
        if enabled and reason not in reasons:
            reasons.append(reason)
    # 周期性兜底
    if (
        trigger_input.successful_turn_count > 0
        and trigger_input.successful_turn_count
        % trigger_input.review_interval
        == 0
    ):
        reasons.append("interval")

    return MemoryTriggerDecision(
        should_review=bool(reasons),
        reasons=reasons,
    )

def has_explicit_correction(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized.startswith(("如果", "假如", "若")):
        return False
    markers = (
        "不对",
        "不正确",
        "错了",
        "更正",
        "纠正一下",
        "应该是",
        "请改为",
        "改成",
        "that's wrong",
        "correction",
        "should be",
        "change it to",
    )

    return any(marker in normalized for marker in markers)

def has_confirmed_project_decision(text: str) -> bool:
    normalized = text.strip().lower()

    confirmation_markers = (
        "确认",
        "确定",
        "最终决定",
        "就按",
        "以此为准",
        "定了",
        "confirm",
        "we decided",
        "final decision",
    )

    decision_context_markers = (
        "项目",
        "方案",
        "架构",
        "设计",
        "约束",
        "实现",
        "技术路线",
        "project",
        "architecture",
        "design",
        "implementation",
    )

    strong_decision_markers = (
        "采用",
        "使用",
        "决定采用",
        "按这个方案",
        "按此实现",
        "adopt",
        "use this architecture",
    )

    has_confirmation = any(
        marker in normalized
        for marker in confirmation_markers
    )
    has_context = any(
        marker in normalized
        for marker in decision_context_markers
    )
    has_strong_decision = any(
        marker in normalized
        for marker in strong_decision_markers
    )
    acknowledgement_markers = (
        "收到",
        "了解",
        "明白",
        "noted",
        "got it",
    )

    has_acknowledgement = any(
        marker in normalized
        for marker in acknowledgement_markers
    )
    if has_acknowledgement and not has_strong_decision:
        return False

    return has_context and (
        has_strong_decision
        or has_confirmation
    )

def has_memory_rule_signal(text: str) -> bool:
    normalized = text.strip().lower()

    markers = (
        "请记住",
        "记住这一点",
        "以后请",
        "以后都",
        "今后请",
        "今后都",
        "默认",
        "我喜欢",
        "我偏好",
        "我不喜欢",
        "我的习惯",
        "不要再",
        "请不要再",
        "必须",
        "约束是",
        "please remember",
        "my preference",
        "i prefer",
        "i like",
        "i dislike",
        "do not do this again",
        "by default",
    )

    return any(
        marker in normalized
        for marker in markers
    )
