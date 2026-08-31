from typing import Literal
from typing_extensions import TypedDict

FindingStatus = Literal[
    "supported", # 现有证据足以支持结论
    "conflicted", # 源之间存在明确冲突
    "insufficient", # 当前证据不足。
]

# ResearchFinding 保存：
# 结论 + 简短证据摘要 + source_id 关系 + 状态

# ResearchFinding 不保存：
# 完整网页正文、隐藏推理、模型思维过程、其他线程数据
class ResearchFinding(TypedDict):
    finding_id: str # 账本条目的唯一 ID。应使用不可预测 ID，例如 uuid4().hex，不能使用 S1 这类来源编号
    claim: str  # 准备支持的明确结论。
    evidence_summary: str # 支持该结论的简短证据概括。
    source_ids: list[str] # 相关来源 ID 列表，例如 ["S1", "S3"]。必须非空，且之后由工具校验这些来源属于当前线程。
    status: FindingStatus
    uncertainty: str # 仍然存在的不确定性。没有内容时使用空字符串。
    conflicts: str # 来源冲突说明。没有冲突时使用空字符串。
    created_at: str # 首次创建时间，使用带时区的 ISO 8601。
    updated_at: str # 最近一次更新的时间，也使用带时区的 ISO 8601。

    